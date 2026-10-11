from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.core.config import Settings
from src.core.schemas import Document, event
from src.core.storage import Storage, now
from src.generation.llm_client import LLMClient
from src.generation.rag_pipeline import RAGEngine
from src.generation.context import normalise_page_references
from src.observability.logger import RequestLogger


class Application:
    def __init__(self, settings: Settings, recover_jobs: bool = True):
        from src.retrieval.engine import Retriever
        from src.agent.react_loop import ResearchAgent
        self.settings = settings
        self.store = Storage(settings.path)
        self.retriever = Retriever(settings, self.store)
        self.llm = LLMClient(settings)
        self.rag = RAGEngine(settings, self.store, self.retriever, self.llm)
        self.agent = ResearchAgent(settings, self.store, self.llm, self.rag)
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingestion")
        self.pending = threading.BoundedSemaphore(30)
        self.index_lock = threading.RLock()
        self.upload_lock = threading.RLock()
        self.session_locks = {}
        self.locks_lock = threading.Lock()
        self.logger = RequestLogger(settings.path / "logs")
        self.active_requests = {}
        self.active_lock = threading.Lock()
        # Interrupted tasks never masquerade as successful; the UI exposes retry.
        with self.store.connect() as con:
            records = con.execute("SELECT id,scope,payload FROM jobs").fetchall() if recover_jobs else []
        for key, scope, raw in records:
            record = json.loads(raw)
            if record.get("status") in {"queued", "running"}:
                record.update(status="error", error="服务重启中断任务，请重试", stage="interrupted")
                self.store.put("jobs", key, scope, record)
                doc = self.store.get_document(record.get("document_id", ""), scope)
                if doc and doc.status != "ready":
                    doc.status, doc.error = "error", record["error"]
                    self.store.put_document(doc)
        if recover_jobs:
            with self.store.connect() as con:
                old_documents = con.execute("SELECT scope,payload FROM documents").fetchall()
            affected = set()
            for scope, payload in old_documents:
                doc = Document.model_validate_json(payload)
                identity = {"embedding_fingerprint": self.retriever.embedder.fingerprint,
                            "chunk_strategy": settings.chunk_strategy, "chunk_size": settings.chunk_size,
                            "chunk_overlap": settings.chunk_overlap, "pdf_parser": settings.pdf_parser,
                            "chunker_implementation": "recursive-character-proposals-v1"}
                if doc.status == "ready" and any(doc.metadata.get(k) != v for k, v in identity.items()):
                    doc.status, doc.error = "error", "模型权重或解析/分块配置已变更，请在论文库重新索引"
                    self.store.put_document(doc)
                    affected.add(scope)
            for scope in affected:
                self.store.bump_version(scope)

    def close(self):
        self.worker.shutdown(wait=False, cancel_futures=True)
        if hasattr(self.agent, "close"):
            self.agent.close()

    def _job(self, scope, job_id, **values):
        job = self.store.get("jobs", job_id, scope) or {"job_id": job_id, "id": job_id, "created_at": now()}
        job.update(values, updated_at=now())
        self.store.put("jobs", job_id, scope, job)
        return job

    def upload(self, name: str, content: bytes, scope: str):
        with self.upload_lock:
            return self._upload(name, content, scope)

    def _upload(self, name: str, content: bytes, scope: str):
        suffix = Path(name).suffix.lower()
        if suffix not in {".pdf", ".docx", ".txt", ".md", ".markdown"}:
            raise ValueError("仅支持 PDF、DOCX、TXT、Markdown")
        if len(content) > self.settings.max_upload_mb * 1024 * 1024 or not content:
            raise ValueError("文件为空或超过上传大小限制")
        content_hash = hashlib.sha256(content).hexdigest()
        existing = next((d for d in self.store.list_documents(scope) if d.content_hash == content_hash), None)
        if existing:
            return {"document": existing.model_dump(), "duplicate": True, "job": None}
        if not self.pending.acquire(blocking=False):
            raise ValueError("导入队列已满，请稍后重试")
        document_id = uuid.uuid4().hex
        folder = self.settings.path / "documents" / hashlib.sha256(scope.encode()).hexdigest()[:24]
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (document_id + suffix)
        try:
            path.write_bytes(content)
            document = Document(document_id=document_id, scope=scope, name=name[:180], content_hash=content_hash,
                                format=suffix.lstrip("."), path=str(path), created_at=now())
            self.store.put_document(document)
            job_id = uuid.uuid4().hex
            job = self._job(scope, job_id, document_id=document_id, name=name, status="queued", stage="queued", progress=0)
            self.worker.submit(self._ingest, document_id, scope, job_id)
            return {"document": document.model_dump(), "job": job, "duplicate": False}
        except Exception:
            self.pending.release()
            raise

    def retry(self, document_id, scope, reindex=False):
        with self.upload_lock:
            return self._retry(document_id, scope, reindex)

    def _retry(self, document_id, scope, reindex=False):
        doc = self.store.get_document(document_id, scope)
        if not doc:
            raise KeyError("文档不存在")
        if doc.status != "error" and not (reindex and doc.status == "ready"):
            raise ValueError("只能重试失败或中断的导入任务")
        if not self.pending.acquire(blocking=False):
            raise ValueError("任务队列已满")
        job_id = uuid.uuid4().hex
        doc.status, doc.error = "queued", ""
        self.store.put_document(doc)
        self.store.bump_version(scope)
        job = self._job(scope, job_id, document_id=document_id, name=doc.name, status="queued", stage="queued", progress=0)
        self.worker.submit(self._ingest, document_id, scope, job_id)
        return job

    def _ingest(self, document_id, scope, job_id):
        from src.data_loader.loaders import load_blocks, extract_metadata
        from src.chunking.strategies import split_blocks
        try:
            with self.index_lock:
                doc = self.store.get_document(document_id, scope)
                if not doc:
                    raise ValueError("文档已删除")
                doc.status = "processing"
                self.store.put_document(doc)
                self._job(scope, job_id, status="running", stage="parsing", progress=10)
                blocks = load_blocks(doc.path, self.settings.pdf_parser)
                if not blocks or not any(b.text.strip() for b in blocks):
                    raise ValueError("未提取到文本；扫描 PDF 需要先进行 OCR")
                metadata = extract_metadata(blocks, doc.name)
                for key in ("title", "authors", "year", "abstract", "doi"):
                    if key in metadata and not doc.metadata.get("metadata_reviewed"):
                        setattr(doc, key, metadata[key])
                if doc.format == "pdf":
                    import fitz
                    with fitz.open(doc.path) as original:
                        doc.metadata["page_count"] = len(original)
                self._job(scope, job_id, stage="chunking", progress=35)
                chunks = split_blocks(doc, blocks, self.settings.chunk_strategy,
                                      self.settings.chunk_size, self.settings.chunk_overlap)
                if not chunks:
                    raise ValueError("分块后没有有效文本")
                self._job(scope, job_id, stage="embedding", progress=55)
                self.retriever.index(chunks)
                self._job(scope, job_id, stage="committing", progress=90)
                doc.metadata.update(chunk_strategy=self.settings.chunk_strategy, chunk_size=self.settings.chunk_size,
                                    chunker_implementation="recursive-character-proposals-v1",
                                    chunk_overlap=self.settings.chunk_overlap, pdf_parser=self.settings.pdf_parser,
                                    embedding_fingerprint=self.retriever.embedder.fingerprint,
                                    embedding_model=self.settings.embedding_model, embedding_backend=self.settings.embedding_backend)
                self.store.commit_document(doc, chunks)
                self._job(scope, job_id, status="done", stage="ready", progress=100, chunks=len(chunks))
        except Exception as exc:
            doc = self.store.get_document(document_id, scope)
            if doc:
                doc.status, doc.error = "error", str(exc)[:1200]
                self.store.put_document(doc)
            self._job(scope, job_id, status="error", stage="failed", error=str(exc)[:1200])
        finally:
            self.pending.release()

    def delete_document(self, document_id, scope):
        with self.index_lock:
            doc = self.store.get_document(document_id, scope)
            if not doc:
                raise KeyError("文档不存在")
            if doc.status in {"queued", "processing"}:
                raise ValueError("请等待导入任务完成后删除")
            self.retriever.delete(document_id, scope)
            self.store.delete_document(document_id, scope)
            path = Path(doc.path).resolve()
            if path.is_relative_to(self.settings.path / "documents") and path.is_file():
                path.unlink()

    def _session_lock(self, session_id, scope):
        with self.locks_lock:
            return self.session_locks.setdefault((scope, session_id), threading.Lock())

    def live_runs(self, scope):
        with self.active_lock:
            return [{k: v for k, v in record.items() if k not in {"scope", "cancel"}}
                    for record in self.active_requests.values() if record["scope"] == scope]

    def cancel_request(self, request_id, scope):
        with self.active_lock:
            record = self.active_requests.get(request_id)
            if record is None or record["scope"] != scope:
                raise KeyError("当前没有该工作区可取消的请求")
            record["cancel"].set()
            record["status"] = "cancelling"
            return {"id": request_id, "status": "cancelling", "message": "停止请求已提交；当前调用释放后保存已完成部分"}

    def chat(self, session_id, scope, request, retry_last=False):
        lock = self._session_lock(session_id, scope)
        if not lock.acquire(blocking=False):
            yield event("error", text="同一会话已有请求执行中，请等待完成")
            return
        request_id, started, events, final = uuid.uuid4().hex, time.perf_counter(), [], None
        status, error_text, partial = "running", "", ""
        cancel, stream = threading.Event(), None
        with self.active_lock:
            self.active_requests[request_id] = {"id": request_id, "request_id": request_id, "scope": scope,
                                               "session_id": session_id, "status": "running", "cancel": cancel,
                                               "question": request.message, "created_at": now()}
        try:
            session = self.store.get("sessions", session_id, scope)
            if not session:
                raise ValueError("会话不存在")
            documents = request.document_ids if "document_ids" in request.model_fields_set else session.get("document_ids", [])
            for doc_id in documents:
                doc = self.store.get_document(doc_id, scope)
                if not doc or doc.status != "ready":
                    raise ValueError("所选论文不存在或索引尚未就绪")
            session["document_ids"] = documents
            if not session["messages"] and not session.get("title_custom"):
                session["title"] = request.message[:28]
            self.store.put("sessions", session_id, scope, session)
            messages = session["messages"]
            if not (retry_last and messages and messages[-1].get("role") == "user" and messages[-1].get("content") == request.message):
                self.store.add_message(session_id, scope, "user", request.message, document_ids=documents,
                                       retrieval_mode=request.retrieval_mode, agent=request.agent)
            if request.agent:
                stream = self.agent.stream(request.message, scope, session_id, documents, request.retrieval_mode)
            else:
                stream = self.rag.stream(request.message, scope, documents, request.retrieval_mode,
                                         self.store.list_messages(session_id, scope)[-7:-1])
            for item in stream:
                if cancel.is_set():
                    status, error_text = "cancelled", "用户已请求停止，当前流响应已结束"
                    cancelled = event("cancelled", text=error_text, request_id=request_id, answer=partial)
                    events.append(cancelled)
                    yield cancelled
                    break
                item = {**item, "request_id": request_id}
                events.append(item)
                if item["type"] == "answer_delta":
                    partial += item.get("text", "")
                if item["type"] == "done":
                    normalised, corrections = normalise_page_references(item.get("answer", partial), item.get("citations", []))
                    item["answer"] = normalised
                    if corrections:
                        item.setdefault("audit", {}).setdefault("page_reference_corrections", []).extend(corrections)
                    final = item
                    if status != "error":
                        status = "error" if item.get("origin") == "agent_error" else "done"
                    item.setdefault("metrics", {})["end_to_end_ms"] = (time.perf_counter()-started)*1000
                    self.store.add_message(session_id, scope, "assistant", item.get("answer", partial),
                                           citations=item.get("citations", []), metrics=item["metrics"],
                                           origin=item.get("origin"), request_id=request_id, audit=item.get("audit", {}))
                if item["type"] == "error":
                    status = "error"
                    error_text = item.get("text", "工具执行失败")
                yield item
            if final is None and status not in {"error", "cancelled"}:
                raise RuntimeError("模型未完成回答，请重试")
        except GeneratorExit:
            status, error_text = "cancelled", "客户端断开；保留已执行的记录"
            raise
        except Exception as exc:
            status, error_text = "error", str(exc)[:1200]
            item = event("error", text=error_text, request_id=request_id)
            events.append(item)
            yield item
        finally:
            if stream is not None and hasattr(stream, "close"):
                try:
                    stream.close()
                except Exception as exc:
                    error_text = (error_text + f" 流资源释放失败 ({type(exc).__name__})").strip()
            record = {"id": request_id, "request_id": request_id, "session_id": session_id, "question": request.message,
                      "created_at": now(), "status": status, "error": error_text, "events": events,
                      "answer": (final or {}).get("answer", partial), "citations": (final or {}).get("citations", []),
                      "metrics": (final or {}).get("metrics", {}), "origin": (final or {}).get("origin", ""),
                      "configuration": self.settings.public_dict(), "index_version": (final or {}).get("metrics", {}).get("index_version", self.store.index_version(scope)),
                      "elapsed_ms": (time.perf_counter()-started)*1000}
            try:
                self.store.put("runs", request_id, scope, record)
                try:
                    self.logger.write(request_id, record)
                except OSError:
                    # SQLite remains authoritative if the supplementary JSON log cannot be written.
                    pass
            finally:
                with self.active_lock:
                    self.active_requests.pop(request_id, None)
                lock.release()

    def run_experiment(self, scope, cases, mode):
        experiment_id = uuid.uuid4().hex
        record = {"id": experiment_id, "status": "queued", "created_at": now(), "mode": mode, "case_count": len(cases)}
        self.store.put("experiments", experiment_id, scope, record)
        def execute():
            from src.evaluation.runner import run_retrieval
            try:
                self.store.put("experiments", experiment_id, scope, {**record, "status": "running"})
                with self.index_lock:
                    result = run_retrieval(self.retriever, self.store, cases, scope, mode,
                                           self.settings.path / "experiments" / experiment_id, self.settings)
                self.store.put("experiments", experiment_id, scope, {**record, "status": "done", "result": result})
            except Exception as exc:
                self.store.put("experiments", experiment_id, scope, {**record, "status": "error", "error": str(exc)[:1200]})
        self.worker.submit(execute)
        return record

    def health(self):
        llm = self.llm.health()
        retrieval = self.retriever.health()
        retrieval.pop("last_search", None)
        return {"alive": True, "ready": bool(llm.get("ready") and retrieval.get("ready")),
                "llm": llm, "retrieval": retrieval, "storage": self.store.health(),
                "demo": self.settings.llm_backend == "extractive", "offline": self.settings.offline}
