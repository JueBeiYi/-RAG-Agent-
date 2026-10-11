from __future__ import annotations

import json
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from src.core.config import load_settings
from src.core.schemas import ChatRequest
from src.services.application import Application


class SessionCreate(BaseModel):
    title: str = Field(default="新对话", max_length=80)


class MetadataUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=500)
    authors: list[str] | None = None
    year: int | None = Field(default=None, ge=1800, le=2100)
    abstract: str | None = Field(default=None, max_length=20000)
    doi: str | None = Field(default=None, max_length=300)
    source_url: str | None = Field(default=None, max_length=2000)
    aliases: list[str] | None = None


class SessionUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=80)
    document_ids: list[str] | None = None


def create_app(settings=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.service = Application(settings or load_settings())
        yield
        app.state.service.close()

    app = FastAPI(title="论文科研助理 API", version="0.1.0", lifespan=lifespan)

    def service():
        return app.state.service

    def scope(x_workspace_token: Annotated[str | None, Header()] = None):
        if not x_workspace_token or not re.fullmatch(r"[A-Za-z0-9_-]{16,128}", x_workspace_token):
            raise HTTPException(401, "请提供有效的 X-Workspace-Token（至少16位随机标识）")
        return x_workspace_token

    def require(table, key, scope):
        record = service().store.get(table, key, scope)
        if not record:
            raise HTTPException(404, "记录不存在")
        return record

    def public_document(doc):
        data = doc.model_dump() if hasattr(doc, "model_dump") else dict(doc)
        data.pop("path", None)
        data.pop("scope", None)
        return data

    @app.get("/health")
    def health():
        return service().health()

    @app.get("/config")
    def config():
        return {**service().settings.public_dict(), "demo": service().settings.llm_backend == "extractive"}

    @app.post("/documents")
    async def upload(files: Annotated[list[UploadFile], File()], workspace: str = Depends(scope)):
        if len(files) > 20:
            raise HTTPException(400, "每次最多上传20份文件")
        received = []
        total_bytes = 0
        for file in files:
            filename = (file.filename or "file").replace("\\", "/").rsplit("/", 1)[-1]
            if Path(filename).suffix.lower() not in {".pdf", ".docx", ".txt", ".md", ".markdown"}:
                raise HTTPException(400, "文件类型不支持")
            content = await file.read(service().settings.max_upload_mb * 1024 * 1024 + 1)
            if not content or len(content) > service().settings.max_upload_mb * 1024 * 1024:
                raise HTTPException(413, "文件为空或超过大小限制")
            total_bytes += len(content)
            if total_bytes > 200 * 1024 * 1024:
                raise HTTPException(413, "单批文件总大小不能超过200MB")
            received.append((filename, content))
        jobs, documents = [], []
        for filename, content in received:
            try:
                result = service().upload(filename, content, workspace)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            documents.append({**public_document(result["document"]), "duplicate": result["duplicate"]})
            if result["job"]:
                jobs.append(result["job"])
        return {"jobs": jobs, "documents": documents}

    @app.get("/jobs/{job_id}")
    def job(job_id: str, workspace: str = Depends(scope)):
        return require("jobs", job_id, workspace)

    @app.get("/jobs")
    def jobs(workspace: str = Depends(scope)):
        return service().store.list("jobs", workspace, 200)

    @app.get("/documents")
    def documents(workspace: str = Depends(scope)):
        return [public_document(d) for d in service().store.list_documents(workspace)]

    @app.get("/documents/{document_id}")
    def document(document_id: str, workspace: str = Depends(scope)):
        record = require("documents", document_id, workspace)
        if record["format"] == "pdf" and not record.get("metadata", {}).get("page_count"):
            import fitz
            path = Path(record["path"]).resolve()
            if path.is_relative_to(service().settings.path / "documents") and path.is_file():
                try:
                    with fitz.open(path) as source:
                        record.setdefault("metadata", {})["page_count"] = len(source)
                        record["metadata"]["preview_available"] = not source.needs_pass
                except (RuntimeError, ValueError):
                    record.setdefault("metadata", {}).update(page_count=0, preview_available=False,
                                                              preview_error="PDF损坏或无法读取，请检查原件后重新上传")
        return public_document(record)

    @app.get("/documents/{document_id}/chunks")
    def document_chunks(document_id: str, offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=100),
                        workspace: str = Depends(scope)):
        require("documents", document_id, workspace)
        chunks = sorted(service().store.list_chunks(workspace, [document_id]),
                        key=lambda c: (c.page_start or 0, c.paragraph or 0,
                                       c.bbox[1] if c.bbox else 0, c.chunk_id))
        values = [c.model_dump(exclude={"scope"}) for c in chunks[offset:offset+limit]]
        return {"items": values, "total": len(chunks), "offset": offset, "limit": limit}

    @app.post("/documents/{document_id}/metadata")
    def update_metadata(document_id: str, body: MetadataUpdate, workspace: str = Depends(scope)):
        require("documents", document_id, workspace)
        with service().store.lock:
            doc = service().store.get_document(document_id, workspace)
            if not doc or doc.status in {"queued", "processing"}:
                raise HTTPException(409, "请等文档处理完成后核对元信息")
            updates = {key: value for key, value in body.model_dump(exclude_unset=True).items()
                       if value is not None or key == "year"}
            if not updates:
                raise HTTPException(400, "请填写需要修改的元信息")
            source = updates.pop("source_url", None)
            aliases = updates.pop("aliases", None)
            if aliases is not None:
                if len(aliases) > 20 or any(len(value) > 100 for value in aliases):
                    raise HTTPException(400, "论文别名超出限制")
                doc.metadata["aliases"] = aliases
            if any(len(author) > 300 for author in updates.get("authors", [])) or len(updates.get("authors", [])) > 100:
                raise HTTPException(400, "作者列表超出限制")
            for key, value in updates.items():
                setattr(doc, key, value)
            doc.metadata.update(metadata_reviewed=True, metadata_source=source or "user_correction")
            service().store.put_document(doc)
            service().store.bump_version(workspace)
            return public_document(doc)

    @app.delete("/documents/{document_id}")
    def delete_document(document_id: str, workspace: str = Depends(scope)):
        require("documents", document_id, workspace)
        try:
            service().delete_document(document_id, workspace)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"deleted": document_id}

    @app.post("/documents/{document_id}/retry")
    def retry(document_id: str, workspace: str = Depends(scope)):
        require("documents", document_id, workspace)
        try:
            return service().retry(document_id, workspace)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/documents/{document_id}/reindex")
    def reindex(document_id: str, workspace: str = Depends(scope)):
        require("documents", document_id, workspace)
        try:
            return service().retry(document_id, workspace, reindex=True)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/documents/{document_id}/file")
    def original(document_id: str, workspace: str = Depends(scope)):
        record = require("documents", document_id, workspace)
        path = Path(record["path"]).resolve()
        if not path.is_relative_to(service().settings.path / "documents") or not path.is_file():
            raise HTTPException(404, "原件不存在")
        return FileResponse(path, filename=record["name"])

    @app.get("/documents/{document_id}/preview")
    def preview(document_id: str, page: int = 1, chunk_id: str = "", workspace: str = Depends(scope)):
        record = require("documents", document_id, workspace)
        if record["format"] != "pdf":
            raise HTTPException(400, "非 PDF 请使用段落证据查看")
        path = Path(record["path"]).resolve()
        if not path.is_relative_to(service().settings.path / "documents") or not path.is_file():
            raise HTTPException(404, "原件不存在")
        import fitz
        try:
            pdf = fitz.open(path)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(400, "PDF无法预览，请检查原件是否损坏或需要解密") from exc
        with pdf:
            if pdf.needs_pass:
                raise HTTPException(400, "PDF受密码保护，请解密后重新上传")
            if not 1 <= page <= len(pdf):
                raise HTTPException(400, "页码超出范围")
            target = pdf[page-1]
            if chunk_id:
                chunk = next((c for c in service().store.list_chunks(workspace, [document_id]) if c.chunk_id == chunk_id), None)
                if chunk:
                    for block in chunk.sources:
                        if block.page == page and block.bbox:
                            rectangle = fitz.Rect(block.bbox) & target.rect
                            if not rectangle.is_empty and not rectangle.is_infinite:
                                target.add_highlight_annot(rectangle)
            scale = min(1.5, 1800 / max(target.rect.width, target.rect.height, 1))
            pix = target.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            return Response(pix.tobytes("png"), media_type="image/png")

    @app.post("/sessions")
    def create_session(body: SessionCreate, workspace: str = Depends(scope)):
        return service().store.create_session(workspace, body.title)

    @app.get("/sessions")
    def sessions(workspace: str = Depends(scope)):
        return [{k: v for k, v in s.items() if k not in {"messages", "memory"}} for s in service().store.list("sessions", workspace)]

    @app.get("/sessions/{session_id}")
    def session(session_id: str, workspace: str = Depends(scope)):
        record = require("sessions", session_id, workspace)
        record["active_request"] = next((r for r in service().live_runs(workspace) if r["session_id"] == session_id), None)
        return record

    @app.patch("/sessions/{session_id}")
    def patch_session(session_id: str, body: SessionUpdate, workspace: str = Depends(scope)):
        require("sessions", session_id, workspace)
        with service().store.lock:
            if service()._session_lock(session_id, workspace).locked():
                raise HTTPException(409, "会话正在执行，请稍后修改")
            record = require("sessions", session_id, workspace)
            if body.title is not None:
                if not body.title.strip():
                    raise HTTPException(400, "会话名称不能为空")
                record["title"] = body.title.strip()
                record["title_custom"] = True
            if body.document_ids is not None:
                for identifier in body.document_ids:
                    if service().store.get_document(identifier, workspace) is None:
                        raise HTTPException(400, "论文不存在或不属于当前工作区")
                record["document_ids"] = list(dict.fromkeys(body.document_ids))
            service().store.put("sessions", session_id, workspace, record)
            return record

    @app.get("/sessions/{session_id}/export")
    def session_export(session_id: str, format: str = "md", workspace: str = Depends(scope)):
        from src.services.exporting import export_session
        record = require("sessions", session_id, workspace)
        if format not in {"md", "json"}:
            raise HTTPException(400, "仅支持 md/json")
        return Response(export_session(record, format), media_type="application/json" if format == "json" else "text/markdown; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="conversation-{session_id[:8]}.{format}"'})

    @app.post("/sessions/{session_id}/retry-last")
    def retry_last(session_id: str, workspace: str = Depends(scope)):
        record = require("sessions", session_id, workspace)
        last_user = next((m for m in reversed(record["messages"]) if m["role"] == "user"), None)
        if last_user is None:
            raise HTTPException(400, "会话中没有可以重新发送的问题")
        request = ChatRequest(message=last_user["content"], document_ids=last_user.get("document_ids", record.get("document_ids", [])),
                              retrieval_mode=last_user.get("retrieval_mode"), agent=last_user.get("agent", True))
        def generate():
            for item in service().chat(session_id, workspace, request, retry_last=True):
                yield "data: " + json.dumps(item, ensure_ascii=False, default=str) + "\n\n"
        return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.delete("/sessions/{session_id}")
    def delete_session(session_id: str, workspace: str = Depends(scope)):
        require("sessions", session_id, workspace)
        if service()._session_lock(session_id, workspace).locked():
            raise HTTPException(409, "会话正在执行，请稍后删除")
        service().store.remove("sessions", session_id, workspace)
        return {"deleted": session_id}

    @app.post("/sessions/{session_id}/chat")
    def chat(session_id: str, body: ChatRequest, workspace: str = Depends(scope)):
        require("sessions", session_id, workspace)
        if body.retrieval_mode not in {None, "vector", "hybrid", "hybrid_rerank"}:
            raise HTTPException(400, "检索模式无效")
        def generate():
            for item in service().chat(session_id, workspace, body):
                yield "data: " + json.dumps(item, ensure_ascii=False, default=str) + "\n\n"
        return StreamingResponse(generate(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/runs")
    def runs(workspace: str = Depends(scope)):
        records = service().store.list("runs", workspace, 200)
        return service().live_runs(workspace) + [{k: v for k, v in run.items() if k not in {"events", "configuration", "answer", "citations"}} for run in records]

    @app.get("/runs/{run_id}")
    def run(run_id: str, workspace: str = Depends(scope)):
        live = next((r for r in service().live_runs(workspace) if r["id"] == run_id), None)
        return live or require("runs", run_id, workspace)

    @app.post("/runs/{run_id}/cancel")
    def cancel_run(run_id: str, workspace: str = Depends(scope)):
        try:
            return service().cancel_request(run_id, workspace)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/runs/{run_id}/export")
    def run_export(run_id: str, format: str = "json", workspace: str = Depends(scope)):
        from src.services.exporting import export_run
        record = require("runs", run_id, workspace)
        if format not in {"json", "md"}:
            raise HTTPException(400, "仅支持 json/md")
        return Response(export_run(record, format), media_type="application/json" if format == "json" else "text/markdown; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="run-{run_id[:8]}.{format}"'})

    @app.delete("/runs/{run_id}")
    def delete_run(run_id: str, workspace: str = Depends(scope)):
        if any(r["id"] == run_id for r in service().live_runs(workspace)):
            raise HTTPException(409, "请等待当前执行结束后删除记录")
        record = require("runs", run_id, workspace)
        service().store.remove("runs", run_id, workspace)
        log = (service().settings.path / "logs" / (record["request_id"] + ".json")).resolve()
        if log.is_relative_to(service().settings.path / "logs"):
            log.unlink(missing_ok=True)
        return {"deleted": run_id}

    @app.post("/experiments")
    async def experiment(file: Annotated[UploadFile, File()], mode: Annotated[str, Form()] = "hybrid_rerank",
                         split: Annotated[str, Form()] = "test", workspace: str = Depends(scope)):
        from src.evaluation.dataset import load_cases
        if mode not in {"vector", "hybrid", "hybrid_rerank"} or split not in {"dev", "test", "all"}:
            raise HTTPException(400, "评测模式或集合无效")
        raw = await file.read(5 * 1024 * 1024 + 1)
        if len(raw) > 5 * 1024 * 1024:
            raise HTTPException(413, "评测文件超过5MB")
        tmp = service().settings.path / ("cases_" + __import__("uuid").uuid4().hex + ".jsonl")
        try:
            tmp.write_bytes(raw)
            cases = load_cases(tmp, split=None if split == "all" else split)
            if not cases:
                raise ValueError("所选集合没有题目")
            return service().run_experiment(workspace, cases, mode)
        except (ValueError, UnicodeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            tmp.unlink(missing_ok=True)

    @app.get("/experiments")
    def experiments(workspace: str = Depends(scope)):
        return [{k: v for k, v in e.items() if k != "result"} for e in service().store.list("experiments", workspace)]

    @app.get("/experiments/{experiment_id}")
    def experiment_result(experiment_id: str, workspace: str = Depends(scope)):
        return require("experiments", experiment_id, workspace)

    @app.delete("/experiments/{experiment_id}")
    def delete_experiment(experiment_id: str, workspace: str = Depends(scope)):
        record = require("experiments", experiment_id, workspace)
        if record["status"] in {"queued", "running"}:
            raise HTTPException(409, "请等待实验结束后删除记录")
        service().store.remove("experiments", experiment_id, workspace)
        return {"deleted": experiment_id}

    @app.get("/experiments/{experiment_id}/export")
    def export(experiment_id: str, format: str = "json", workspace: str = Depends(scope)):
        from src.evaluation.runner import serialize_export
        record = require("experiments", experiment_id, workspace)
        if record["status"] != "done":
            raise HTTPException(409, "实验尚未成功完成")
        if format not in {"json", "csv", "xlsx"}:
            raise HTTPException(400, "仅支持 json/csv/xlsx")
        types = {"json": "application/json", "csv": "text/csv", "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}
        return Response(serialize_export(record["result"], format), media_type=types[format],
                        headers={"Content-Disposition": f'attachment; filename="{experiment_id}.{format}"'})
    return app


app = create_app()
