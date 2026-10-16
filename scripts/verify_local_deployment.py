"""Verify an actual local 127.0.0.1 deployment with real models and a real PDF.

No Docker calls, process claims, service starts, or restarts occur here. The
parent launcher captures process, listening-port, and application log evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import secrets
import sys
import time
import traceback
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.verify_docker_deployment import Acceptance, now


class LocalAcceptance(Acceptance):
    def __init__(self, args):
        # The shared recorder uses standard-library HTTP and redaction only.
        # Compose metadata is removed and no Docker command method is called.
        args.project = "local-127-acceptance"
        args.profile = "standard"
        args.docker = "unused-local-http-only"
        args.compose = None
        args.restart = False
        super().__init__(args)
        self.report.pop("project", None)
        self.report.pop("commands", None)
        self.report.update(
            deployment="local-loopback",
            acceptance_scope="Actual local API/UI health, real-model readiness, real PDF parsing/indexing, retrieval, SSE generation, citations, original-file hash, PDF physical-page preview, saved conversation/run and export. Process and port evidence is captured separately by the local launcher; no restart is claimed.",
            transport="HTTP against loopback URLs; fresh private workspace",
        )

    def execute(self):
        for name, value in (("api", self.args.api), ("ui", self.args.ui)):
            parsed = urllib.parse.urlparse(value)
            self.check(f"{name}-is-local-loopback", parsed.scheme in {"http", "https"} and parsed.hostname in {"127.0.0.1", "localhost", "::1"})
        source = Path(self.args.document).resolve()
        self.check("real-source-pdf-exists", source.is_file() and source.suffix.lower() == ".pdf", {"filename": source.name})
        health = self.wait_ready("initial")
        deadline = time.monotonic() + min(self.args.timeout, 120)
        while True:
            status, _, _ = self.http("frontend-health", "/_stcore/health", base=self.args.ui, required=False, timeout=10)
            if status == 200 or time.monotonic() >= deadline:
                self.check("frontend-ready", status == 200)
                break
            time.sleep(2)
        _, config, _ = self.http("public-config", "/config")
        self.check("real-model-and-retrieval-backends", not health.get("demo") and not health.get("retrieval", {}).get("demo")
                   and health.get("llm", {}).get("backend") in {"ollama", "cloud"}
                   and config.get("llm_backend") != "extractive", health)
        raw = source.read_bytes()
        self.report["document"] = {"filename": source.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                                   "kind": "real research-paper PDF"}
        boundary = "AcceptanceBoundary" + secrets.token_hex(16)
        filename = source.name.replace('"', "_").replace("\r", "_").replace("\n", "_")
        media = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
        multipart = (f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="{filename}"\r\n'
                     f'Content-Type: {media}\r\n\r\n').encode() + raw + f"\r\n--{boundary}--\r\n".encode()
        _, imported, _ = self.http("upload", "/documents", method="POST", body=multipart,
                                   content_type=f"multipart/form-data; boundary={boundary}")
        doc_id = imported["documents"][0]["document_id"]
        self.report["document"]["document_id"] = doc_id
        deadline = time.monotonic() + self.args.timeout
        for job in imported["jobs"]:
            identifier = job.get("job_id") or job["id"]
            while True:
                _, state, _ = self.http("indexing-job", f"/jobs/{identifier}")
                if state.get("status") in {"done", "error"}:
                    self.check("indexing-completed", state["status"] == "done", state)
                    break
                self.check("indexing-within-time-budget", time.monotonic() < deadline, state)
                time.sleep(2)
        _, doc, _ = self.http("document-ready", f"/documents/{doc_id}")
        self.check("parsed-and-indexed", doc.get("status") == "ready" and doc.get("chunks", 0) > 0, doc)
        before_chunks = self.chunks(doc_id, "chunks-before")
        self.check("chunk-text-real-and-readable", bool(before_chunks) and all(c.get("text", "").strip() for c in before_chunks))
        _, session, _ = self.http("create-session", "/sessions", method="POST", body={"title": "本地127实际运行验收"})
        session_id = session["id"]
        question = self.args.question or "请根据所选论文，用中文简要概述核心贡献，并给出至少一个[S1]形式的来源引用。"
        _, sse, _ = self.http("rag-stream", f"/sessions/{session_id}/chat", method="POST",
                              body={"message": question, "document_ids": [doc_id], "agent": False,
                                    "retrieval_mode": "hybrid_rerank"})
        events = [json.loads(line[5:].strip()) for line in sse.splitlines() if line.startswith("data:")]
        self.save("rag-events.json", events)
        done = next((e for e in reversed(events) if e.get("type") == "done"), None)
        self.check("sse-completed-without-error", done is not None and not any(e.get("type") == "error" for e in events), events[-3:])
        self.check("answer-streamed", any(e.get("type") == "answer_delta" and e.get("text") for e in events)
                   and bool(done.get("answer", "").strip()))
        self.check("retrieval-returned-real-chunks", done.get("metrics", {}).get("retrieved_count", 0) > 0 and bool(done.get("hits")))
        self.check("answer-from-real-model-with-evidence", done.get("origin") == "knowledge_rag"
                   and not done.get("metrics", {}).get("demo") and done.get("metrics", {}).get("output_tokens", 0) > 0,
                   {"origin": done.get("origin"), "metrics": done.get("metrics")})
        citations = done.get("citations", [])
        by_id = {c["chunk_id"]: c for c in before_chunks}
        self.check("citations-point-to-persisted-chunks", bool(citations) and all(c.get("document_id") == doc_id and c.get("chunk_id") in by_id for c in citations),
                   [{k: c.get(k) for k in ("citation_id", "document_id", "chunk_id", "page_start", "paragraph")} for c in citations])
        self.check("answer-cites-available-sources", bool(done.get("audit", {}).get("used_ids")) and not done.get("audit", {}).get("invalid_ids"), done.get("audit"))
        _, original, _ = self.http("source-download", f"/documents/{doc_id}/file", binary=True)
        self.check("original-file-hash-matches", original["sha256"] == hashlib.sha256(raw).hexdigest(), original)
        if source.suffix.lower() == ".pdf":
            citation = citations[0]
            page = citation.get("page_start") or 1
            _, preview, preview_bytes = self.http("citation-page-preview", f"/documents/{doc_id}/preview?page={page}&chunk_id={citation['chunk_id']}", binary=True)
            self.check("citation-preview-is-png", preview["content_type"].startswith("image/png") and preview["bytes"] > 100)
            (self.folder / "citation-page-preview.png").write_bytes(preview_bytes)
        self.report["answer"] = {k: done.get(k) for k in ("answer", "origin", "metrics", "audit")}
        run_id = done.get("request_id")
        self.check("request-id-recorded", bool(run_id))
        _, persisted_run, _ = self.http("persisted-run", f"/runs/{run_id}")
        self.check("run-record-saved", persisted_run.get("status") == "done" and persisted_run.get("answer") == done["answer"])
        _, persisted_session, _ = self.http("persisted-session", f"/sessions/{session_id}")
        self.check("conversation-saved", any(m.get("role") == "assistant" and m.get("content") == done["answer"]
                   for m in persisted_session.get("messages", [])))
        self.http("session-export", f"/sessions/{session_id}/export?format=json")
        self.report["status"] = "passed"

    def finish(self):
        self.report["finished_at"] = now()
        self.report["passed_check_count"] = sum(bool(c["passed"]) for c in self.report["checks"])
        self.report["failed_check_count"] = sum(not c["passed"] for c in self.report["checks"])
        self.save("acceptance.json", self.report)
        lines = ["# 本地127实际运行验收记录", "", f"- 状态：`{self.report['status']}`", f"- 开始：{self.report['started_at']}",
                 f"- 结束：{self.report['finished_at']}", f"- API：{self.args.api}", f"- 前端：{self.args.ui}",
                 "- 模式：standard，要求真实模型和真实检索后端，非demo。",
                 f"- 通过检查：{self.report['passed_check_count']}；失败检查：{self.report['failed_check_count']}", "",
                 "记录来自本轮实际HTTP请求；不包含容器或重启验收。进程、监听端口和应用日志由启动器另行采集。", "",
                 "| 验收检查 | 实测状态 |", "|---|---|"]
        lines += [f"| {item['name']} | {'通过' if item['passed'] else '失败'} |" for item in self.report["checks"]]
        lines += ["", "同目录保存逐次HTTP响应、完整SSE事件、PDF引用物理页预览、文档SHA-256、实际模型和检索配置及会话/调用记录。",
                  "workspace token仅在内存中，身份认证和密钥字段已脱敏；独立验收工作区数据保留供复查。",
                  "该验收证明本地核心流程实际可用，不等同于完整题集质量评测、浏览器视觉验收或重启持久性测试。"]
        if self.report.get("error"):
            lines += ["", "实际失败原因：" + str(self.redact(self.report["error"]))]
        (self.folder / "验收记录.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--ui", default="http://127.0.0.1:8501")
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--document", required=True, help="An existing real text-layer research-paper PDF")
    parser.add_argument("--question", help="A question grounded in the PDF; otherwise asks for a concise contribution summary")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args(argv)
    run = LocalAcceptance(args)
    try:
        run.execute()
    except Exception as exc:
        run.report.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        print(run.redact(str(exc)), file=sys.stderr, flush=True)
    finally:
        run.finish()
    print(f"Evidence: {run.folder / 'acceptance.json'}", flush=True)
    return 0 if run.report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
