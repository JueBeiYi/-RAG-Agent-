"""Collect real Docker deployment acceptance evidence; never assume success.

Uses Python's standard library. Run against a separately owned Compose project.
HTTP payloads and Docker output are redacted before saving. Workspace tokens
exist only in memory; a SHA-256 identifier is retained in the public report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import re
import secrets
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


class Acceptance:
    def __init__(self, args):
        self.args = args
        self.folder = Path(args.evidence_dir).resolve()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.token = secrets.token_urlsafe(32)
        self.secret_values = [self.token]
        for key, value in os.environ.items():
            if value and len(value) >= 8 and re.search(r"API_KEY|ACCESS_TOKEN|PASSWORD|SECRET", key, re.I):
                self.secret_values.append(value)
        self.report = {
            "started_at": now(), "status": "running", "profile": args.profile,
            "project": args.project, "api_url": args.api, "ui_url": args.ui,
            "workspace_sha256": hashlib.sha256(self.token.encode()).hexdigest(),
            "scope_policy": "A newly generated private workspace; never uses production workspace tokens.",
            "acceptance_scope": "Real container runtime, ingestion, retrieval, generation, citation navigation, and restart persistence. Not a fresh model quality benchmark or browser visual audit.",
            "checks": [], "commands": [], "requests": [],
        }
        self.compose = [args.docker, "compose", "--project-name", args.project]
        for path in args.compose or [str(ROOT / "docker/docker-compose.yml")]:
            self.compose += ["-f", str(Path(path).resolve())]
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def redact(self, value):
        if isinstance(value, dict):
            return {str(k): ("[REDACTED]" if re.search(r"(^scope$|api.?key|password|authorization|secret|workspace.?token)", str(k), re.I)
                             else self.redact(v)) for k, v in value.items()}
        if isinstance(value, list):
            return [self.redact(v) for v in value]
        if isinstance(value, str):
            for secret in self.secret_values:
                value = value.replace(secret, "[REDACTED]")
            value = re.sub(r"(?i)(https?://)[^\s/@]+:[^\s/@]+@", r"\1[REDACTED]@", value)
            value = re.sub(r"(?i)((?:api[_-]?key|access[_-]?token|password|secret|x-workspace-token)[\s\"'=:\-]+)[^\s\"',}\]]+", r"\1[REDACTED]", value)
        return value

    def save(self, name, value):
        path = self.folder / name
        path.write_text(json.dumps(self.redact(value), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def check(self, name, passed, detail=None):
        item = {"name": name, "passed": bool(passed), "at": now(), "detail": self.redact(detail)}
        self.report["checks"].append(item)
        self.save("acceptance.json", self.report)
        print(f"{'PASS' if passed else 'FAIL'} {name}", flush=True)
        if not passed:
            raise RuntimeError(f"Acceptance check failed: {name}; {self.redact(detail)}")

    def command(self, name, argv, *, required=True, timeout=120):
        started = time.perf_counter()
        try:
            proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=timeout, check=False)
            code, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        except (OSError, subprocess.TimeoutExpired) as exc:
            code, stdout, stderr = -1, "", str(exc)
        entry = {"name": name, "command": argv, "at": now(), "exit_code": code,
                 "seconds": round(time.perf_counter() - started, 3), "stdout": stdout, "stderr": stderr}
        self.save(f"command-{name}.json", entry)
        self.report["commands"].append({k: entry[k] for k in ("name", "command", "at", "exit_code", "seconds")})
        if required:
            self.check(name, code == 0, {"exit_code": code, "stderr": stderr[-1500:]})
        return code, stdout

    def http(self, name, path, *, method="GET", body=None, binary=False, base=None,
             content_type=None, required=True, timeout=None):
        headers = {"X-Workspace-Token": self.token}
        payload = body
        if body is not None and not isinstance(body, bytes):
            payload = json.dumps(body, ensure_ascii=False).encode()
            content_type = "application/json"
        if content_type:
            headers["Content-Type"] = content_type
        url = (base or self.args.api).rstrip("/") + path
        request = urllib.request.Request(url, data=payload, headers=headers, method=method)
        started = time.perf_counter()
        status, raw, media = 0, b"", ""
        try:
            with self.opener.open(request, timeout=timeout or self.args.timeout) as response:
                status, raw, media = response.status, response.read(), response.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            status, raw, media = exc.code, exc.read(), exc.headers.get("Content-Type", "")
        except (OSError, urllib.error.URLError) as exc:
            raw = str(exc).encode()
        text = raw.decode("utf-8", errors="replace")
        if binary:
            result = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "content_type": media}
        else:
            try:
                result = json.loads(text)
            except ValueError:
                result = text
        entry = {"name": name, "method": method, "url": url, "at": now(),
                 "status": status, "seconds": round(time.perf_counter() - started, 3),
                 "request": {"file_bytes": len(body)} if isinstance(body, bytes) else body,
                 "content_type": media, "response": result}
        self.save(f"http-{name}.json", entry)
        self.report["requests"].append({k: entry[k] for k in ("name", "method", "url", "status", "seconds", "at")})
        if required:
            self.check(name, 200 <= status < 300, {"status": status, "response": result if status >= 300 else None})
        return status, result, raw

    def wait_ready(self, phase):
        deadline = time.monotonic() + self.args.timeout
        last = None
        attempts = 0
        while time.monotonic() < deadline:
            attempts += 1
            status, last, _ = self.http(f"{phase}-health-poll", "/health", required=False, timeout=10)
            if status == 200 and isinstance(last, dict) and last.get("alive") and last.get("ready"):
                self.check(f"{phase}-backend-ready", True, {"attempts": attempts, "health": last})
                return last
            time.sleep(3)
        self.check(f"{phase}-backend-ready", False, last)

    def inventory(self, phase):
        self.command(f"{phase}-compose-ps", self.compose + ["ps", "--all", "--format", "json"])
        _, stdout = self.command(f"{phase}-container-ids", self.compose + ["ps", "-q"])
        ids = stdout.split()
        self.check(f"{phase}-containers-found", bool(ids))
        # Inspect only selected public fields: Config.Env may contain credentials.
        template = '{{json .Id}}|{{json .Name}}|{{json .Config.Labels}}|{{json .State}}|{{json .Mounts}}|{{json .Image}}'
        _, stdout = self.command(f"{phase}-inspect", [self.args.docker, "inspect", "--format", template, *ids])
        rows = []
        for line in stdout.splitlines():
            values = line.split("|")
            if len(values) == 6:
                identity, name, labels, state, mounts, image = map(json.loads, values)
                rows.append({"id": identity, "name": name, "labels": labels, "state": state, "mounts": mounts, "image": image})
        self.save(f"containers-{phase}.json", rows)
        owned = [r for r in rows if r["labels"].get("com.docker.compose.project") == self.args.project]
        self.check(f"{phase}-isolated-compose-project", len(owned) == len(ids), {"containers": len(owned)})
        for service in ("api", "frontend"):
            matches = [r for r in owned if r["labels"].get("com.docker.compose.service") == service]
            self.check(f"{phase}-{service}-running", len(matches) == 1 and matches[0]["state"].get("Running"), matches)
        api = next((r for r in owned if r["labels"].get("com.docker.compose.service") == "api"), None)
        if api and api["state"].get("Health", {}).get("Status") == "starting":
            deadline = time.monotonic() + min(self.args.timeout, 90)
            while time.monotonic() < deadline:
                time.sleep(3)
                _, state = self.command(f"{phase}-healthcheck-poll", [self.args.docker, "inspect", "--format", "{{json .State}}", api["id"]])
                api["state"] = json.loads(state)
                if api["state"].get("Health", {}).get("Status") != "starting":
                    break
            self.save(f"containers-{phase}.json", rows)
        self.check(f"{phase}-api-healthcheck", api is not None and api["state"].get("Health", {}).get("Status") == "healthy")
        self.check(f"{phase}-persistent-app-volume", any(r["labels"].get("com.docker.compose.service") == "api" and
                   any(m.get("Type") in {"volume", "bind"} and m.get("Destination") == "/app/data/runtime" for m in r["mounts"]) for r in owned))
        return rows

    def chunks(self, doc_id, name):
        rows, offset = [], 0
        while True:
            _, response, _ = self.http(f"{name}-{offset}", f"/documents/{doc_id}/chunks?offset={offset}&limit=100")
            rows.extend(response["items"])
            offset += len(response["items"])
            if offset >= response["total"] or not response["items"]:
                return rows

    def execute(self):
        self.command("docker-version", [self.args.docker, "version"])
        self.command("docker-info", [self.args.docker, "info", "--format", "{{json .}}"])
        self.command("compose-version", [self.args.docker, "compose", "version"])
        self.command("compose-config", self.compose + ["config", "--format", "json"])
        self.command("compose-images", self.compose + ["images", "--format", "json"])
        health = self.wait_ready("initial")
        self.inventory("before")
        self.http("frontend-health", "/_stcore/health", base=self.args.ui)
        _, config, _ = self.http("public-config", "/config")
        if self.args.profile == "standard":
            self.check("real-model-and-retrieval-backends", not health.get("demo") and not health.get("retrieval", {}).get("demo")
                       and health.get("llm", {}).get("backend") in {"ollama", "cloud"}
                       and config.get("llm_backend") != "extractive", health)
        else:
            self.check("explicit-demo-mode", health.get("demo") is True, health)
        source = Path(self.args.document).resolve() if self.args.document else self.folder / "docker-acceptance-fixture.txt"
        if not self.args.document:
            source.write_text("Docker部署验收专用文档\n\n此文档用于验证真实上传、解析、索引、问答与重启持久性。\n"
                              "验收实验设定的试验温度为23摄氏度，试验样本数量为47份。\n"
                              "所有试验样本经过统一预处理，再按随机顺序执行测量。\n"
                              "注意：这些数字是部署验收夹具中的设定，不是课程论文研究结果。\n", encoding="utf-8")
        raw = source.read_bytes()
        self.report["document"] = {"filename": source.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                                   "kind": "user-selected-source" if self.args.document else "explicit synthetic operational fixture, not a research paper"}
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
        _, session, _ = self.http("create-session", "/sessions", method="POST", body={"title": "Docker实际部署验收"})
        session_id = session["id"]
        question = self.args.question or ("请根据所选论文，用中文简要概述核心贡献，并给出至少一个[S1]形式的来源引用。" if self.args.document
                                         else "根据文档，部署验收的试验样本数量和试验温度分别是多少？请附来源编号。")
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
        if self.args.profile == "standard":
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
        if run_id:
            self.http("persisted-run", f"/runs/{run_id}")
        self.http("session-export", f"/sessions/{session_id}/export?format=json")
        if self.args.restart:
            self.check("restart-target-is-dedicated-acceptance-project", self.args.project == "nau-rag-acceptance" or
                       bool(re.fullmatch(r"(?:rag|course)-acceptance-[a-z0-9_-]+", self.args.project)))
            self.command("restart-owned-api-frontend", self.compose + ["restart", "api", "frontend"], timeout=self.args.timeout)
            self.wait_ready("after-restart")
            self.inventory("after")
            # Streamlit can take slightly longer than the backend to bind its port.
            deadline = time.monotonic() + min(self.args.timeout, 90)
            while True:
                status, _, _ = self.http("after-restart-frontend-health", "/_stcore/health", base=self.args.ui, required=False, timeout=10)
                if status == 200 or time.monotonic() >= deadline:
                    self.check("after-restart-frontend-ready", status == 200)
                    break
                time.sleep(2)
            _, persisted, _ = self.http("document-after-restart", f"/documents/{doc_id}")
            after_chunks = self.chunks(doc_id, "chunks-after")
            self.check("document-persisted-after-restart", persisted.get("status") == "ready" and persisted.get("content_hash") == doc.get("content_hash"))
            canonical = lambda rows: sorted((r["chunk_id"], r["text"]) for r in rows)
            self.check("all-chunks-persisted-after-restart", canonical(before_chunks) == canonical(after_chunks))
            _, persisted_session, _ = self.http("session-after-restart", f"/sessions/{session_id}")
            self.check("conversation-persisted-after-restart", any(m.get("role") == "assistant" and m.get("content") == done["answer"] for m in persisted_session.get("messages", [])))
            if run_id:
                _, persisted_run, _ = self.http("run-after-restart", f"/runs/{run_id}")
                self.check("run-persisted-after-restart", persisted_run.get("status") == "done" and persisted_run.get("answer") == done["answer"])
            _, persisted_file, _ = self.http("original-after-restart", f"/documents/{doc_id}/file", binary=True)
            self.check("original-persisted-after-restart", persisted_file["sha256"] == hashlib.sha256(raw).hexdigest())
        self.report["status"] = ("passed" if self.args.profile == "standard" and self.args.restart
                                 else "demo_control_flow_passed" if self.args.profile == "demo" else "partial_restart_not_tested")

    def finish(self):
        self.command("final-compose-logs", self.compose + ["logs", "--no-color", "--tail", "250"], required=False)
        self.report["finished_at"] = now()
        self.report["passed_check_count"] = sum(bool(c["passed"]) for c in self.report["checks"])
        self.report["failed_check_count"] = sum(not c["passed"] for c in self.report["checks"])
        self.save("acceptance.json", self.report)
        lines = ["# Docker实际部署验收记录", "", f"- 状态：`{self.report['status']}`", f"- 开始：{self.report['started_at']}",
                 f"- 结束：{self.report['finished_at']}", f"- Compose项目：`{self.args.project}`", f"- 模式：`{self.args.profile}`",
                 f"- 通过检查：{self.report['passed_check_count']}；失败检查：{self.report['failed_check_count']}", "",
                 "本记录由实际命令与HTTP请求生成。AI/模型答案质量仍需单独评价；demo模式不能证明真实模型部署通过。", "",
                 "| 验收检查 | 实测状态 |", "|---|---|"]
        lines += [f"| {item['name']} | {'通过' if item['passed'] else '失败'} |" for item in self.report["checks"]]
        lines += ["", "逐次HTTP请求、SSE事件、容器状态、版本、镜像、Compose配置与日志保存在同目录。workspace token及API密钥已脱敏。",
                  "未删除验收数据及Docker卷，便于复查；只会在明确启用 --restart 时重启指定验收项目的 api/frontend。"]
        (self.folder / "验收记录.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--ui", default="http://127.0.0.1:8501")
    parser.add_argument("--compose", action="append", help="Repeat in Compose override order")
    parser.add_argument("--project", required=True, help="Dedicated project: nau-rag-acceptance or rag-acceptance-YYYYMMDD")
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--profile", choices=("standard", "demo"), default="standard")
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--timeout", type=int, default=900, help="Maximum seconds for readiness, indexing, and individual model request")
    parser.add_argument("--document", help="Existing real PDF/DOCX/TXT/MD; omitted creates clearly labelled operational fixture")
    parser.add_argument("--question", help="Question grounded in the selected document")
    parser.add_argument("--restart", action="store_true", help="Verify persistence by restarting only this project's api/frontend")
    args = parser.parse_args(argv)
    run = Acceptance(args)
    try:
        run.execute()
    except Exception as exc:
        run.report.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        print(run.redact(str(exc)), file=sys.stderr, flush=True)
    finally:
        run.finish()
    print(f"Evidence: {run.folder / 'acceptance.json'}", flush=True)
    return 0 if run.report["status"] in {"passed", "demo_control_flow_passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
