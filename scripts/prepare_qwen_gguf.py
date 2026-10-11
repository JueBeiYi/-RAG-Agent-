"""Explicitly prepare official Qwen2.5 Q4_K_M GGUF shards for local Ollama.

No driver, service or permanent PATH changes are made. The default independent
tag research-qwen:7b avoids collisions with registry pulls. All model files,
license and reproducibility metadata stay under the ignored data/models folder.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "data" / "models" / "qwen-gguf"
MANIFEST = ROOT / "data" / "models" / "local-llm.json"
TEMPLATE_URL = "https://ollama.com/library/qwen2.5:7b-instruct-q4_K_M/blobs/eb4402837c78"
TEMPLATE_SHA256 = "eb4402837c7829a690fa845de4d7f3fd842c2adee476d5341da8a46ea9255175"
REPOSITORY = "Qwen/Qwen2.5-7B-Instruct-GGUF"
DEFAULT_REVISION = "bb5d59e06d9551d752d08b292a50eb208b07ab1f"
SHARDS = {
    "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf":
        (3993201344, "dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db"),
    "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf":
        (689872288, "539cf93f78e887edea1c04e2d7d8cdaca9d01dae9c9025bcb8accbe29df3d72a"),
}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def state():
    return json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}


def save(**values):
    record = {**state(), **values, "updated_at": stamp()}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(MANIFEST)
    return record


@contextmanager
def operation_lock():
    """Prevent two preparation commands mutating the same shards/manifest."""
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = DIRECTORY / ".prepare.lock"
    with path.open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("已有模型准备进程执行中；拒绝重复下载或并发导入") from exc
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def download(args):
    # Only this explicit operation permits network access to model artifacts.
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    # HTTP byte-range resume is predictable on this Windows/network setup.
    # The previous interrupted Xet download's incomplete bytes are retained.
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    info = api.model_info(REPOSITORY, revision=args.revision, files_metadata=True)
    if info.card_data.get("license") != "apache-2.0":
        raise ValueError("官方模型许可与已核实Apache-2.0不一致，请人工检查")
    official = {entry.rfilename: entry for entry in info.siblings}
    for name, (size, sha) in SHARDS.items():
        entry = official.get(name)
        if entry is None or entry.size != size or entry.lfs.sha256 != sha:
            raise ValueError(f"{name}: 官方固定revision的大小/SHA256与已核实值不一致")
    save(repository=REPOSITORY, revision=info.sha, license="Apache-2.0",
         source_url="https://huggingface.co/" + REPOSITORY,
         license_url="https://huggingface.co/" + REPOSITORY + "/blob/" + info.sha + "/LICENSE",
         ollama_model=args.model, status="downloading", download_started_at=stamp(), process_id=os.getpid(),
         error=None, error_type=None, failed_stage=None,
         files=[{"filename": name, "expected_bytes": size, "expected_sha256": sha} for name, (size, sha) in SHARDS.items()])
    print(f"Downloading official {REPOSITORY}@{info.sha}; total {sum(v[0] for v in SHARDS.values())} bytes", flush=True)

    def fetch(name):
        started = time.perf_counter()
        size, expected = SHARDS[name]
        target = DIRECTORY / name
        if target.is_file() and target.stat().st_size == size and digest(target) == expected:
            print(f"Verified cached shard: {name}", flush=True)
            return target
        print(f"Downloading shard: {name} ({size} bytes)", flush=True)
        path = Path(hf_hub_download(REPOSITORY, name, revision=info.sha, local_dir=DIRECTORY,
                                   cache_dir=DIRECTORY / ".hub-cache", token=False))
        if path.stat().st_size != size or digest(path) != expected:
            raise ValueError(f"{name}: 下载不完整或SHA256校验失败")
        with path.open("rb") as stream:
            if stream.read(4) != b"GGUF":
                raise ValueError(f"{name}: 不含GGUF文件头")
        print(f"Verified shard: {name}; seconds={time.perf_counter()-started:.1f}", flush=True)
        return path

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(fetch, SHARDS))
    for name in ("LICENSE", "README.md"):
        hf_hub_download(REPOSITORY, name, revision=info.sha, local_dir=DIRECTORY,
                        cache_dir=DIRECTORY / ".hub-cache", token=False)
    save(status="downloaded", download_completed_at=stamp(), process_id=None,
         files=[{"filename": name, "path": str((DIRECTORY / name).resolve()), "bytes": size, "sha256": sha,
                 "official_checksum_verified": True} for name, (size, sha) in SHARDS.items()])


def verify_shards():
    for name, (size, expected) in SHARDS.items():
        path = DIRECTORY / name
        if not path.is_file() or path.stat().st_size != size or digest(path) != expected:
            raise ValueError(f"{name}: 本地模型分片缺失或校验失败，请执行download")
    return state()


def get_template():
    path = DIRECTORY / "qwen2.5-tools.gotmpl"
    if path.exists() and digest(path) == TEMPLATE_SHA256:
        return path.read_text(encoding="utf-8")
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        response = client.get(TEMPLATE_URL)
        response.raise_for_status()
    # Official library renders one div per source line; retain exact newlines.
    page = response.text
    start = page.index("<div>{{- if .Messages }}")
    end = page.index("</div></div>", start) + len("</div>")
    template = html.unescape(page[start:end].replace("<div>", "").replace("</div>", ""))
    if hashlib.sha256(template.encode()).hexdigest() != TEMPLATE_SHA256:
        raise ValueError("官方工具模板HTML或内容发生变化，拒绝使用未核实模板")
    path.write_text(template, encoding="utf-8", newline="")
    return template


def stream_file(path):
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            yield data


def import_model(args):
    verify_shards()
    template = get_template()
    license_path = DIRECTORY / "LICENSE"
    if not license_path.is_file():
        raise ValueError("缺少官方LICENSE，请先完成download")
    license_text = license_path.read_text(encoding="utf-8")
    parameters = {"num_ctx": 4096, "temperature": 0.1, "top_p": 0.9, "top_k": 40,
                  "stop": ["<|im_start|>", "<|im_end|>"]}
    # This is a native split import, not concatenation of GGUF binary shards.
    modelfile = "FROM ./qwen2.5-7b-instruct-q4_k_m-*.gguf\n"
    modelfile += 'TEMPLATE """' + template + '"""\n'
    modelfile += 'LICENSE """' + license_text + '"""\n'
    modelfile += "PARAMETER num_ctx 4096\nPARAMETER temperature 0.1\nPARAMETER top_p 0.9\nPARAMETER top_k 40\n"
    modelfile += 'PARAMETER stop "<|im_start|>"\nPARAMETER stop "<|im_end|>"\n'
    (DIRECTORY / "Modelfile").write_text(modelfile, encoding="utf-8")
    base = args.ollama_url.rstrip("/")
    save(status="importing", ollama_model=args.model, ollama_url=base, import_started_at=stamp(),
         process_id=os.getpid(), template_url=TEMPLATE_URL, template_sha256=TEMPLATE_SHA256,
         modelfile=str((DIRECTORY / "Modelfile").resolve()), import_method="Ollama /api/blobs + /api/create files with original split names")
    with httpx.Client(base_url=base, timeout=httpx.Timeout(1200, connect=15)) as client:
        version = client.get("/api/version")
        version.raise_for_status()
        for name, (size, sha) in SHARDS.items():
            blob = "/api/blobs/sha256:" + sha
            exists = client.head(blob)
            if exists.status_code == 200:
                print(f"Cached Ollama blob: {name}", flush=True)
                continue
            if exists.status_code != 404:
                exists.raise_for_status()
            print(f"Importing Ollama blob: {name}", flush=True)
            response = client.post(blob, content=stream_file(DIRECTORY / name),
                                   headers={"Content-Length": str(size), "Content-Type": "application/octet-stream"})
            response.raise_for_status()
        request = {"model": args.model, "files": {name: "sha256:" + data[1] for name, data in SHARDS.items()},
                   "template": template, "license": license_text, "parameters": parameters, "stream": True}
        success, last = False, None
        with client.stream("POST", "/api/create", json=request) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("error"):
                    raise RuntimeError(record["error"])
                status = record.get("status", "")
                if status != last:
                    print("Ollama create:", status, flush=True)
                    last = status
                if status == "success":
                    success = True
        if not success:
            raise RuntimeError("Ollama create未返回完整成功状态")
        shown = client.post("/api/show", json={"model": args.model})
        shown.raise_for_status()
        details = shown.json()
        if "tools" not in details.get("capabilities", []):
            raise RuntimeError("导入后未识别工具能力，不能把模型标为Agent就绪")
    save(status="imported", import_completed_at=stamp(), process_id=None, error=None, error_type=None, failed_stage=None,
         ollama_version=version.json()["version"],
         capabilities=details.get("capabilities", []), model_details=details.get("details", {}),
         model_info=details.get("model_info", {}))


def metrics(response, elapsed):
    duration = response.get("eval_duration", 0)
    return {"elapsed_ms": round(elapsed * 1000, 2), "input_tokens": response.get("prompt_eval_count"),
            "output_tokens": response.get("eval_count"), "load_duration_ms": response.get("load_duration", 0) / 1e6,
            "tokens_per_second": response.get("eval_count", 0) / (duration / 1e9) if duration else None,
            "token_counts_from_provider": True, "done": response.get("done"), "done_reason": response.get("done_reason")}


def verify_model(args):
    base = args.ollama_url.rstrip("/")
    save(status="verifying", verification_started_at=stamp(), process_id=os.getpid())
    options = {"num_ctx": 4096, "num_predict": 96, "temperature": 0, "num_gpu": 99}
    with httpx.Client(base_url=base, timeout=httpx.Timeout(180, connect=15)) as client:
        shown = client.post("/api/show", json={"model": args.model})
        shown.raise_for_status()
        capabilities = shown.json().get("capabilities", [])
        if "completion" not in capabilities or "tools" not in capabilities:
            raise ValueError(f"模型能力不完整: {capabilities}")
        chat_messages = [{"role": "system", "content": "你是中文科研助理，请用一句话回答。"},
                         {"role": "user", "content": "为什么论文问答应附上可核查的原文来源？"}]
        started = time.perf_counter()
        response = client.post("/api/chat", json={"model": args.model, "messages": chat_messages, "stream": False,
                                                  "options": options, "keep_alive": "15m"})
        response.raise_for_status()
        chat = response.json()
        if chat.get("error") or not chat.get("done") or not chat.get("message", {}).get("content", "").strip():
            raise RuntimeError(f"真实聊天未成功完成: {chat}")
        chat_metrics = metrics(chat, time.perf_counter() - started)
        print("Chat verified:", chat["message"]["content"], json.dumps(chat_metrics, ensure_ascii=False), flush=True)
        tools = [{"type": "function", "function": {"name": "calculator",
                  "description": "计算安全算术表达式并返回数值。", "parameters": {"type": "object",
                  "properties": {"expression": {"type": "string", "description": "仅使用数字、括号和加减乘除"}},
                  "required": ["expression"], "additionalProperties": False}}}]
        messages = [{"role": "system", "content": "算术问题必须调用calculator工具；不要心算，不要直接回答。"},
                    {"role": "user", "content": "请调用calculator计算 (42+8)/5。"}]
        started = time.perf_counter()
        response = client.post("/api/chat", json={"model": args.model, "messages": messages, "tools": tools,
                                                  "stream": False, "options": options, "keep_alive": "15m"})
        response.raise_for_status()
        called = response.json()
        tool_metrics = metrics(called, time.perf_counter() - started)
        calls = called.get("message", {}).get("tool_calls", [])
        if not called.get("done") or not calls or calls[0].get("function", {}).get("name") != "calculator":
            save(status="verification_failed", chat_response=chat, native_tool_response=called,
                 verification_error="未收到实际native calculator tool_call")
            raise RuntimeError("未收到实际native calculator tool_call")
        function = calls[0]["function"]
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        expression = arguments.get("expression", "")
        import ast
        import operator
        operators = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
        def arithmetic(node):
            if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
                return node.value
            if isinstance(node, ast.BinOp) and type(node.op) in operators:
                return operators[type(node.op)](arithmetic(node.left), arithmetic(node.right))
            raise ValueError("模型给出不允许的表达式")
        value = arithmetic(ast.parse(expression, mode="eval").body)
        if value != 10:
            raise ValueError("实际calculator参数与测试算式不一致")
        print("Native tool verified:", json.dumps(function, ensure_ascii=False), "result", value, json.dumps(tool_metrics), flush=True)
        messages.extend([called["message"], {"role": "tool", "tool_name": "calculator", "content": json.dumps({"result": value})}])
        started = time.perf_counter()
        response = client.post("/api/chat", json={"model": args.model, "messages": messages, "tools": tools,
                                                  "stream": False, "options": options, "keep_alive": "15m"})
        response.raise_for_status()
        final = response.json()
        final_metrics = metrics(final, time.perf_counter() - started)
        if not final.get("done") or "10" not in final.get("message", {}).get("content", ""):
            raise ValueError("模型未依据实际工具结果返回正确答案")
        ps = client.get("/api/ps")
        ps.raise_for_status()
        running = next((item for item in ps.json().get("models", []) if item.get("name") == args.model), {})
    gpu = None
    try:
        process = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=True)
        gpu = process.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    save(status="ready", ollama_model=args.model, ollama_url=base, verified_at=stamp(), process_id=None,
         error=None, error_type=None, failed_stage=None,
         reproduction={"prepare": "python scripts/prepare_qwen_gguf.py all",
                       "offline_import": "python scripts/prepare_qwen_gguf.py import",
                       "smoke_check": "python scripts/prepare_qwen_gguf.py verify",
                       "application_environment": {"RAG_LLM_BACKEND": "ollama", "RAG_LLM_MODEL": args.model,
                                                   "RAG_OLLAMA_URL": base},
                       "ollama_serve_required": True, "permanent_path_modified": False,
                       "system_service_installed": False},
         verification={"capabilities": capabilities, "chat": {"response": chat, "metrics": chat_metrics},
                       "native_tool_call": {"response": called, "metrics": tool_metrics, "calculator_result": value},
                       "tool_roundtrip": {"response": final, "metrics": final_metrics},
                       "ollama_running_model": running, "nvidia_smi_gpu": gpu,
                       "gpu_loaded": bool(running.get("size_vram", 0)), "not_formal_quality_evaluation": True})
    print("Model ready:", args.model, "VRAM bytes", running.get("size_vram"), "GPU", gpu, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "import", "verify", "all"])
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--model", default="research-qwen:7b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    args = parser.parse_args()
    with operation_lock():
        try:
            if args.command in {"download", "all"}:
                download(args)
            if args.command in {"import", "all"}:
                import_model(args)
            if args.command in {"verify", "all"}:
                verify_model(args)
        except BaseException as exc:
            save(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "error", failed_stage=args.command,
                 error_type=type(exc).__name__, error=str(exc)[:2000], failed_at=stamp(), process_id=None)
            raise


if __name__ == "__main__":
    main()
