"""Explicit online preparation, never run automatically by the application.

The actual downloaded Hugging Face snapshot revisions are saved in a manifest.
Use the printed snapshot paths in config.yaml for reproducible offline loading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="下载并缓存本地 Embedding/重排模型，按需拉取 Ollama")
    parser.add_argument("--models", nargs="+", default=["BAAI/bge-large-zh-v1.5", "moka-ai/m3e-base", "BAAI/bge-reranker-base"])
    parser.add_argument("--revision", default="main", help="可传固定 commit/tag；实际snapshot commit仍会记录")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--manifest", default="data/models/manifest.json")
    parser.add_argument("--ollama", default=None, help="例如 qwen2.5:7b-instruct-q4_K_M；需已安装 Ollama")
    args = parser.parse_args()
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise SystemExit("请先执行 .venv\\Scripts\\python.exe -m pip install -r requirements-models.txt") from exc
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "models": []}
    for model in args.models:
        print(f"准备模型：{model}（首次运行会联网下载较大文件）", flush=True)
        info = HfApi().model_info(model, revision=args.revision)
        names = [file.rfilename for file in info.siblings]
        # Download one weight format, not both .bin and .safetensors copies.
        excluded = ["*.onnx", "onnx/*", "openvino/*", "*.h5", "*.msgpack", "*.ot"]
        if any(name.endswith(".safetensors") and "/" not in name for name in names):
            excluded.append("*.bin")
        path = Path(snapshot_download(repo_id=model, revision=args.revision, cache_dir=args.cache_dir,
                                      ignore_patterns=excluded))
        checksums = {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                     for name in ("config.json", "modules.json", "tokenizer_config.json") if (path / name).is_file()}
        manifest["models"].append({"model_id": model, "requested_revision": args.revision,
                                   "snapshot_revision": path.name, "local_path": str(path.resolve()),
                                   "config_sha256": checksums, "license": (info.card_data or {}).get("license"),
                                   "source": "https://huggingface.co/" + model})
        # Persist completed downloads even if a later model/network operation fails.
        destination = Path(args.manifest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"可在配置中填写模型路径：{path.resolve()}", flush=True)
    if args.ollama:
        executable = shutil.which("ollama")
        if not executable:
            raise SystemExit("Ollama 未安装或不在 PATH；Embedding下载已完成，LLM请准备后重试。")
        subprocess.run([executable, "pull", args.ollama], check=True)
        # `ollama show` is metadata only; capture JSON-independent digest output.
        shown = subprocess.run([executable, "show", args.ollama], check=True, capture_output=True, text=True)
        manifest["ollama"] = {"model": args.ollama, "show": shown.stdout}
    destination = Path(args.manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"下载清单：{destination.resolve()}")


if __name__ == "__main__":
    main()
