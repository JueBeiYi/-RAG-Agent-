"""Run the minimal real course retrieval matrix in bounded serial stages."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Set these before any lazy Hugging Face constants/model imports.
os.environ["HF_HUB_CACHE"] = str(ROOT / "data" / "models" / "hub")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "chunks", "m3-queries", "m3e", "bge-zh", "stores", "rank", "rerank", "summary", "all"])
    parser.add_argument("--config", default=str(ROOT / "config.standard.yaml"))
    parser.add_argument("--data-dir", default=str(ROOT / "data/runtime/standard"))
    parser.add_argument("--workspace-file", default=str(ROOT / "data/runtime/workspace-standard.json"))
    parser.add_argument("--cases", default=str(ROOT / "reports/datasets/standard/vision_transformer_80_mapped.jsonl"))
    parser.add_argument("--cache", default=str(ROOT / "data/runtime/benchmarks/retrieval-cache"))
    parser.add_argument("--output", default=str(ROOT / "reports/experiments/retrieval"))
    parser.add_argument("--rerank-batch", type=int, choices=[1, 2, 4], default=2)
    args = parser.parse_args()
    if args.stage == "all":
        # Process exit releases each model before the next model can load.
        for stage in ("prepare", "chunks", "m3e", "bge-zh", "m3-queries", "stores", "rank", "rerank", "summary"):
            command = [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), stage]
            for key in ("config", "data_dir", "workspace_file", "cases", "cache", "output", "rerank_batch"):
                command.extend(["--" + key.replace("_", "-"), str(getattr(args, key))])
            subprocess.run(command, check=True)
        return
    from src.core.config import load_settings
    from src.evaluation import retrieval as study
    settings = replace(load_settings(args.config), data_dir=args.data_dir, embedding_model="BAAI/bge-m3",
                       embedding_backend="sentence_transformers", embedding_device="cpu", embedding_batch_size=1,
                       embedding_max_tokens=512, cpu_threads=8, reranker_backend="cross_encoder", offline=True)
    if args.stage == "prepare":
        workspace = json.loads(Path(args.workspace_file).read_text(encoding="utf-8"))
        scope = workspace["workspace_token"]
        study.snapshot_corpus(settings, scope, Path(args.cases), args.cache)
        study.parser_comparison(args.cache, args.output)
    elif args.stage == "chunks":
        study.chunk_variants(args.cache)
    elif args.stage == "m3-queries":
        study.encode_queries(settings, args.cache)
    elif args.stage in {"m3e", "bge-zh"}:
        model = "moka-ai/m3e-base" if args.stage == "m3e" else "BAAI/bge-large-zh-v1.5"
        study.embedding_study(replace(settings, embedding_model=model), args.stage, args.cache, args.output)
    elif args.stage == "stores":
        study.vectorstore_study(settings, args.cache, args.output)
    elif args.stage == "rank":
        study.rank_study(settings, args.cache, args.output)
    elif args.stage == "rerank":
        study.rerank_study(replace(settings, embedding_batch_size=args.rerank_batch), args.cache, args.output)
    elif args.stage == "summary":
        study.summarise_suite(args.cache, args.output)


if __name__ == "__main__":
    main()
