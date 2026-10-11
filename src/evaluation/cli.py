from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import time
from dataclasses import replace
from pathlib import Path

from src.core.config import Settings, load_settings
from src.core.storage import Storage
from .dataset import load_cases
from .runner import export_result, run_retrieval, serialize_export

PRESETS = {
    "pdf": {"pdf_parser": ["pymupdf", "pdfplumber", "pypdf2"]},
    "chunking": {"chunk_strategy": ["fixed", "recursive", "semantic"], "chunk_size": [256, 512, 1024]},
    "embedding": {"embedding_model": ["BAAI/bge-large-zh-v1.5", "moka-ai/m3e-base"]},
    "retrieval": {"retrieval_mode": ["vector", "hybrid", "hybrid_rerank"]},
    "generation": {"temperature": [0.0, 0.1, 0.5], "top_p": [0.7, 0.9], "top_k_sampling": [20, 40]},
    "parallel": {"max_tool_workers": [1, 2]},
    "routing": {"agent_routing_enabled": [False, True]},
    "cache": {"cache_enabled": [False, True]},
    "context": {"context_tokens": [2048, 4096]},
}


def _build_variant(settings: Settings, source: Storage, scope: str):
    from src.data_loader.loaders import load_blocks
    from src.chunking.strategies import split_blocks
    from src.retrieval.engine import Retriever
    store = Storage(settings.path)
    retriever = Retriever(settings, store)
    stats = []
    for original in source.list_documents(scope):
        if original.status != "ready":
            continue
        document = original.model_copy(deep=True)
        start = time.perf_counter()
        blocks = load_blocks(document.path, settings.pdf_parser)
        parse_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        chunks = split_blocks(document, blocks, settings.chunk_strategy, settings.chunk_size, settings.chunk_overlap)
        chunk_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        retriever.index(chunks)
        embed_ms = (time.perf_counter() - start) * 1000
        document.chunks = len(chunks)
        store.put_document(document)
        store.replace_chunks(document.document_id, scope, chunks)
        stats.append({"document_id": document.document_id, "name": document.name,
                      "parser": settings.pdf_parser, "block_count": len(blocks),
                      "text_characters": sum(len(block.text) for block in blocks),
                      "chunk_count": len(chunks), "parse_ms": parse_ms,
                      "chunk_ms": chunk_ms, "embedding_index_ms": embed_ms})
    store.bump_version(scope)
    return store, retriever, stats


def main(argv=None):
    parser = argparse.ArgumentParser(description="科研助理真实评测与参数实验")
    parser.add_argument("--config", default=None, help="可选 YAML 配置文件")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="检查评测 JSONL，拒绝未人工核对的题目")
    validate.add_argument("cases")
    validate.add_argument("--allow-unreviewed", action="store_true", help="只检查模板结构，不代表可用于评测")
    validate.add_argument("--split", choices=["dev", "test"])
    for name in ("retrieval", "generation", "matrix"):
        command = sub.add_parser(name)
        command.add_argument("--cases", required=True)
        command.add_argument("--scope", required=True, help="与前端工作区标识一致")
        command.add_argument("--split", choices=["dev", "test"], default="dev")
        command.add_argument("--output", required=True)
        if name != "matrix":
            command.add_argument("--mode", choices=["vector", "hybrid", "hybrid_rerank"], default="hybrid_rerank")
        if name == "generation":
            command.add_argument("--rag-only", action="store_true")
            command.add_argument("--repeats", type=int, default=1)
        if name == "matrix":
            command.add_argument("--preset", choices=list(PRESETS), default="retrieval")
            command.add_argument("--grid", help='JSON 文件，例如 {"chunk_size":[256,512,1024]}，覆盖preset')
            command.add_argument("--task", choices=["retrieval", "generation"], default="retrieval")
            command.add_argument("--repeats", type=int, default=1)
    compare = sub.add_parser("compare", help="汇总已完成结果，保证语料与数据集相同")
    compare.add_argument("results", nargs="+")
    compare.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "validate":
        cases = load_cases(args.cases, split=args.split, require_reviewed=not args.allow_unreviewed)
        print(f"结构检查通过：{len(cases)} 题；dev={sum(c.split == 'dev' for c in cases)}；test={sum(c.split == 'test' for c in cases)}")
        if args.allow_unreviewed:
            print("模板检查模式，未核对题目不能用于正式评测。")
        return
    if args.command == "compare":
        results = [json.loads(Path(path).read_text(encoding="utf-8-sig")) for path in args.results]
        identity = {(result["dataset_hash"], result["corpus_hash"], str(result["split"])) for result in results}
        if len(identity) != 1:
            raise ValueError("语料、评测集或 split 不一致，不可直接比较")
        rows = [{"experiment_id": result["experiment_id"], "mode": result.get("mode"), "kind": result.get("kind"),
                 "demo": result.get("demo"), "config_hash": result.get("config_hash"),
                 **{key: value for key, value in result["metrics"].items() if not isinstance(value, dict)}} for result in results]
        _write_summary(rows, Path(args.output))
        return
    from src.retrieval.engine import Retriever
    settings = load_settings(args.config)
    source = Storage(settings.path)
    cases = load_cases(args.cases, split=args.split)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.command == "retrieval":
        result = run_retrieval(Retriever(settings, source), source, cases, args.scope, args.mode, output, settings)
        print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    elif args.command == "generation":
        from src.services.application import Application
        from .generation import run_generation
        application = Application(settings, recover_jobs=False)
        try:
            result = run_generation(application, cases, args.scope, agent=not args.rag_only,
                                    mode=args.mode, repeats=args.repeats)
            for fmt in ("json", "csv", "xlsx"):
                (output / f"results.{fmt}").write_bytes(serialize_export(result, fmt))
            print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
        finally:
            application.close()
    else:
        if args.split != "dev":
            raise ValueError("参数扫描只允许 dev 集，避免在 test 集上挑选最佳参数")
        grid = json.loads(Path(args.grid).read_text(encoding="utf-8-sig")) if args.grid else PRESETS[args.preset]
        if not grid or any(not isinstance(values, list) or not values for values in grid.values()):
            raise ValueError("grid 必须为非空列表组成的 JSON 对象")
        unknown = set(grid) - set(settings.public_dict())
        if unknown:
            raise ValueError(f"未知设置：{sorted(unknown)}")
        if "data_dir" in grid:
            raise ValueError("matrix 的 data_dir 由隔离实验目录管理，不能作为扫描参数")
        rows = []
        for number, values in enumerate(itertools.product(*grid.values()), 1):
            overrides = dict(zip(grid, values))
            variant_dir = output / f"variant-{number:03d}"
            variant = replace(settings, **overrides, data_dir=str(variant_dir / "runtime"))
            print(f"[{number}] {json.dumps(overrides, ensure_ascii=False)}", flush=True)
            try:
                if variant.chunk_overlap >= variant.chunk_size:
                    raise ValueError("重叠不可大于或等于分块大小")
                store, retriever, parsing = _build_variant(variant, source, args.scope)
                if args.task == "generation":
                    from src.services.application import Application
                    from .generation import run_generation
                    application = Application(variant, recover_jobs=False)
                    try:
                        result = run_generation(application, cases, args.scope, repeats=args.repeats,
                                                mode=variant.retrieval_mode)
                    finally:
                        application.close()
                        del application
                    for fmt in ("json", "csv", "xlsx"):
                        (variant_dir / f"results.{fmt}").write_bytes(serialize_export(result, fmt))
                else:
                    result = run_retrieval(retriever, store, cases, args.scope, variant.retrieval_mode,
                                           settings=variant)
                    result["ingestion_measurements"] = parsing
                    export_result(result, variant_dir)
                rows.append({"variant": number, **overrides, "error": "", "demo": result.get("demo"),
                             **{key: value for key, value in result["metrics"].items() if not isinstance(value, dict)}})
            except Exception as exc:
                rows.append({"variant": number, **overrides, "error": str(exc)})
                print(f"变体失败，保留错误记录：{exc}", flush=True)
            _write_summary(rows, output)
            # Release CPU model objects before loading a large next variant.
            if "retriever" in locals():
                del retriever
            gc.collect()


def _write_summary(rows: list[dict], output: Path):
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    print(f"汇总保存至 {output.resolve()}")


if __name__ == "__main__":
    main()
