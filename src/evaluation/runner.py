from __future__ import annotations

import csv
import hashlib
import io
import json
import platform
import time
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .dataset import EvaluationCase
from .metrics import map_evidence_to_chunks, retrieval_metrics, summarise_records

MATCHING_RULE = "document+physical-page/paragraph+normalised-contiguous-quote-overlap>=min(40,quote_length); evidence-level recall"


def _dump(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {key: _dump(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_dump(value) for value in obj]
    return obj


def _fingerprint(value: Any) -> str:
    raw = json.dumps(_dump(value), sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def package_versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for name in ("numpy", "chromadb", "sentence-transformers", "rank-bm25", "PyMuPDF", "transformers"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not-installed"
    return result


def _model_identity(retriever: Any, settings: Any) -> dict:
    """Capture configured identifiers and local snapshot commit when available."""
    result = {key: getattr(settings, key, "") for key in
              ("embedding_backend", "embedding_model", "reranker_backend", "reranker_model", "llm_backend", "llm_model")}
    for key in ("embedding_model", "reranker_model"):
        candidate = Path(str(result[key]))
        if candidate.is_dir():
            result[key + "_local_path"] = str(candidate.resolve())
            # Hugging Face snapshot directory name is the immutable revision.
            result[key + "_revision"] = candidate.name if candidate.parent.name == "snapshots" else "local-unspecified"
            tracked = {}
            for name in ("config.json", "modules.json", "tokenizer_config.json"):
                artifact = candidate / name
                if artifact.is_file():
                    tracked[name] = hashlib.sha256(artifact.read_bytes()).hexdigest()
            result[key + "_config_hashes"] = tracked
    backend = getattr(retriever, "embedder", getattr(retriever, "embedding", None))
    if backend is not None and hasattr(backend, "identity"):
        identity = backend.identity
        result["runtime_embedding_identity"] = identity() if callable(identity) else identity
    if backend is not None and hasattr(backend, "fingerprint"):
        result["embedding_fingerprint"] = backend.fingerprint
    for name, adapter in (("embedding", backend), ("reranker", getattr(retriever, "reranker", None))):
        model = getattr(adapter, "model", None)
        if model is None:
            continue
        try:
            transformer = model[0].auto_model if name == "embedding" else model.model
            commit = getattr(transformer.config, "_commit_hash", None)
            if commit:
                result[name + "_loaded_revision"] = commit
        except (AttributeError, TypeError, IndexError, KeyError):
            pass
    return result


def run_retrieval(retriever: Any, store: Any, cases: list[EvaluationCase], scope: str,
                  mode: str = "hybrid_rerank", output_dir: str | Path | None = None,
                  settings: Any = None) -> dict:
    if mode not in {"vector", "hybrid", "hybrid_rerank"}:
        raise ValueError(f"未知检索模式：{mode}")
    if not cases:
        raise ValueError("评测集为空")
    chunks = store.list_chunks(scope)
    evidence_maps = {}
    for case in cases:
        if not case.reviewed or not case.relevant_sources or not case.reference_answer.strip():
            raise ValueError(f"{case.case_id} 未通过人工核对或缺少答案/证据")
        evidence = map_evidence_to_chunks(case.relevant_sources, chunks)
        missing = [i + 1 for i, ids in enumerate(evidence) if not ids]
        if missing:
            raise ValueError(f"{case.case_id} 的证据 {missing} 未映射到当前知识库。请核对文档 ID、物理页码、摘录与解析结果")
        evidence_maps[case.case_id] = evidence
    config = settings.public_dict() if settings and hasattr(settings, "public_dict") else {}
    # Evaluate >= 5 returned results even when the application uses a smaller k.
    ranking_k = max(5, int(config.get("candidate_k", 20)))
    documents = store.list_documents(scope)
    corpus = [{"document_id": d.document_id, "content_hash": d.content_hash, "name": d.name} for d in documents]
    index_before = store.index_version(scope)
    started = time.perf_counter()
    records = []
    for case in cases:
        row = {"case_id": case.case_id, "question": case.question, "category": case.category,
               "split": case.split, "reference_answer": case.reference_answer,
               "relevant_sources": [_dump(source) for source in case.relevant_sources],
               "expected_tools": case.expected_tools, "error": ""}
        case_start = time.perf_counter()
        try:
            hits = retriever.search(case.question, scope, mode=mode, top_k=ranking_k)
            # Tolerate a (hits, metadata) return without losing the raw ranking.
            if isinstance(hits, tuple):
                hits = hits[0]
            raw = [_dump(hit) for hit in hits]
            ids = [hit["chunk"]["chunk_id"] for hit in raw]
            row.update(retrieval_metrics(ids, evidence_maps[case.case_id], k=5))
            row.update(retrieved=raw, retrieved_chunk_ids=ids,
                       top1_score=raw[0].get("score") if raw else None,
                       matching_chunk_ids=[sorted(ids) for ids in evidence_maps[case.case_id]])
        except Exception as exc:
            row.update(error=f"{type(exc).__name__}: {exc}", hit_at_5=0.0, mrr=0.0, recall_at_5=0.0,
                       retrieved=[], retrieved_chunk_ids=[])
        row["elapsed_ms"] = round((time.perf_counter() - case_start) * 1000, 3)
        records.append(row)
    index_after = store.index_version(scope)
    if index_before != index_after:
        raise ValueError("评测期间知识库发生变更，此次结果不能作正式对比；请完成导入后重跑")
    demo = config.get("embedding_backend") == "hash" or config.get("llm_backend") == "extractive"
    result = {
        "experiment_id": str(uuid.uuid4()), "kind": "retrieval", "mode": mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": sorted({case.split for case in cases}), "status": "completed",
        "demo": demo, "formal_models": not demo,
        "dataset_requirement_checks": {
            "at_least_50_reviewed_cases": len(cases) >= 50,
            "all_four_categories": {case.category for case in cases} == {"factual", "comparison", "synthesis", "reasoning"},
            "document_count_10_to_20": 10 <= len(documents) <= 20,
            "document_count": len(documents),
            "human_corpus_confirmation": "not_recorded",
            "note": "文档数量不能证明都是论文、同一研究方向或非演示语料，须人工确认；此对象不声明课程验收已完成。",
        },
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "config": config, "config_hash": _fingerprint(config),
        "dataset_hash": _fingerprint([case.model_dump() for case in cases]),
        "corpus_hash": _fingerprint(sorted(corpus, key=lambda d: d["document_id"])),
        "corpus": corpus, "index_version": index_before,
        "models": _model_identity(retriever, settings), "package_versions": package_versions(),
        "matching_rule": MATCHING_RULE,
        "metrics": summarise_records(records), "records": records,
        "notes": ["只测检索，不代表生成质量；人工答案评分须另行完成。",
                  "MRR 使用完整返回排序；Hit@5 与 Recall@5 使用前五条。失败题计零分。",
                  "首次检索可能包含模型初始化成本，比较时应采用相同冷/热启动条件。"],
    }
    if output_dir:
        result["artifacts"] = export_result(result, output_dir)
    return result


def flat_records(result: dict) -> list[dict]:
    return [{key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
             for key, value in row.items() if key != "retrieved"} for row in result.get("records", [])]


def serialize_export(result: dict, format: str = "json") -> bytes:
    if format == "json":
        return json.dumps(result, ensure_ascii=False, indent=2, default=str).encode("utf-8")
    if format == "csv":
        rows = flat_records(result)
        out = io.StringIO(newline="")
        fields = sorted({key for row in rows for key in row})
        if fields:
            writer = csv.DictWriter(out, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return b"\xef\xbb\xbf" + out.getvalue().encode("utf-8")
    if format == "xlsx":
        import pandas as pd
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as workbook:
            pd.DataFrame(flat_records(result)).to_excel(workbook, sheet_name="cases", index=False)
            metric_rows = [{"metric": key, "value": value} for key, value in result.get("metrics", {}).items()
                           if not isinstance(value, (dict, list))]
            pd.DataFrame(metric_rows).to_excel(workbook, sheet_name="metrics", index=False)
            metadata = [{"field": key, "value": json.dumps(value, ensure_ascii=False, default=str)}
                        for key, value in result.items() if key not in {"records", "metrics"}]
            pd.DataFrame(metadata).to_excel(workbook, sheet_name="metadata", index=False)
        return out.getvalue()
    raise ValueError("导出格式必须为 json/csv/xlsx")


def export_result(result: dict, output_dir: str | Path) -> dict[str, str]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for format in ("json", "csv", "xlsx"):
        path = directory / f"results.{format}"
        path.write_bytes(serialize_export(result, format))
        artifacts[format] = str(path.resolve())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    values = [result["metrics"][key] for key in ("hit_at_5", "mrr", "recall_at_5")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].bar(["Hit@5", "MRR", "Recall@5"], values, color=["#287c85", "#4e71a8", "#6f9670"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title(f"{result['mode']} | n={len(result['records'])}")
    for i, value in enumerate(values):
        axes[0].text(i, value + 0.015, f"{value:.3f}", ha="center")
    latencies = [row["elapsed_ms"] for row in result["records"]]
    axes[1].hist(latencies, bins=min(15, max(1, len(latencies))), color="#4e71a8", edgecolor="white")
    axes[1].set_xlabel("Retrieval latency (ms)")
    axes[1].set_ylabel("Cases")
    fig.tight_layout()
    chart = directory / "metrics.png"
    fig.savefig(chart, dpi=160)
    plt.close(fig)
    artifacts["chart"] = str(chart.resolve())
    scores = [row.get("top1_score") for row in result["records"] if row.get("top1_score") is not None]
    if scores:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.hist(scores, bins=min(15, max(1, len(scores))), color="#287c85", edgecolor="white")
        ax.set_xlabel("Top-1 score (backend-specific, not a probability)")
        ax.set_ylabel("Cases")
        fig.tight_layout()
        score_chart = directory / "top1_scores.png"
        fig.savefig(score_chart, dpi=160)
        plt.close(fig)
        artifacts["top1_scores"] = str(score_chart.resolve())
    return artifacts
