from __future__ import annotations

import difflib
import re
from collections.abc import Iterable
from statistics import mean
from typing import Any

from .dataset import RelevantSource


def _value(obj: Any, field: str, default: Any = None) -> Any:
    return obj.get(field, default) if isinstance(obj, dict) else getattr(obj, field, default)


def normalise_quote(text: str) -> str:
    return re.sub(r"\s+", "", text).casefold()


def evidence_matches_chunk(source: RelevantSource, chunk: Any) -> bool:
    """Match source locations plus exact text overlap, not document identity alone.

    Long quotes may straddle chunk boundaries. A contiguous overlap of at least
    40 normalised characters is sufficient; shorter quotes must be fully present.
    This deterministic rule is recorded in exported experiment metadata.
    """
    if source.document_id != _value(chunk, "document_id"):
        return False
    if source.page is not None:
        start, end = _value(chunk, "page_start"), _value(chunk, "page_end")
        pages = {_value(block, "page") for block in (_value(chunk, "sources", []) or [])}
        on_page = (start is not None and start <= source.page <= (end or start)) or source.page in pages
        if not on_page:
            return False
    if source.paragraph is not None:
        paragraphs = {_value(block, "paragraph") for block in (_value(chunk, "sources", []) or [])}
        if _value(chunk, "paragraph") != source.paragraph and source.paragraph not in paragraphs:
            return False
    quote, text = normalise_quote(source.quote), normalise_quote(_value(chunk, "text", ""))
    if not quote or not text:
        return False
    if quote in text:
        return True
    # Use the chunk's actual text: original SourceBlocks can contain much more
    # than this chunk, and using them would falsely label neighbouring chunks.
    minimum_overlap = min(40, len(quote))
    overlap = difflib.SequenceMatcher(None, quote, text, autojunk=False).find_longest_match().size
    return overlap >= minimum_overlap


def map_evidence_to_chunks(sources: list[RelevantSource], chunks: Iterable[Any]) -> list[set[str]]:
    pool = list(chunks)
    return [
        {_value(chunk, "chunk_id") for chunk in pool if evidence_matches_chunk(source, chunk)}
        for source in sources
    ]


def retrieval_metrics(retrieved_ids: list[str], evidence_chunk_ids: list[set[str]], k: int = 5) -> dict:
    """Recall denominator is annotated evidence count, never number of chunks.

    Two relevant fragments of the same evidence do not double-count recall.
    MRR uses the full supplied ranking; Hit and Recall use the first k results.
    """
    if not evidence_chunk_ids:
        raise ValueError("评测必须包含来源证据")
    relevant_ids = set().union(*evidence_chunk_ids)
    top_ids = set(retrieved_ids[:k])
    first_rank = next((i for i, cid in enumerate(retrieved_ids, 1) if cid in relevant_ids), None)
    recalled = sum(bool(ids & top_ids) for ids in evidence_chunk_ids)
    return {
        f"hit_at_{k}": float(bool(top_ids & relevant_ids)),
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        f"recall_at_{k}": recalled / len(evidence_chunk_ids),
        "evidence_count": len(evidence_chunk_ids),
        "recalled_evidence_count": recalled,
        "first_relevant_rank": first_rank,
    }


def summarise_records(records: list[dict]) -> dict:
    successful = [row for row in records if not row.get("error")]
    keys = ("hit_at_5", "mrr", "recall_at_5", "elapsed_ms")
    metrics: dict[str, Any] = {
        "case_count": len(records), "successful_count": len(successful),
        "error_count": len(records) - len(successful),
        # Failures count as zero quality, so failure-prone systems cannot obtain
        # an inflated quality score by excluding failed questions.
        **{key: mean(row.get(key, 0.0) for row in records) if records else 0.0 for key in keys[:3]},
    }
    latencies = sorted(row["elapsed_ms"] for row in records if "elapsed_ms" in row)
    if latencies:
        import numpy as np
        metrics.update(latency_mean_ms=mean(latencies), latency_p50_ms=float(np.percentile(latencies, 50)),
                       latency_p95_ms=float(np.percentile(latencies, 95)))
    metrics["by_category"] = {
        category: {key: mean(row.get(key, 0.0) for row in records if row["category"] == category)
                   for key in keys[:3]}
        for category in sorted({row["category"] for row in records})
    }
    return metrics
