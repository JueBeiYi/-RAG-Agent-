from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(rankings: dict[str, list[str]] | list[list[str]],
                           k: int = 60) -> list[tuple[str, float]]:
    """Handwritten RRF: sum 1/(k+one-based-rank), with per-list deduplication.

    Input scores are intentionally ignored so heterogeneous vector/BM25 score
    scales cannot accidentally dominate the fusion. Ties use a stable ID order.
    """
    if k <= 0:
        raise ValueError("RRF k 必须大于 0")
    scores: dict[str, float] = defaultdict(float)
    lists = rankings.values() if isinstance(rankings, dict) else rankings
    for ranking in lists:
        seen = set()
        rank = 0
        for identifier in ranking:
            if identifier in seen:
                continue
            seen.add(identifier)
            rank += 1
            scores[identifier] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
