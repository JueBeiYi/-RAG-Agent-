from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict

import numpy as np

from src.core.config import Settings
from src.core.schemas import Chunk, RetrievalHit
from .embeddings import Embedder
from .reranking import Reranker
from .rrf import reciprocal_rank_fusion
from .tokenization import lexical_coverage, tokenize
from .vectorstores import ChromaVectors, FaissVectors, SQLiteVectors, text_hash


logger = logging.getLogger(__name__)


class Retriever:
    """Scoped hybrid search over persistent vectors and live document records."""

    MODES = {"vector", "hybrid", "hybrid_rerank", "bm25"}

    def __init__(self, settings: Settings, store):
        self.settings = settings
        self.store = store
        self.embedder = Embedder(settings)
        self.reranker = Reranker(settings)
        if settings.vector_backend not in {"sqlite", "chroma", "faiss"}:
            raise ValueError("vector_backend 可选 sqlite / chroma / faiss")
        cls = {"sqlite": SQLiteVectors, "chroma": ChromaVectors, "faiss": FaissVectors}[settings.vector_backend]
        self.vectors = cls(settings.path / "indices", self.embedder.fingerprint)
        self.lock = threading.RLock()
        self.bm25_cache = OrderedDict()
        self._context = threading.local()
        self.last_search = {}
        self.last_warning = ""

    @property
    def last_search(self) -> dict:
        return getattr(self._context, "last_search", {})

    @last_search.setter
    def last_search(self, value: dict):
        self._context.last_search = value

    @property
    def last_warning(self) -> str:
        return getattr(self._context, "last_warning", "")

    @last_warning.setter
    def last_warning(self, value: str):
        self._context.last_warning = value

    def index(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        identifiers = [(chunk.scope, chunk.chunk_id) for chunk in chunks]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("一次索引不能包含重复 chunk_id")
        if any(not c.text.strip() for c in chunks):
            raise ValueError("不能索引空文本")
        with self.lock:
            existing = self.vectors.existing(chunks)
            pending = [c for c in chunks if existing.get((c.scope, c.chunk_id)) != text_hash(c.text)]
            # One atomic SQLite upsert / bounded Chroma batches. A failed model
            # call leaves the previous index intact and the worker marks failure.
            if pending:
                encoded = self.embedder.encode([c.text for c in pending])
                if encoded.ndim != 2 or len(encoded) != len(pending) or not np.isfinite(encoded).all():
                    raise RuntimeError("Embedding 返回了无效向量")
                self.vectors.upsert(pending, encoded)
            for scope, doc_id in {(c.scope, c.document_id) for c in chunks}:
                self.vectors.retain(scope, doc_id, [c.chunk_id for c in chunks if c.scope == scope and c.document_id == doc_id])
            self.bm25_cache.clear()

    def delete(self, document_id: str, scope: str) -> None:
        with self.lock:
            self.vectors.delete(document_id, scope)
            self.bm25_cache.clear()

    def _chunks(self, scope: str, document_ids: list[str] | None) -> list[Chunk]:
        chunks = self.store.list_chunks(scope, document_ids=document_ids)
        chunks = [c if isinstance(c, Chunk) else Chunk.model_validate(c) for c in chunks]
        selected = set(document_ids or [])
        chunks = [c for c in chunks if c.scope == scope and (not selected or c.document_id in selected)]
        if hasattr(self.store, "get_document"):
            ready_ids = set()
            for identifier in {c.document_id for c in chunks}:
                document = self.store.get_document(identifier, scope)
                if document is not None:
                    status = document.status if hasattr(document, "status") else document.get("status")
                    if status == "ready":
                        ready_ids.add(identifier)
            chunks = [c for c in chunks if c.document_id in ready_ids]
        return chunks

    def _bm25(self, query: str, chunks: list[Chunk], k: int) -> tuple[list[tuple[str, float]], dict[str, float]]:
        from rank_bm25 import BM25Okapi

        query_terms = tokenize(query)
        if not query_terms:
            return [], {}
        identity = hashlib.sha256("\0".join(c.chunk_id + text_hash(c.text) for c in chunks).encode()).hexdigest()
        key = (chunks[0].scope, identity)
        if key not in self.bm25_cache:
            documents = [tokenize(chunk.text) or ["__empty__"] for chunk in chunks]
            self.bm25_cache[key] = (BM25Okapi(documents), documents)
            if len(self.bm25_cache) > 16:
                self.bm25_cache.popitem(last=False)
        model, terms = self.bm25_cache[key]
        scores = model.get_scores(query_terms)
        coverage = {c.chunk_id: len(set(query_terms) & set(terms[i])) / len(set(query_terms))
                    for i, c in enumerate(chunks)}
        # BM25 can be negative when a term occurs in most documents. Relevance
        # eligibility uses actual term overlap, preserving raw BM25 for experiments.
        ranks = sorted((i for i, c in enumerate(chunks) if coverage[c.chunk_id] > 0),
                       key=lambda i: (-float(scores[i]), -coverage[chunks[i].chunk_id], chunks[i].chunk_id))[:k]
        return [(chunks[i].chunk_id, float(scores[i])) for i in ranks], coverage

    def search(self, query: str, scope: str, document_ids: list[str] | None = None,
               mode: str | None = None, top_k: int | None = None) -> list[RetrievalHit]:
        mode = mode or self.settings.retrieval_mode
        if mode not in self.MODES:
            raise ValueError(f"未知检索模式 {mode}；可选 {sorted(self.MODES)}")
        limit = self.settings.result_k if top_k is None else top_k
        if limit < 1 or limit > 1000:
            raise ValueError("top_k 必须在 1..1000")
        if not query.strip():
            return []
        started = time.perf_counter()
        warning = ""
        with self.lock:
            chunks = self._chunks(scope, document_ids)
            if not chunks:
                self.last_search = {"mode": mode, "scope": scope, "candidates": 0, "results": 0,
                                    "elapsed_ms": (time.perf_counter() - started) * 1000}
                return []
            by_id = {c.chunk_id: c for c in chunks}
            candidate_k = min(max(self.settings.candidate_k, limit), len(chunks))
            vector, keyword, coverage = [], [], {}
            vector_ms, keyword_ms, rerank_ms = 0.0, 0.0, 0.0
            if mode != "bm25":
                timer = time.perf_counter()
                encoded = self.embedder.encode_query(query)
                vector = self.vectors.query(encoded, scope, set(by_id), candidate_k)
                vector_ms = (time.perf_counter() - timer) * 1000
            if mode != "vector":
                timer = time.perf_counter()
                keyword, coverage = self._bm25(query, chunks, candidate_k)
                keyword_ms = (time.perf_counter() - timer) * 1000
            vector_scores, keyword_scores = dict(vector), dict(keyword)
            vector_ranks = {identifier: rank for rank, (identifier, _) in enumerate(vector, 1)}
            keyword_ranks = {identifier: rank for rank, (identifier, _) in enumerate(keyword, 1)}
            fused = reciprocal_rank_fusion({"vector": list(vector_scores), "bm25": list(keyword_scores)}, self.settings.rrf_k)
            rrf_scores = dict(fused)
            order = list(vector_scores) if mode == "vector" else (list(keyword_scores) if mode == "bm25" else [item[0] for item in fused])
            hits = []
            for identifier in order[:candidate_k]:
                if identifier not in by_id:
                    continue
                vector_score, lexical = vector_scores.get(identifier), coverage.get(identifier, 0.0)
                relevance = max(0.0, min(1.0, vector_score or 0.0)) if mode == "vector" else max(lexical, max(0.0, min(1.0, vector_score or 0.0)))
                ranks = {}
                if identifier in vector_ranks:
                    ranks["vector"] = vector_ranks[identifier]
                if identifier in keyword_ranks:
                    ranks["bm25"] = keyword_ranks[identifier]
                hits.append(RetrievalHit(chunk=by_id[identifier], score=relevance, vector_score=vector_score,
                                         bm25_score=keyword_scores.get(identifier),
                                         rrf_score=rrf_scores.get(identifier) if mode.startswith("hybrid") else None,
                                         ranks=ranks))
            if mode == "hybrid_rerank":
                timer = time.perf_counter()
                try:
                    hits = self.reranker.rerank(query, hits)
                except Exception as exc:
                    warning = f"重排失败，已降级为 RRF 混合检索: {exc}"
                    logger.warning(warning)
                rerank_ms = (time.perf_counter() - timer) * 1000
            self.last_warning = warning
            self.last_search = {"mode": mode, "scope": scope, "candidate_k": candidate_k,
                                "candidates": len(hits), "results": min(limit, len(hits)),
                                "vector_ms": vector_ms, "bm25_ms": keyword_ms, "rerank_ms": rerank_ms,
                                "elapsed_ms": (time.perf_counter() - started) * 1000,
                                "warning": warning, "index_version": self.store.index_version(scope),
                                "embedding_fingerprint": self.embedder.fingerprint,
                                "score_is_probability": False}
            return hits[:limit]

    def health(self) -> dict:
        embedding, vectors, reranker = self.embedder.health(), self.vectors.health(), self.reranker.health()
        return {"ready": embedding["ready"] and vectors["ready"] and reranker["ready"],
                "demo": any(info.get("demo") for info in (embedding, vectors, reranker)),
                "embedding": embedding, "vectors": vectors, "reranker": reranker,
                # Health is public dependency state. Request scopes and
                # request-specific queries/results stay in protected run logs.
                "warning": self.last_warning,
                "score_is_probability": False}
