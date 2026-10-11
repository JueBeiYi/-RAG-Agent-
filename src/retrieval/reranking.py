from __future__ import annotations

import importlib.util
import inspect
import math
import threading

from src.core.config import Settings
from src.core.schemas import RetrievalHit
from .embeddings import model_cached
from .tokenization import lexical_coverage
from .token_windows import WINDOW_POLICY, pair_windows, classifier_scores


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-min(value, 700)))
    exp = math.exp(max(value, -700))
    return exp / (1 + exp)


class Reranker:
    def __init__(self, settings: Settings):
        if settings.reranker_backend not in {"cross_encoder", "lexical", "none"}:
            raise ValueError("reranker_backend 可选 cross_encoder / lexical / none")
        self.settings = settings
        self.model = None
        self.lock = threading.RLock()

    def _load(self):
        if self.model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError("真实重排模型需要 requirements-models.txt；demo 使用明确标注的词覆盖重排") from exc
            try:
                kwargs = {"device": self.settings.embedding_device, "trust_remote_code": False}
                parameters = inspect.signature(CrossEncoder).parameters
                if "local_files_only" in parameters:
                    kwargs["local_files_only"] = self.settings.offline
                else:
                    # Sentence Transformers 3.x forwards these to Hugging Face.
                    kwargs.update(automodel_args={"local_files_only": self.settings.offline},
                                  tokenizer_args={"local_files_only": self.settings.offline},
                                  config_args={"local_files_only": self.settings.offline})
                self.model = CrossEncoder(self.settings.reranker_model, **kwargs)
            except Exception as exc:
                raise RuntimeError(f"重排模型加载失败；离线模式需先下载模型: {exc}") from exc
        return self.model

    def rerank(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        if not hits or self.settings.reranker_backend == "none":
            return hits
        if self.settings.reranker_backend == "lexical":
            for hit in hits:
                score = lexical_coverage(query, hit.chunk.text)
                hit.rerank_score = score
                hit.score = .8 * score + .2 * max(0, hit.vector_score or 0)
        else:
            with self.lock:
                model = self._load()
                tokenizer = model.tokenizer
                limit = int(getattr(model, "max_seq_length", None) or getattr(model, "max_length", None)
                            or tokenizer.model_max_length)
                if not 8 <= limit <= 1_000_000:
                    raise RuntimeError("重排模型缺少可靠长度上限，拒绝静默截断")
                windows = pair_windows(tokenizer, query, [hit.chunk.text for hit in hits], limit)
                raw = classifier_scores(model.model, tokenizer, windows, self.settings.embedding_batch_size)
                best = [float("-inf")] * len(hits)
                for owner, score in zip(windows.owners, raw):
                    best[owner] = max(best[owner], float(score))
                for hit, score in zip(hits, best):
                    hit.rerank_score = score
                    # A monotonic score for thresholds, explicitly not a
                    # calibrated relevance or answer-correctness probability.
                    hit.score = sigmoid(score)
        return sorted(hits, key=lambda hit: (-hit.score, -(hit.rrf_score or 0), hit.chunk.chunk_id))

    def health(self) -> dict:
        backend = self.settings.reranker_backend
        available = backend != "cross_encoder" or importlib.util.find_spec("sentence_transformers") is not None
        cached = backend != "cross_encoder" or self.model is not None or model_cached(self.settings.reranker_model)
        return {"backend": backend, "model": "DEMO lexical coverage" if backend == "lexical" else self.settings.reranker_model,
                "available": available, "model_cached": cached, "ready": available and cached,
                "loaded": backend != "cross_encoder" or self.model is not None,
                "demo": backend == "lexical", "score_is_probability": False,
                "window_policy_version": WINDOW_POLICY,
                "long_input_policy": "original query/document token-ID windows + max logit; no decode expansion or silent truncation"}
