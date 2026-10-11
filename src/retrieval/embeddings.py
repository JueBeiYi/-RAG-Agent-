from __future__ import annotations

import hashlib
import importlib.util
import json
import threading
from collections import Counter
from pathlib import Path

import numpy as np

from src.core.config import Settings
from .tokenization import tokenize
from .token_windows import WINDOW_POLICY, single_windows, sentence_vectors


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def model_cached(model_name: str) -> bool:
    """Inspect local weights without downloading anything or loading PyTorch."""
    path = Path(model_name)
    if path.is_dir():
        return any(path.rglob("*.safetensors")) or any(path.rglob("pytorch_model*.bin"))
    try:
        from huggingface_hub import try_to_load_from_cache

        config = try_to_load_from_cache(model_name, "config.json")
        if not isinstance(config, str):
            return False
        for filename in ("model.safetensors", "pytorch_model.bin"):
            if isinstance(try_to_load_from_cache(model_name, filename), str):
                return True
        for filename in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            index = try_to_load_from_cache(model_name, filename)
            if isinstance(index, str):
                shards = set(json.loads(Path(index).read_text(encoding="utf-8")).get("weight_map", {}).values())
                if shards and all(isinstance(try_to_load_from_cache(model_name, shard), str) for shard in shards):
                    return True
    except (ImportError, ValueError, OSError):
        pass
    return False


def local_model_revision(model_name: str) -> str:
    """Cache commit or local file signature, used to keep model versions apart."""
    path = Path(model_name)
    if path.is_dir():
        files = sorted(list(path.rglob("*.safetensors")) + list(path.rglob("pytorch_model*.bin"))
                       + list(path.rglob("config.json")))
        signature = "|".join(f"{file.relative_to(path)}:{file.stat().st_size}:{file.stat().st_mtime_ns}"
                             for file in files)
        return "local-" + hashlib.sha256(signature.encode()).hexdigest()[:20]
    try:
        from huggingface_hub import try_to_load_from_cache
        config = try_to_load_from_cache(model_name, "config.json")
        if isinstance(config, str):
            config_path = Path(config)
            # Hugging Face's snapshots/<commit>/config.json path binds the
            # index to downloaded artifact revision rather than a mutable name.
            parts = config_path.parts
            revision = parts[parts.index("snapshots") + 1] if "snapshots" in parts else "cached"
            return revision + "-" + hashlib.sha256(config_path.read_bytes()).hexdigest()[:12]
    except (ImportError, ValueError, OSError):
        pass
    return "not-prepared"


class Embedder:
    """Lazy model loading; long text is fully represented by window pooling.

    The demo hash backend is deterministic feature hashing, not a pretrained
    semantic embedding model. Its fingerprint cannot share a real-model index.
    """

    def __init__(self, settings: Settings):
        if settings.embedding_backend not in {"hash", "sentence_transformers"}:
            raise ValueError("embedding_backend 可选 hash / sentence_transformers")
        self.settings = settings
        self.model = None
        self.lock = threading.RLock()
        self.last_window_count = 0
        self.long_input_count = 0
        self.revision = "demo-v2" if settings.embedding_backend == "hash" else local_model_revision(settings.embedding_model)
        identity = "demo-hash-384-v2" if settings.embedding_backend == "hash" else f"{settings.embedding_model}|{self.revision}|{WINDOW_POLICY}|cap={settings.embedding_max_tokens}"
        self.fingerprint = hashlib.sha256(identity.encode()).hexdigest()[:24]

    def _load(self):
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                import torch
                torch.set_num_threads(max(1, self.settings.cpu_threads))
            except ImportError as exc:
                raise RuntimeError("真实 Embedding 需要安装 requirements-models.txt；可用 RAG_PROFILE=demo 验证流程") from exc
            try:
                self.model = SentenceTransformer(self.settings.embedding_model,
                                                 device=self.settings.embedding_device,
                                                 local_files_only=self.settings.offline,
                                                 trust_remote_code=False)
            except Exception as exc:
                mode = "离线模式只读取本地缓存；先运行 prepare_models 命令准备模型" if self.settings.offline else "请检查模型名称和网络"
                raise RuntimeError(f"Embedding 模型加载失败。{mode}。原始错误: {exc}") from exc
        return self.model

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            dimensions = 384 if self.settings.embedding_backend == "hash" else (
                self.model.get_sentence_embedding_dimension() if self.model is not None else 0)
            return np.empty((0, dimensions), dtype=np.float32)
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("encode 需要字符串列表")
        if self.settings.embedding_backend == "hash":
            vectors = np.zeros((len(texts), 384), dtype=np.float32)
            for row, text in enumerate(texts):
                for term, frequency in Counter(tokenize(text)).items():
                    digest = hashlib.blake2b(term.encode(), digest_size=8).digest()
                    slot = int.from_bytes(digest[:4], "little") % 384
                    sign = 1 if digest[4] & 1 else -1
                    vectors[row, slot] += sign * (1 + np.log(frequency))
            self.last_window_count = len(texts)
            return normalize(vectors)
        with self.lock:
            model = self._load()
            tokenizer = model.tokenizer
            model_limit = int(model.max_seq_length or tokenizer.model_max_length)
            limit = min(model_limit, int(self.settings.embedding_max_tokens))
            if not 8 <= limit <= 1_000_000:
                raise RuntimeError("模型未提供可靠输入长度上限，拒绝静默截断")
            windows = single_windows(tokenizer, texts, limit)
            self.long_input_count += windows.long_input_count
            self.last_window_count = len(windows.features)
            return sentence_vectors(model, windows, self.settings.embedding_batch_size)

    def encode_query(self, query: str) -> np.ndarray:
        if self.settings.embedding_backend == "sentence_transformers" and "bge-large-zh" in self.settings.embedding_model.lower():
            query = "为这个句子生成表示以用于检索相关文章：" + query
        return self.encode([query])[0]

    def health(self) -> dict:
        demo = self.settings.embedding_backend == "hash"
        available = demo or importlib.util.find_spec("sentence_transformers") is not None
        cached = demo or self.model is not None or model_cached(self.settings.embedding_model)
        return {"backend": self.settings.embedding_backend, "model": "DEMO feature hashing (384)" if demo else self.settings.embedding_model,
                "demo": demo, "available": available, "loaded": demo or self.model is not None,
                "model_cached": cached, "ready": available and cached,
                "offline": self.settings.offline, "fingerprint": self.fingerprint,
                "artifact_revision": self.revision,
                "encoding_window_tokens": self.settings.embedding_max_tokens,
                "window_policy_version": WINDOW_POLICY,
                "long_input_policy": "original token-ID windows + weighted mean (all tokens and UNK retained, no decode roundtrip)",
                "long_inputs_processed": self.long_input_count, "last_windows": self.last_window_count}
