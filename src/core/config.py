from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    data_dir: str = str(ROOT / "data" / "runtime")
    embedding_backend: str = "sentence_transformers"
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 16
    embedding_max_tokens: int = 512
    cpu_threads: int = 8
    vector_backend: str = "chroma"
    reranker_backend: str = "cross_encoder"
    reranker_model: str = "BAAI/bge-reranker-base"
    llm_backend: str = "ollama"
    llm_model: str = "qwen2.5:7b-instruct-q4_K_M"
    ollama_url: str = "http://127.0.0.1:11434"
    cloud_url: str = ""
    cloud_api_key: str = ""
    offline: bool = True
    temperature: float = 0.1
    top_p: float = 0.9
    top_k_sampling: int = 40
    context_tokens: int = 4096
    answer_tokens: int = 768
    chunk_strategy: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64
    pdf_parser: str = "pymupdf"
    retrieval_mode: str = "hybrid_rerank"
    candidate_k: int = 20
    result_k: int = 5
    rrf_k: int = 60
    minimum_relevance: float = 0.15
    cache_enabled: bool = True
    cache_threshold: float = 0.96
    cache_ttl_seconds: int = 86400
    max_upload_mb: int = 50
    max_agent_steps: int = 6
    agent_routing_enabled: bool = True
    tool_timeout_seconds: float = 90.0
    request_timeout_seconds: float = 240.0
    max_tool_workers: int = 2
    memory_tokens: int = 1400
    memory_recent_messages: int = 8
    network_search_enabled: bool = False

    @property
    def path(self) -> Path:
        return Path(self.data_dir).resolve()

    def public_dict(self) -> dict:
        result = asdict(self)
        result.pop("cloud_api_key", None)
        return result


def load_settings(path: str | Path | None = None) -> Settings:
    values = {}
    config = Path(path or os.environ.get("RAG_CONFIG", ROOT / "config.yaml"))
    if config.exists():
        import yaml
        values = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    valid = {f.name for f in fields(Settings)}
    unknown = set(values) - valid
    if unknown:
        raise ValueError(f"未知配置字段: {sorted(unknown)}")
    for f in fields(Settings):
        value = os.environ.get("RAG_" + f.name.upper())
        if value is not None:
            default = getattr(Settings(), f.name)
            if isinstance(default, bool):
                if value.lower() not in {"true", "false", "1", "0"}:
                    raise ValueError(f"RAG_{f.name.upper()} 必须为 true/false")
                values[f.name] = value.lower() in {"true", "1"}
            elif isinstance(default, int):
                values[f.name] = int(value)
            elif isinstance(default, float):
                values[f.name] = float(value)
            else:
                values[f.name] = value
    if os.environ.get("RAG_PROFILE") == "demo":
        values.update(embedding_backend="hash", vector_backend="sqlite",
                      reranker_backend="lexical", llm_backend="extractive")
    settings = Settings(**values)
    if settings.chunk_size < 64 or not 0 <= settings.chunk_overlap < settings.chunk_size:
        raise ValueError("分块大小至少 64，重叠必须小于块大小")
    if settings.offline and (settings.llm_backend == "cloud" or settings.network_search_enabled):
        raise ValueError("离线模式不能同时开启云端模型或联网搜索")
    if settings.llm_backend not in {"ollama", "cloud", "extractive"}:
        raise ValueError("不支持的 LLM 后端")
    return settings
