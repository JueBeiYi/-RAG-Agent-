from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class SourceBlock(BaseModel):
    text: str
    page: int | None = None
    paragraph: int | None = None
    section: str = ""
    bbox: list[float] | None = None
    kind: str = "text"


class Document(BaseModel):
    document_id: str
    scope: str = "local"
    name: str
    content_hash: str
    format: str
    path: str = ""
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    abstract: str = ""
    doi: str = ""
    status: str = "queued"
    error: str = ""
    chunks: int = 0
    created_at: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    scope: str = "local"
    name: str = ""
    text: str
    page_start: int | None = None
    page_end: int | None = None
    paragraph: int | None = None
    section: str = ""
    bbox: list[float] | None = None
    sources: list[SourceBlock] = Field(default_factory=list)
    strategy: str = "recursive"
    index_version: str = ""


class RetrievalHit(BaseModel):
    chunk: Chunk
    score: float = 0.0
    vector_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    ranks: dict[str, int] = Field(default_factory=dict)


class Citation(BaseModel):
    citation_id: str
    document_id: str
    chunk_id: str
    name: str
    text: str
    page_start: int | None = None
    page_end: int | None = None
    paragraph: int | None = None
    bbox: list[float] | None = None
    sources: list[SourceBlock] = Field(default_factory=list)
    verified: bool = True


class ToolResult(BaseModel):
    tool: str
    content: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    citations: list[Citation] = Field(default_factory=list)
    error: str = ""
    retryable: bool = False
    elapsed_ms: float = 0.0


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20000)
    document_ids: list[str] = Field(default_factory=list)
    retrieval_mode: str | None = None
    agent: bool = True


def event(kind: str, **data: Any) -> dict:
    return {"type": kind, **data}
