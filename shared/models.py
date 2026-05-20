"""Domain models for news articles and RAG chunks."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class RawNewsArticle(BaseModel):
    article_id: str
    feed_id: str
    region: str
    source_type: str
    source_url: str
    title: str | None = None
    summary: str | None = None
    body: str | None = None
    published_at: datetime | None = None
    language_hint: str | None = None
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    ingested_at: datetime


class CleanedNewsArticle(BaseModel):
    article_id: str
    feed_id: str
    region: str
    title: str
    content: str
    content_hash: str
    language: str
    published_at: datetime | None
    source_url: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    processed_at: datetime


class RagChunk(BaseModel):
    chunk_id: str
    article_id: str
    feed_id: str
    region: str
    language: str
    chunk_index: int
    chunk_text: str
    token_count: int
    published_at: datetime | None
    source_url: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class RetrievalHit(BaseModel):
    chunk_id: str
    score: float
    chunk_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
