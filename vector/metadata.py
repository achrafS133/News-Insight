"""Pinecone metadata and namespace conventions."""

from __future__ import annotations

from typing import Any


def namespace_for_region(region: str) -> str:
    normalized = region.strip().lower().replace(" ", "_")
    return normalized or "global"


def build_vector_metadata(
    *,
    chunk_id: str,
    article_id: str,
    feed_id: str,
    region: str,
    language: str,
    chunk_index: int,
    published_at: str | None,
    source_url: str | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "chunk_id": chunk_id,
        "article_id": article_id,
        "feed_id": feed_id,
        "region": region,
        "language": language,
        "chunk_index": chunk_index,
    }
    if published_at:
        metadata["published_at"] = published_at
    if source_url:
        metadata["source_url"] = source_url
    if extra:
        metadata.update(extra)
    return metadata
