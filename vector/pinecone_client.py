"""Pinecone upsert and semantic search with metadata filters."""

from __future__ import annotations

from typing import Any

from pinecone import Pinecone, ServerlessSpec

from config.settings import Settings, get_settings
from observability.langfuse_tracing import LangfuseTraceContext, flush_langfuse
from shared.models import RetrievalHit
from shared.retry import api_retry
from vector.metadata import namespace_for_region


class PineconeVectorStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = Pinecone(api_key=self.settings.pinecone_api_key.get_secret_value())
        self._index = self._resolve_index()

    def _resolve_index(self):
        if self.settings.pinecone_host:
            return self._client.Index(host=self.settings.pinecone_host)
        return self._client.Index(self.settings.pinecone_index_name)

    def ensure_index(self) -> None:
        name = self.settings.pinecone_index_name
        index_list = self._client.list_indexes()
        if hasattr(index_list, "names"):
            existing = set(index_list.names())
        else:
            existing = {idx.name for idx in index_list}
        if name not in existing:
            self._client.create_index(
                name=name,
                dimension=self.settings.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

    @api_retry(max_attempts=3)
    def upsert_batch(
        self,
        vectors: list[tuple[str, list[float], dict[str, Any]]],
        *,
        region: str,
    ) -> int:
        namespace = namespace_for_region(region)
        with LangfuseTraceContext(
            "pinecone-upsert-batch",
            metadata={"namespace": namespace, "vector_count": len(vectors)},
            tags=["vector", "pinecone"],
        ):
            payload = [
                {"id": vid, "values": values, "metadata": meta}
                for vid, values, meta in vectors
            ]
            self._index.upsert(vectors=payload, namespace=namespace)
            flush_langfuse()
            return len(payload)

    @api_retry(max_attempts=3)
    def query(
        self,
        embedding: list[float],
        *,
        region: str | None = None,
        language: str | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalHit]:
        namespace = namespace_for_region(region or self.settings.pinecone_namespace_default)
        filter_dict: dict[str, Any] = {}
        if language:
            filter_dict["language"] = {"$eq": language}

        with LangfuseTraceContext(
            "pinecone-query",
            metadata={"namespace": namespace, "top_k": top_k, "filter": filter_dict},
            tags=["vector", "retrieval"],
        ):
            response = self._index.query(
                vector=embedding,
                top_k=top_k or self.settings.retrieval_top_k,
                include_metadata=True,
                namespace=namespace,
                filter=filter_dict or None,
            )
            hits: list[RetrievalHit] = []
            for match in response.matches or []:
                meta = match.metadata or {}
                hits.append(
                    RetrievalHit(
                        chunk_id=str(meta.get("chunk_id", match.id)),
                        score=float(match.score or 0.0),
                        chunk_text=str(meta.get("chunk_text", "")),
                        metadata=dict(meta),
                    )
                )
            flush_langfuse()
            return hits
