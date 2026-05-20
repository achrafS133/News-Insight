"""Hugging Face multilingual embeddings for news chunks."""

from __future__ import annotations

from typing import Sequence

from sentence_transformers import SentenceTransformer

from config.settings import Settings, get_settings
from observability.langfuse_tracing import LangfuseTraceContext, flush_langfuse


class EmbeddingService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            token = None
            if self.settings.hf_token:
                token = self.settings.hf_token.get_secret_value()
            self._model = SentenceTransformer(
                self.settings.embedding_model_name,
                token=token,
            )
        return self._model

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        with LangfuseTraceContext(
            "hf-embed-batch",
            metadata={
                "model": self.settings.embedding_model_name,
                "batch_size": len(texts),
            },
            tags=["embedding", "huggingface"],
        ):
            vectors = self.model.encode(
                list(texts),
                batch_size=self.settings.embedding_batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            flush_langfuse()
            return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]
