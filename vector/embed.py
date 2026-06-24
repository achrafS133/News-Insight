"""Hugging Face multilingual embeddings for news chunks."""

from __future__ import annotations

from typing import Sequence, Any

from config.settings import Settings, get_settings
from observability.langfuse_tracing import LangfuseTraceContext, flush_langfuse

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:  # pragma: nocover - optional dependency
    SentenceTransformer = None  # type: ignore
    _HAS_ST = False


class EmbeddingService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        if not _HAS_ST:
            raise RuntimeError("sentence-transformers not available")
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
        # If sentence-transformers is available, use local model for embeddings
        if _HAS_ST:
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

        # Fallback to OpenAI embeddings if HF model isn't installed
        # If OpenAI key is missing, return zero vectors to allow offline runs
        key = self.settings.openai_api_key.get_secret_value()
        if not key or not key.strip():
            dim = getattr(self.settings, "embedding_dimension", 384)
            return [[0.0] * dim for _ in texts]

        from openai import OpenAI

        client = OpenAI(api_key=key)
        model_name = getattr(self.settings, "openai_embedding_model", "text-embedding-3-small")
        with LangfuseTraceContext(
            "openai-embed-batch",
            metadata={"model": model_name, "batch_size": len(texts)},
            tags=["embedding", "openai"],
        ):
            resp = client.embeddings.create(model=model_name, input=list(texts))
            flush_langfuse()
            return [item.embedding for item in resp.data]

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]
