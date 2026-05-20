"""Base RAG + LLM utilities for specialist agents."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

import tiktoken
from openai import OpenAI

from config.settings import Settings, get_settings
from observability.langfuse_tracing import flush_langfuse, generation_span
from shared.models import RetrievalHit
from shared.retry import api_retry
from vector.embed import EmbeddingService
from vector.pinecone_client import PineconeVectorStore


class BaseRagAgent(ABC):
    agent_name: str = "base"
    system_prompt: str = ""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.embedder = EmbeddingService(self.settings)
        self.vector_store = PineconeVectorStore(self.settings)
        self._openai = OpenAI(api_key=self.settings.openai_api_key.get_secret_value())
        self._encoding = tiktoken.encoding_for_model("gpt-4o-mini")

    def count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))

    def retrieve(
        self,
        query: str,
        *,
        region: str | None = None,
        language: str | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalHit]:
        embedding = self.embedder.embed_query(query)
        return self.vector_store.query(
            embedding,
            region=region,
            language=language,
            top_k=top_k,
        )

    def _format_context(self, hits: list[RetrievalHit]) -> str:
        blocks: list[str] = []
        for hit in hits:
            blocks.append(
                f"[score={hit.score:.3f}] {hit.chunk_text}\nmeta={json.dumps(hit.metadata)}"
            )
        return "\n\n".join(blocks) if blocks else "No relevant news context found."

    @api_retry(max_attempts=3)
    def _call_llm(self, messages: list[dict[str, str]], *, trace_name: str) -> str:
        with generation_span(
            trace_name,
            model=self.settings.openai_model,
            input_messages=messages,
            metadata={"agent": self.agent_name},
        ) as generation:
            response = self._openai.chat.completions.create(
                model=self.settings.openai_model,
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            choice = response.choices[0].message.content or ""
            usage = response.usage
            if generation is not None:
                generation.update(
                    output=choice,
                    usage_details={
                        "input": usage.prompt_tokens if usage else 0,
                        "output": usage.completion_tokens if usage else 0,
                        "total": usage.total_tokens if usage else 0,
                    },
                )
            flush_langfuse()
            return choice

    def run(
        self,
        user_query: str,
        *,
        region: str | None = None,
        language: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        hits = self.retrieve(user_query, region=region, language=language)
        context = self._format_context(hits)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Query: {user_query}\n\n"
                    f"Retrieved news context:\n{context}\n\n"
                    "Respond with structured JSON."
                ),
            },
        ]
        raw = self._call_llm(
            messages,
            trace_name=f"{self.agent_name}-generation",
        )
        return {
            "agent": self.agent_name,
            "query": user_query,
            "region": region,
            "language": language,
            "session_id": session_id,
            "retrieval_count": len(hits),
            "response": self.parse_response(raw),
            "raw_llm_output": raw,
        }

    @abstractmethod
    def parse_response(self, raw: str) -> dict[str, Any]:
        """Parse LLM JSON output into a typed dict."""
