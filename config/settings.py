"""Central configuration for the News-to-Insight engine."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Databricks / Delta
    databricks_host: str = ""
    databricks_token: SecretStr = SecretStr("")
    news_insight_catalog: str = "news_insight"
    bronze_schema: str = "bronze"
    silver_schema: str = "silver"
    gold_schema: str = "gold"
    bronze_table: str = "raw_news"
    silver_table: str = "cleaned_news"
    gold_table: str = "rag_chunks"

    # Embeddings
    embedding_model_name: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    embedding_dimension: int = 384
    embedding_batch_size: int = 64
    hf_token: SecretStr | None = None

    # Pinecone
    pinecone_api_key: SecretStr = SecretStr("")
    pinecone_index_name: str = "news-insight-multilingual"
    pinecone_host: str = ""
    pinecone_namespace_default: str = "global"

    # Langfuse
    langfuse_public_key: str = ""
    langfuse_secret_key: SecretStr = SecretStr("")
    langfuse_host: str = "https://cloud.langfuse.com"

    # LLM
    openai_api_key: SecretStr = SecretStr("")
    openai_model: str = "gpt-4o-mini"
    openai_max_retries: int = 3
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64
    retrieval_top_k: int = 8

    # Pipeline
    environment: Literal["dev", "staging", "prod"] = "dev"

    @property
    def bronze_fqn(self) -> str:
        return f"{self.news_insight_catalog}.{self.bronze_schema}.{self.bronze_table}"

    @property
    def silver_fqn(self) -> str:
        return f"{self.news_insight_catalog}.{self.silver_schema}.{self.silver_table}"

    @property
    def gold_fqn(self) -> str:
        return f"{self.news_insight_catalog}.{self.gold_schema}.{self.gold_table}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
