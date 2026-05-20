"""
Gold chunking: split silver articles into token-bounded chunks for RAG embedding.
Uses sliding window over words as a Spark-native approximation (no row-wise Python loops).
"""

from __future__ import annotations

from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, IntegerType, StringType

try:
    from databricks.src.common import get_spark, merge_delta, widget_get
except ImportError:
    from common import get_spark, merge_delta, widget_get

CATALOG = widget_get("catalog", "news_insight")
INGESTION_DATE = widget_get("ingestion_date", "")
CHUNK_SIZE = int(widget_get("chunk_size_tokens", "512"))
CHUNK_OVERLAP = int(widget_get("chunk_overlap_tokens", "64"))

SILVER_FQN = f"{CATALOG}.silver.cleaned_news"
GOLD_FQN = f"{CATALOG}.gold.rag_chunks"

WORDS_PER_TOKEN = 0.75  # heuristic: ~1 token per 0.75 words for multilingual text
CHUNK_WORDS = max(int(CHUNK_SIZE * WORDS_PER_TOKEN), 50)
OVERLAP_WORDS = max(int(CHUNK_OVERLAP * WORDS_PER_TOKEN), 10)
STEP_WORDS = max(CHUNK_WORDS - OVERLAP_WORDS, 1)


@F.udf(ArrayType(StringType()))
def chunk_words_udf(text: str) -> list[str]:
    if not text:
        return []
    words = text.split()
    if len(words) <= CHUNK_WORDS:
        return [" ".join(words)]
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + CHUNK_WORDS, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += STEP_WORDS
    return chunks


@F.udf(IntegerType())
def estimate_tokens_udf(text: str) -> int:
    if not text:
        return 0
    return max(int(len(text.split()) / WORDS_PER_TOKEN), 1)


def run() -> int:
    spark = get_spark("news_insight_gold_chunking")

    silver_df = spark.table(SILVER_FQN)
    if INGESTION_DATE:
        silver_df = silver_df.filter(F.col("ingestion_date") == F.lit(INGESTION_DATE))

    chunked = (
        silver_df.withColumn("chunk_list", chunk_words_udf(F.col("content")))
        .select("*", F.posexplode("chunk_list").alias("chunk_index", "chunk_text"))
        .select(
            F.concat_ws(
                "_",
                F.col("article_id"),
                F.col("chunk_index").cast("string"),
            ).alias("chunk_id"),
            "article_id",
            "feed_id",
            "region",
            "language",
            "chunk_index",
            "chunk_text",
            estimate_tokens_udf(F.col("chunk_text")).alias("token_count"),
            "published_at",
            "source_url",
            "metadata",
            F.current_timestamp().alias("created_at"),
            "ingestion_date",
            F.lit(None).cast("timestamp").alias("embedded_at"),
        )
        .filter(F.length(F.col("chunk_text")) > 20)
    )

    merge_delta(chunked, GOLD_FQN, "target.chunk_id = source.chunk_id", spark)
    return chunked.count()


if __name__ == "__main__":
    count = run()
    print(f"Gold chunking complete: {count} chunks")
