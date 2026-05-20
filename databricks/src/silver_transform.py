"""
Silver transform: clean text, detect language hint, deduplicate by content hash.
"""

from __future__ import annotations

from pyspark.sql import Window
from pyspark.sql import functions as F

try:
    from databricks.src.common import (
        content_hash_udf,
        get_spark,
        merge_delta,
        normalize_text,
        widget_get,
    )
except ImportError:
    from common import (
        content_hash_udf,
        get_spark,
        merge_delta,
        normalize_text,
        widget_get,
    )

CATALOG = widget_get("catalog", "news_insight")
INGESTION_DATE = widget_get("ingestion_date", "")

BRONZE_FQN = f"{CATALOG}.bronze.raw_news"
SILVER_FQN = f"{CATALOG}.silver.cleaned_news"


def run() -> int:
    spark = get_spark("news_insight_silver_transform")

    bronze_df = spark.table(BRONZE_FQN)
    if INGESTION_DATE:
        bronze_df = bronze_df.filter(F.col("ingestion_date") == F.lit(INGESTION_DATE))

    content_col = F.coalesce(F.col("body"), F.col("summary"), F.col("title"))
    cleaned = (
        bronze_df.withColumn("title_clean", normalize_text(F.coalesce(F.col("title"), F.lit(""))))
        .withColumn("content_clean", normalize_text(content_col))
        .withColumn("content_hash", content_hash_udf(F.col("content_clean")))
        .withColumn(
            "language",
            F.coalesce(F.lower(F.col("language_hint")), F.lit("unknown")),
        )
        .withColumn(
            "metadata",
            F.create_map(
                F.lit("feed_id"),
                F.col("feed_id"),
                F.lit("region"),
                F.col("region"),
            ),
        )
        .withColumn("processed_at", F.current_timestamp())
        .filter(F.length(F.col("content_clean")) > 50)
    )

    window = Window.partitionBy("content_hash", "ingestion_date").orderBy(F.col("ingested_at").desc())
    deduped = (
        cleaned.withColumn("row_num", F.row_number().over(window))
        .filter(F.col("row_num") == 1)
        .select(
            "article_id",
            "feed_id",
            "region",
            F.col("title_clean").alias("title"),
            F.col("content_clean").alias("content"),
            "content_hash",
            "language",
            "published_at",
            "source_url",
            "metadata",
            "processed_at",
            "ingestion_date",
        )
    )

    merge_delta(deduped, SILVER_FQN, "target.article_id = source.article_id", spark)
    return deduped.count()


if __name__ == "__main__":
    count = run()
    print(f"Silver transform complete: {count} rows")
