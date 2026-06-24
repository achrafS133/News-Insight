"""
Bronze ingest: load staged JSONL from Airflow into Delta raw_news.
Run on Databricks as a notebook or python wheel task.
"""

from __future__ import annotations

from pyspark.sql import functions as F

try:
    from databricks.src.common import get_spark, merge_delta, read_json_staging, widget_get
except ImportError:
    from common import get_spark, merge_delta, read_json_staging, widget_get

CATALOG = widget_get("catalog", "news_insight")
STAGING_PATH = widget_get("staging_path", "")
INGESTION_DATE = widget_get("ingestion_date", "")

BRONZE_FQN = f"{CATALOG}.bronze.raw_news"


def run() -> int:
    spark = get_spark("news_insight_bronze_ingest")
    if not STAGING_PATH:
        raise ValueError("staging_path widget is required")

    raw_df = read_json_staging(spark, STAGING_PATH)
    bronze_df = (
        raw_df.select(
            "article_id",
            "feed_id",
            "region",
            "source_type",
            "source_url",
            "title",
            "summary",
            "body",
            F.to_timestamp("published_at").alias("published_at"),
            "language_hint",
            F.to_json(F.col("raw_payload")).alias("raw_payload"),
            F.to_timestamp("ingested_at").alias("ingested_at"),
            F.coalesce(F.col("ingestion_date"), F.lit(INGESTION_DATE)).alias("ingestion_date"),
        )
        .dropDuplicates(["article_id", "ingestion_date"])
    )

    merge_delta(bronze_df, BRONZE_FQN, "target.article_id = source.article_id", spark)
    return bronze_df.count()


if __name__ == "__main__":
    count = run()
    print(f"Bronze ingest complete: {count} rows")
