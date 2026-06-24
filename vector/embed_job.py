"""
Databricks / local job: read Gold chunks, embed with Hugging Face, upsert to Pinecone.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from config.settings import get_settings
from observability.langfuse_tracing import LangfuseTraceContext, flush_langfuse
from vector.embed import EmbeddingService
from vector.metadata import build_vector_metadata
from vector.pinecone_client import PineconeVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed Gold chunks and upsert to Pinecone")
    parser.add_argument("--catalog", default="news_insight")
    parser.add_argument("--ingestion_date", default="")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    settings = get_settings()
    gold_fqn = f"{args.catalog}.{settings.gold_schema}.{settings.gold_table}"

    spark = SparkSession.builder.appName("news_insight_vector_upsert").getOrCreate()
    df = spark.table(gold_fqn).filter(F.col("embedded_at").isNull())
    if args.ingestion_date:
        df = df.filter(F.col("ingestion_date") == F.lit(args.ingestion_date))

    rows = [row.asDict() for row in df.limit(args.batch_size * 10).collect()]
    if not rows:
        return 0

    embedder = EmbeddingService(settings)
    store = PineconeVectorStore(settings)
    store.ensure_index()

    total_upserted = 0
    for i in range(0, len(rows), args.batch_size):
        batch = rows[i : i + args.batch_size]
        texts = [r["chunk_text"] for r in batch]
        vectors = embedder.embed_texts(texts)

        with LangfuseTraceContext(
            "vector-pipeline-batch",
            metadata={"batch_index": i // args.batch_size, "size": len(batch)},
            tags=["vector", "pipeline"],
        ):
            by_region: dict[str, list[tuple[str, list[float], dict]]] = {}
            for row, vec in zip(batch, vectors):
                region = row["region"]
                meta = build_vector_metadata(
                    chunk_id=row["chunk_id"],
                    article_id=row["article_id"],
                    feed_id=row["feed_id"],
                    region=region,
                    language=row["language"],
                    chunk_index=int(row["chunk_index"]),
                    published_at=str(row.get("published_at") or ""),
                    source_url=str(row.get("source_url") or ""),
                    extra={"chunk_text": row["chunk_text"][:1000]},
                )
                by_region.setdefault(region, []).append((row["chunk_id"], vec, meta))

            for region, payload in by_region.items():
                total_upserted += store.upsert_batch(payload, region=region)

        chunk_ids = [r["chunk_id"] for r in batch]
        updates_df = (
            spark.createDataFrame([(cid,) for cid in chunk_ids], ["chunk_id"])
            .withColumn("embedded_at", F.lit(datetime.now(timezone.utc)))
        )
        try:
            from delta.tables import DeltaTable

            (
                DeltaTable.forName(spark, gold_fqn)
                .alias("target")
                .merge(updates_df.alias("source"), "target.chunk_id = source.chunk_id")
                .whenMatchedUpdate(set={"embedded_at": "source.embedded_at"})
                .execute()
            )
        except Exception:
            spark.sql(
                f"UPDATE {gold_fqn} SET embedded_at = current_timestamp() "
                f"WHERE chunk_id IN ({','.join(repr(c) for c in chunk_ids)})"
            )

    flush_langfuse()
    return total_upserted


if __name__ == "__main__":
    print(f"Upserted {run()} vectors")
