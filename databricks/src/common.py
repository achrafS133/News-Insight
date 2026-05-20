"""Shared Spark utilities for medallion jobs."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType


def get_spark(app_name: str) -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .getOrCreate()
    )


def normalize_text(column: F.Column) -> F.Column:
    collapsed = F.regexp_replace(F.trim(column), r"\s+", " ")
    return F.regexp_replace(collapsed, r"[^\w\s\-\.,;:!?\'\"]", "")


@F.udf(StringType())
def content_hash_udf(text: str | None) -> str:
    if not text:
        return ""
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def read_json_staging(spark: SparkSession, staging_path: str) -> DataFrame:
    return (
        spark.read.option("multiline", "false")
        .json(staging_path)
        .withColumn("ingestion_date", F.to_date(F.col("ingestion_date")))
    )


def merge_delta(
    df: DataFrame,
    target_fqn: str,
    merge_condition: str,
    spark: SparkSession,
) -> None:
    if not spark.catalog.tableExists(target_fqn):
        df.write.format("delta").mode("overwrite").saveAsTable(target_fqn)
        return

    (
        df.write.format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .saveAsTable(target_fqn)
    )


def widget_get(name: str, default: str = "") -> str:
    try:
        from pyspark.dbutils import DBUtils  # type: ignore

        dbutils = DBUtils(SparkSession.getActiveSession())
        return dbutils.widgets.get(name)
    except Exception:
        return default
