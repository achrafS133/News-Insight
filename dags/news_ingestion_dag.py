"""
Airflow DAG: ingest global/regional news feeds -> Bronze volume -> trigger Databricks medallion job.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import feedparser
import yaml
from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable
from airflow.providers.databricks.operators.databricks import DatabricksRunNowOperator
from airflow.providers.http.hooks.http import HttpHook

# Ensure project root is importable when DAGs are parsed from repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from observability.langfuse_tracing import LangfuseTraceContext, flush_langfuse

DEFAULT_ARGS = {
    "owner": "news-insight",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

FEEDS_PATH = PROJECT_ROOT / "config" / "feeds.yaml"
DATABRICKS_CONN_ID = "databricks_default"
MEDALLION_JOB_VAR = "news_insight_medallion_job_id"


def _load_feeds() -> list[dict[str, Any]]:
    with FEEDS_PATH.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return list(payload.get("feeds", []))


def _article_id(feed_id: str, entry: dict[str, Any]) -> str:
    link = entry.get("link") or entry.get("id") or ""
    digest = hashlib.sha256(f"{feed_id}:{link}".encode()).hexdigest()[:32]
    return f"{feed_id}_{digest}"


def _parse_published(entry: dict[str, Any]) -> str | None:
    published = entry.get("published_parsed") or entry.get("updated_parsed")
    if not published:
        return None
    dt = datetime(*published[:6], tzinfo=timezone.utc)
    return dt.isoformat()


@task
def fetch_and_stage_feeds(**context: Any) -> dict[str, int]:
    """Pull RSS feeds and write newline-delimited JSON records to staging path."""
    logical_date = context["logical_date"]
    feeds = _load_feeds()
    records: list[dict[str, Any]] = []
    ingested_at = datetime.now(timezone.utc).isoformat()

    with LangfuseTraceContext(
        "airflow-fetch-feeds",
        session_id=context["run_id"],
        metadata={"logical_date": str(logical_date), "feed_count": len(feeds)},
        tags=["ingestion", "airflow"],
    ) as trace:
        http = HttpHook(method="GET", http_conn_id="http_default")
        for feed in feeds:
            response = http.run(endpoint=feed["url"])
            parsed = feedparser.parse(response.text)
            for entry in parsed.entries:
                entry_dict = dict(entry)
                records.append(
                    {
                        "article_id": _article_id(feed["feed_id"], entry_dict),
                        "feed_id": feed["feed_id"],
                        "region": feed["region"],
                        "source_type": feed["source_type"],
                        "source_url": entry_dict.get("link"),
                        "title": entry_dict.get("title"),
                        "summary": entry_dict.get("summary"),
                        "body": entry_dict.get("content", [{}])[0].get("value")
                        if entry_dict.get("content")
                        else None,
                        "published_at": _parse_published(entry_dict),
                        "language_hint": feed.get("language_hint"),
                        "raw_payload": entry_dict,
                        "ingested_at": ingested_at,
                        "ingestion_date": logical_date.strftime("%Y-%m-%d"),
                    }
                )

        staging_uri = Variable.get(
            "news_insight_bronze_staging_uri",
            default_var="/tmp/news_insight/bronze_staging",
        )
        staging_path = Path(staging_uri) / logical_date.strftime("%Y/%m/%d")
        staging_path.mkdir(parents=True, exist_ok=True)
        out_file = staging_path / f"batch_{context['run_id']}.jsonl"
        with out_file.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, default=str) + "\n")

        trace.update_output({"record_count": len(records), "staging_path": str(out_file)})
        flush_langfuse()
        return {"record_count": len(records), "staging_path": str(out_file)}


with DAG(
    dag_id="news_insight_ingestion",
    description="Ingest news feeds to Bronze and run Databricks medallion pipeline",
    default_args=DEFAULT_ARGS,
    schedule="@hourly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["news-insight", "ingestion", "medallion"],
) as dag:
    stage_task = fetch_and_stage_feeds()

    run_medallion = DatabricksRunNowOperator(
        task_id="run_medallion_pipeline",
        databricks_conn_id=DATABRICKS_CONN_ID,
        job_id="{{ var.value.get('news_insight_medallion_job_id', '') }}",
        notebook_params={
            "ingestion_date": "{{ ds }}",
            "staging_path": "{{ ti.xcom_pull(task_ids='fetch_and_stage_feeds')['staging_path'] }}",
        },
    )

    stage_task >> run_medallion
