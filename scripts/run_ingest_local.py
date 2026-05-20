#!/usr/bin/env python3
"""Local smoke test: fetch RSS feeds and write Bronze staging JSONL (no Airflow)."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

FEEDS_PATH = PROJECT_ROOT / "config" / "feeds.yaml"
STAGING_ROOT = PROJECT_ROOT / "data" / "bronze_staging"


def _article_id(feed_id: str, link: str) -> str:
    digest = hashlib.sha256(f"{feed_id}:{link}".encode()).hexdigest()[:32]
    return f"{feed_id}_{digest}"


def main() -> int:
    with FEEDS_PATH.open(encoding="utf-8") as handle:
        feeds = yaml.safe_load(handle).get("feeds", [])

    ingested_at = datetime.now(timezone.utc).isoformat()
    ingestion_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    records: list[dict] = []

    for feed in feeds:
        parsed = feedparser.parse(feed["url"])
        print(f"Fetched {feed['feed_id']}: {len(parsed.entries)} entries")
        for entry in parsed.entries:
            link = entry.get("link") or entry.get("id") or ""
            records.append(
                {
                    "article_id": _article_id(feed["feed_id"], link),
                    "feed_id": feed["feed_id"],
                    "region": feed["region"],
                    "source_type": feed["source_type"],
                    "source_url": link,
                    "title": entry.get("title"),
                    "summary": entry.get("summary"),
                    "body": entry.get("content", [{}])[0].get("value")
                    if entry.get("content")
                    else None,
                    "published_at": None,
                    "language_hint": feed.get("language_hint"),
                    "raw_payload": dict(entry),
                    "ingested_at": ingested_at,
                    "ingestion_date": ingestion_date,
                }
            )

    out_dir = STAGING_ROOT / ingestion_date.replace("-", "/")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"batch_{datetime.now(timezone.utc).strftime('%H%M%S')}.jsonl"
    with out_file.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, default=str) + "\n")

    print(f"Wrote {len(records)} records to {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
