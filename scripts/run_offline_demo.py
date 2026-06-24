#!/usr/bin/env python3
"""Offline demo: ingest + chunk + keyword retrieval (no API keys required)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_local_pipeline import build_chunks, load_latest_staging


def retrieve_local(chunks: list[dict], query: str, top_k: int = 5) -> list[dict]:
    terms = {t.lower() for t in query.split() if len(t) > 3}

    def score(chunk: dict) -> int:
        text = chunk["chunk_text"].lower()
        return sum(1 for term in terms if term in text)

    ranked = sorted(chunks, key=score, reverse=True)
    return [c for c in ranked if score(c) > 0][:top_k] or ranked[:top_k]


def main() -> int:
    query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What supply chain and procurement risks are emerging globally?"
    )
    staging = load_latest_staging()
    chunks = build_chunks(staging, max_articles=25)
    hits = retrieve_local(chunks, query)

    print(f"Query: {query}\n")
    print(f"Staging: {staging}")
    print(f"Chunks available: {len(chunks)}")
    print(f"Top {len(hits)} local retrieval hits:\n")

    for i, hit in enumerate(hits, 1):
        print(f"--- Hit {i} ({hit['region']} / {hit['feed_id']}) ---")
        print(hit["chunk_text"][:500])
        print()

    summary = {
        "mode": "offline_keyword_retrieval",
        "query": query,
        "hit_count": len(hits),
        "feeds_represented": sorted({h["feed_id"] for h in hits}),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
