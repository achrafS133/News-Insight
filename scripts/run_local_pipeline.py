#!/usr/bin/env python3
"""
Local end-to-end pipeline (no Airflow/Databricks):
ingest JSONL -> chunk -> embed -> Pinecone upsert -> optional agent query.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from config.settings import get_settings
from observability.langfuse_tracing import LangfuseTraceContext, flush_langfuse
from vector.embed import EmbeddingService
from vector.metadata import build_vector_metadata
from vector.pinecone_client import PineconeVectorStore

CHUNK_WORDS = 380
OVERLAP_WORDS = 60
STEP = CHUNK_WORDS - OVERLAP_WORDS


def _normalize(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text.strip())
    return re.sub(r"[^\w\s\-\.,;:!?\'\"]", "", collapsed)


def _chunk_text(text: str) -> list[str]:
    words = text.split()
    if len(words) <= CHUNK_WORDS:
        return [" ".join(words)] if words else []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + CHUNK_WORDS, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += STEP
    return chunks


def load_latest_staging() -> Path:
    root = PROJECT_ROOT / "data" / "bronze_staging"
    files = sorted(root.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError("No staging JSONL found. Run scripts/run_ingest_local.py first.")
    return files[0]


def build_chunks(staging_file: Path, max_articles: int) -> list[dict]:
    seen_hashes: set[str] = set()
    chunks: list[dict] = []
    with staging_file.open(encoding="utf-8") as handle:
        for line in handle:
            if len(chunks) >= max_articles * 5:
                break
            row = json.loads(line)
            content = _normalize(
                row.get("body") or row.get("summary") or row.get("title") or ""
            )
            if len(content) < 50:
                continue
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)

            for idx, chunk_text in enumerate(_chunk_text(content)):
                chunk_id = f"{row['article_id']}_{idx}"
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "article_id": row["article_id"],
                        "feed_id": row["feed_id"],
                        "region": row["region"],
                        "language": (row.get("language_hint") or "unknown").lower(),
                        "chunk_index": idx,
                        "chunk_text": chunk_text,
                        "published_at": row.get("published_at"),
                        "source_url": row.get("source_url"),
                    }
                )
    return chunks


def upsert_chunks(chunks: list[dict], batch_size: int) -> int:
    settings = get_settings()
    if not settings.pinecone_api_key.get_secret_value():
        raise ValueError("PINECONE_API_KEY is not set in .env")

    embedder = EmbeddingService(settings)
    store = PineconeVectorStore(settings)
    store.ensure_index()

    total = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["chunk_text"] for c in batch]
        vectors = embedder.embed_texts(texts)

        by_region: dict[str, list] = {}
        for row, vec in zip(batch, vectors):
            meta = build_vector_metadata(
                chunk_id=row["chunk_id"],
                article_id=row["article_id"],
                feed_id=row["feed_id"],
                region=row["region"],
                language=row["language"],
                chunk_index=row["chunk_index"],
                published_at=str(row.get("published_at") or ""),
                source_url=str(row.get("source_url") or ""),
                extra={"chunk_text": row["chunk_text"][:1000]},
            )
            by_region.setdefault(row["region"], []).append(
                (row["chunk_id"], vec, meta)
            )

        with LangfuseTraceContext(
            "local-pipeline-upsert",
            metadata={"batch": i // batch_size, "size": len(batch)},
            tags=["local", "vector"],
        ):
            for region, payload in by_region.items():
                total += store.upsert_batch(payload, region=region)

    flush_langfuse()
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local news insight pipeline")
    parser.add_argument("--max-articles", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--query",
        default=None,
        help="Optional agent query after upsert",
    )
    parser.add_argument("--skip-upsert", action="store_true")
    parser.add_argument("--skip-agent", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    staging = load_latest_staging()
    print(f"Using staging file: {staging}")

    chunks = build_chunks(staging, args.max_articles)
    print(f"Built {len(chunks)} chunks from staging data")

    if not args.skip_upsert:
        try:
            upserted = upsert_chunks(chunks, args.batch_size)
            print(f"Upserted {upserted} vectors to Pinecone")
        except ValueError as exc:
            print(f"Skipping Pinecone upsert: {exc}")
            if args.query and not args.skip_agent:
                print("Cannot run agent without vectors in Pinecone.")
                return 1

    if args.query and not args.skip_agent:
        if not settings.openai_api_key.get_secret_value():
            print("OPENAI_API_KEY is not set — cannot run agent.")
            return 1
        from agents.router import RouterAgent

        router = RouterAgent()
        result = router.handle(args.query, session_id="local-pipeline")
        print(json.dumps(result, indent=2, default=str))
        flush_langfuse()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
