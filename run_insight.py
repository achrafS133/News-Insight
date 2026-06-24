#!/usr/bin/env python3
"""CLI entrypoint for the multi-agent news insight engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from agents.router import RouterAgent
from observability.langfuse_tracing import flush_langfuse


def main() -> int:
    parser = argparse.ArgumentParser(description="Live News-to-Insight Engine")
    parser.add_argument("query", help="Natural language question")
    parser.add_argument("--region", default=None, help="Pinecone namespace region filter")
    parser.add_argument("--language", default=None, help="Metadata language filter")
    parser.add_argument("--session-id", default=None, help="Langfuse session id")
    args = parser.parse_args()

    router = RouterAgent()
    result = router.handle(
        args.query,
        region=args.region,
        language=args.language,
        session_id=args.session_id,
    )
    print(json.dumps(result, indent=2, default=str))
    flush_langfuse()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
