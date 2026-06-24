"""Trend Analyzer agent: macro/news trend synthesis from RAG context."""

from __future__ import annotations

import json
from typing import Any

from agents.base import BaseRagAgent


class TrendAnalyzerAgent(BaseRagAgent):
    agent_name = "trend_analyzer"
    system_prompt = (
        "You are a senior market intelligence analyst. Using only the retrieved news context, "
        "produce JSON with keys: summary, key_trends (list), risk_signals (list), "
        "opportunity_signals (list), affected_sectors (list), confidence (0-1). "
        "Be concise and cite article_ids from metadata when possible."
    )

    def parse_response(self, raw: str) -> dict[str, Any]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {
                "summary": raw,
                "key_trends": [],
                "risk_signals": [],
                "opportunity_signals": [],
                "affected_sectors": [],
                "confidence": 0.4,
            }
