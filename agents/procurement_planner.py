"""Procurement Planner agent: supply-chain and purchasing recommendations from news RAG."""

from __future__ import annotations

import json
from typing import Any

from agents.base import BaseRagAgent


class ProcurementPlannerAgent(BaseRagAgent):
    agent_name = "procurement_planner"
    system_prompt = (
        "You are a predictive procurement strategist. Using only retrieved news context, "
        "produce JSON with keys: executive_summary, supplier_risks (list), "
        "recommended_actions (list), categories_to_watch (list), "
        "lead_time_impact (low|medium|high), confidence (0-1). "
        "Ground every action in cited article_ids from metadata."
    )

    def parse_response(self, raw: str) -> dict[str, Any]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {
                "executive_summary": raw,
                "supplier_risks": [],
                "recommended_actions": [],
                "categories_to_watch": [],
                "lead_time_impact": "medium",
                "confidence": 0.4,
            }
