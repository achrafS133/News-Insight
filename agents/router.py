"""Router agent: classifies user intent and delegates to specialists."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from agents.base import BaseRagAgent
from agents.procurement_planner import ProcurementPlannerAgent
from agents.trend_analyzer import TrendAnalyzerAgent
from config.settings import get_settings
from observability.langfuse_tracing import LangfuseTraceContext, flush_langfuse


class AgentRoute(str, Enum):
    TREND_ANALYZER = "trend_analyzer"
    PROCUREMENT_PLANNER = "procurement_planner"


class RouteDecision(BaseModel):
    route: AgentRoute
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    detected_language: str | None = None
    detected_region: str | None = None


class RouterAgent(BaseRagAgent):
    agent_name = "router"
    system_prompt = (
        "You are a routing agent for a multilingual news insight platform. "
        "Classify the user request into exactly one route:\n"
        "- trend_analyzer: market/news trends, sentiment, macro signals, sector momentum\n"
        "- procurement_planner: supply chain, sourcing, purchasing, inventory, supplier risk\n"
        "Return JSON: {\"route\": \"...\", \"confidence\": 0.0-1.0, \"rationale\": \"...\", "
        "\"detected_language\": \"en|es|...\", \"detected_region\": \"global|emea|...\"}"
    )

    def __init__(self) -> None:
        super().__init__(get_settings())
        self._trend = TrendAnalyzerAgent(self.settings)
        self._procurement = ProcurementPlannerAgent(self.settings)

    def parse_response(self, raw: str) -> dict[str, Any]:
        try:
            payload = json.loads(raw)
            decision = RouteDecision.model_validate(payload)
            return decision.model_dump()
        except Exception:
            lowered = raw.lower()
            route = (
                AgentRoute.PROCUREMENT_PLANNER
                if any(k in lowered for k in ("procure", "supplier", "sourcing", "inventory"))
                else AgentRoute.TREND_ANALYZER
            )
            return RouteDecision(
                route=route,
                confidence=0.5,
                rationale="Fallback keyword routing",
            ).model_dump()

    def route(self, user_query: str, *, session_id: str | None = None) -> RouteDecision:
        result = self.run(user_query, session_id=session_id)
        return RouteDecision.model_validate(result["response"])

    def handle(
        self,
        user_query: str,
        *,
        region: str | None = None,
        language: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        with LangfuseTraceContext(
            "router-handle",
            session_id=session_id,
            input_data={"query": user_query},
            tags=["agent", "router"],
        ) as trace:
            decision = self.route(user_query, session_id=session_id)
            region = region or decision.detected_region
            language = language or decision.detected_language

            if decision.route == AgentRoute.PROCUREMENT_PLANNER:
                specialist_output = self._procurement.run(
                    user_query,
                    region=region,
                    language=language,
                    session_id=session_id,
                )
            else:
                specialist_output = self._trend.run(
                    user_query,
                    region=region,
                    language=language,
                    session_id=session_id,
                )

            output = {
                "routing": decision.model_dump(),
                "specialist": specialist_output,
            }
            trace.update_output(output)
            flush_langfuse()
            return output
