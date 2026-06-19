"""Live tests for the Claude LLM agents.

These make real Anthropic API calls and are SKIPPED automatically when no
ANTHROPIC_API_KEY is configured (so the suite stays green without credentials,
without mocking the model).
"""

from __future__ import annotations

import pytest

from trialmind.config import settings

pytestmark = pytest.mark.skipif(
    not settings.has_llm_credentials, reason="ANTHROPIC_API_KEY not set"
)

# A real-shaped analysis context (the structure produced by AnalysisAgent).
_SITE = {
    "site_id": 12,
    "predicted_prob": 0.87,
    "country": "USA",
    "region": "North America",
    "features": {
        "enrollment_attainment": 0.58,
        "query_backlog": 41,
        "protocol_deviation_rate": 0.11,
        "dropout_rate": 0.29,
        "avg_visit_delay_days": 4.2,
    },
    "shap_contributions": [
        {"feature": "enrollment_attainment", "value": 0.58, "shap": 0.21},
        {"feature": "query_backlog", "value": 41, "shap": 0.15},
        {"feature": "dropout_rate", "value": 0.29, "shap": 0.12},
        {"feature": "protocol_deviation_rate", "value": 0.11, "shap": 0.06},
        {"feature": "avg_visit_delay_days", "value": 4.2, "shap": 0.03},
    ],
}


def _context():
    return {
        "analysis": {
            "n_sites": 40,
            "n_high_risk_sites": 1,
            "high_risk_sites": [_SITE],
            "site_metrics": {"roc_auc": 0.93, "pr_auc": 0.88},
            "dropout_metrics": {"roc_auc": 0.75, "pr_auc": 0.47},
            "dropout": {
                "n_patients": 3000,
                "n_high_risk": 280,
                "threshold": 0.5,
                "global_importance": {"travel_distance_km": 0.3, "digital_engagement_score": 0.28},
                "examples": [],
            },
        }
    }


def test_explainability_agent_real_call():
    from trialmind.agents import ExplainabilityAgent

    out = ExplainabilityAgent().run(_context())
    text = out["explanations"][12]
    assert isinstance(text, str) and len(text.strip()) > 0


def test_recommendation_agent_real_call():
    from trialmind.agents import ExplainabilityAgent, RecommendationAgent

    ctx = ExplainabilityAgent().run(_context())
    ctx = RecommendationAgent().run(ctx)
    assert ctx["recommendations"][12].strip()


def test_exec_summary_agent_real_call():
    from trialmind.agents import ExecSummaryAgent

    ctx = _context()
    ctx["recommendations"] = {12: "1. Clear the query backlog."}
    out = ExecSummaryAgent().run(ctx)
    assert out["exec_summary"].strip()
