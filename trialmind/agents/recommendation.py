"""Recommendation agent.

Turns each site's risk explanation and KPI drivers into concrete, prioritized,
human-reviewable operational actions. Recommendations are advisory — the pipeline
is designed so a human reviews them before anything is acted on.
"""

from __future__ import annotations

from typing import Dict

from trialmind.agents.base import Context, LLMAgent

SYSTEM = (
    "You are a clinical trial operations lead. Given a site's risk drivers and an "
    "explanation of why it is high-risk, propose 2-4 concrete, prioritized actions "
    "to reduce that risk. Each action must target a specific driver named in the "
    "input (e.g. query backlog, low enrollment, protocol deviations). Do not invent "
    "facts. Output a short numbered list; each item is one actionable sentence "
    "starting with a verb. End with a one-line note that these require human review "
    "before execution."
)


def _format(site: Dict, explanation: str) -> str:
    drivers = ", ".join(
        f"{c['feature']}={c['value']} (SHAP {c['shap']:+.2f})"
        for c in site["shap_contributions"][:4]
    )
    return (
        f"Site {site['site_id']} ({site['country']}, {site['region']}), "
        f"predicted risk {site['predicted_prob']:.2f}.\n"
        f"Top risk drivers: {drivers}.\n\n"
        f"Explanation:\n{explanation}\n\n"
        "Propose the prioritized actions."
    )


class RecommendationAgent(LLMAgent):
    name = "recommendations"

    def run(self, context: Context) -> Context:
        analysis = context["analysis"]
        explanations = context.get("explanations", {})
        recommendations: Dict[int, str] = {}
        for site in analysis["high_risk_sites"]:
            sid = site["site_id"]
            recommendations[sid] = self.complete(
                SYSTEM, _format(site, explanations.get(sid, "")), max_tokens=600
            )
        context["recommendations"] = recommendations
        return context
