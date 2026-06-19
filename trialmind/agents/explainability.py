"""Explainability agent.

For each top high-risk site, asks Claude to explain *why* the model flagged it,
grounded strictly in the supplied KPI values and SHAP drivers. The system prompt
forbids inventing facts beyond the provided data.
"""

from __future__ import annotations

from typing import Dict

from trialmind.agents.base import Context, LLMAgent

SYSTEM = (
    "You are a clinical trial operations analyst. You explain, in plain language, "
    "why a predictive model flagged a research site as high-risk. Ground every "
    "statement strictly in the provided KPI values and SHAP drivers — do not invent "
    "numbers, causes, or context that is not given. Be concise (3-5 sentences), "
    "specific, and neutral. SHAP values quantify how much each feature pushed the "
    "risk prediction up (positive) or down (negative)."
)

_FEATURE_LABELS = {
    "enrollment_attainment": "enrollment attainment (enrolled / target)",
    "query_backlog": "open data-query backlog",
    "protocol_deviation_rate": "protocol-deviation rate",
    "dropout_rate": "patient dropout rate",
    "avg_visit_delay_days": "average visit delay (days)",
}


def _format_site(site: Dict) -> str:
    lines = [
        f"Site {site['site_id']} ({site['country']}, {site['region']})",
        f"Predicted high-risk probability: {site['predicted_prob']:.2f}",
        "",
        "Operational KPIs and their SHAP contribution to the risk score "
        "(sorted by impact):",
    ]
    for c in site["shap_contributions"]:
        label = _FEATURE_LABELS.get(c["feature"], c["feature"])
        direction = "raises" if c["shap"] > 0 else "lowers"
        lines.append(
            f"  - {label}: value={c['value']}, SHAP={c['shap']:+.3f} ({direction} risk)"
        )
    lines.append("")
    lines.append(
        "Explain why this site is high-risk, citing the top drivers above."
    )
    return "\n".join(lines)


class ExplainabilityAgent(LLMAgent):
    name = "explanations"

    def run(self, context: Context) -> Context:
        analysis = context["analysis"]
        explanations: Dict[int, str] = {}
        for site in analysis["high_risk_sites"]:
            explanations[site["site_id"]] = self.complete(
                SYSTEM, _format_site(site), max_tokens=600
            )
        context["explanations"] = explanations
        return context
