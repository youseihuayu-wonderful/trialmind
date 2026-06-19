"""Executive-summary agent.

Synthesizes the full analysis (model performance, high-risk sites, dropout
outlook, and the per-site recommendations) into a single concise, stakeholder-
facing brief. One Claude call.
"""

from __future__ import annotations

from typing import Dict

from trialmind.agents.base import Context, LLMAgent

SYSTEM = (
    "You are writing a one-page executive summary for clinical trial leadership. "
    "Summarize trial operational risk from the structured analysis provided. Be "
    "factual and grounded only in the given numbers; do not invent data. Use four "
    "short sections with headers: 'Overall Risk Posture', 'Top High-Risk Sites', "
    "'Patient Dropout Outlook', and 'Recommended Focus'. Keep it under ~250 words, "
    "decision-oriented, and note that site recommendations require human review."
)


def _format(context: Context) -> str:
    a = context["analysis"]
    sm = a.get("site_metrics", {})
    dm = a.get("dropout_metrics", {})
    drop = a["dropout"]
    recs = context.get("recommendations", {})

    lines = [
        f"Sites in trial: {a['n_sites']}; flagged high-risk: {a['n_high_risk_sites']}.",
        f"Site-risk model ROC-AUC: {sm.get('roc_auc', 'n/a')}, "
        f"PR-AUC: {sm.get('pr_auc', 'n/a')}.",
        f"Dropout model ROC-AUC: {dm.get('roc_auc', 'n/a')}, "
        f"PR-AUC: {dm.get('pr_auc', 'n/a')}.",
        f"Patients: {drop['n_patients']}; predicted high dropout risk: "
        f"{drop['n_high_risk']} (threshold {drop['threshold']}).",
        f"Dropout global drivers: {drop['global_importance']}.",
        "",
        "Top high-risk sites:",
    ]
    for site in a["high_risk_sites"]:
        top = site["shap_contributions"][0]
        lines.append(
            f"  - Site {site['site_id']} ({site['country']}): risk "
            f"{site['predicted_prob']:.2f}, lead driver {top['feature']}={top['value']}."
        )
    lines.append("")
    lines.append("Per-site recommendations were generated for these sites.")
    if recs:
        lines.append(f"Recommendation count: {len(recs)} sites.")
    lines.append("")
    lines.append("Write the executive summary.")
    return "\n".join(lines)


class ExecSummaryAgent(LLMAgent):
    name = "exec_summary"

    def run(self, context: Context) -> Context:
        context["exec_summary"] = self.complete(SYSTEM, _format(context), max_tokens=1200)
        return context
