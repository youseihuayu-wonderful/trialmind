"""Run the full TrialMind pipeline and write a human-reviewable report.

Usage:
    python scripts/run_pipeline.py

Runs deterministic analysis, then the three Claude agents (explainability,
recommendation, executive summary). Without ANTHROPIC_API_KEY it runs the
analysis stage only and prints how to enable the LLM agents. The report is
written to data/artifacts/report.md.
"""

from __future__ import annotations

from trialmind.agents import AnalysisAgent
from trialmind.config import settings
from trialmind.models.evaluation import ARTIFACTS_DIR
from trialmind.orchestrator import build_default_pipeline


def _write_report(context: dict) -> str:
    a = context["analysis"]
    lines = ["# TrialMind — Clinical Trial Risk Report", ""]
    lines.append(
        f"**Sites:** {a['n_sites']}  |  **High-risk sites:** "
        f"{a['n_high_risk_sites']}  |  **Patients:** {a['dropout']['n_patients']}  |  "
        f"**Predicted high dropout risk:** {a['dropout']['n_high_risk']}"
    )
    lines.append("")
    lines.append(
        f"Site-risk model ROC-AUC {a['site_metrics'].get('roc_auc', 'n/a')} · "
        f"Dropout model ROC-AUC {a['dropout_metrics'].get('roc_auc', 'n/a')}"
    )

    if context.get("exec_summary"):
        lines += ["", "## Executive Summary", "", context["exec_summary"]]

    lines += ["", "## High-Risk Sites", ""]
    explanations = context.get("explanations", {})
    recommendations = context.get("recommendations", {})
    for site in a["high_risk_sites"]:
        sid = site["site_id"]
        lines.append(
            f"### Site {sid} ({site['country']}, {site['region']}) — risk "
            f"{site['predicted_prob']:.2f}"
        )
        drivers = ", ".join(
            f"{c['feature']}={c['value']} (SHAP {c['shap']:+.2f})"
            for c in site["shap_contributions"][:3]
        )
        lines.append(f"- Top drivers: {drivers}")
        if sid in explanations:
            lines += ["", f"**Why:** {explanations[sid]}"]
        if sid in recommendations:
            lines += ["", "**Recommended actions:**", "", recommendations[sid]]
        lines.append("")

    report = "\n".join(lines)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "report.md").write_text(report)
    return report


def main() -> None:
    if settings.has_llm_credentials:
        print(f"Running full pipeline with LLM agents (model: {settings.model})...")
        context = build_default_pipeline().run()
    else:
        print(
            "ANTHROPIC_API_KEY not set — running analysis stage only.\n"
            "Set the key in .env to enable the explainability / recommendation / "
            "executive-summary agents.\n"
        )
        context = AnalysisAgent().run({})

    _write_report(context)

    a = context["analysis"]
    print("\n=== Analysis ===")
    print(
        f"sites={a['n_sites']} high_risk_sites={a['n_high_risk_sites']} "
        f"patients={a['dropout']['n_patients']} "
        f"high_dropout_risk={a['dropout']['n_high_risk']}"
    )
    for site in a["high_risk_sites"]:
        print(f"  site {site['site_id']:>3} risk {site['predicted_prob']:.2f}")

    if context.get("exec_summary"):
        print("\n=== Executive Summary ===\n")
        print(context["exec_summary"])

    print(f"\nReport written to {ARTIFACTS_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
