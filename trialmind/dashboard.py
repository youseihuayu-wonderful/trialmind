"""TrialMind Streamlit dashboard.

A visual front end over the analysis pipeline: headline risk posture, model
performance, the ranked high-risk sites with their SHAP drivers, and the
patient-dropout outlook. Runs without an Anthropic API key (the deterministic
analysis only); if a key is configured, the LLM explanation/recommendation/
summary agents can be run from the UI.

Run with:
    streamlit run trialmind/dashboard.py

Pure helpers (no Streamlit calls) live at module top so they are unit-testable;
the Streamlit rendering is in main(), guarded by __name__ == "__main__" (which is
how Streamlit executes the script).
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

# ---------------------------------------------------------------------------
# Pure, testable helpers
# ---------------------------------------------------------------------------


def sites_dataframe(analysis: Dict) -> pd.DataFrame:
    """Ranked high-risk sites as a tidy table."""
    rows = []
    for s in analysis["high_risk_sites"]:
        top = s["shap_contributions"][0]
        rows.append(
            {
                "site_id": s["site_id"],
                "country": s["country"],
                "region": s["region"],
                "predicted_risk": s["predicted_prob"],
                "lead_driver": top["feature"],
            }
        )
    return pd.DataFrame(rows)


def shap_frame(site: Dict) -> pd.DataFrame:
    """Per-feature SHAP contributions for one site, indexed by feature."""
    df = pd.DataFrame(site["shap_contributions"])
    return df.set_index("feature")[["shap"]]


def importance_frame(analysis: Dict) -> pd.DataFrame:
    """Dropout model global feature importance, indexed by feature."""
    imp = analysis["dropout"]["global_importance"]
    return (
        pd.DataFrame({"feature": list(imp.keys()), "importance": list(imp.values())})
        .set_index("feature")
        .sort_values("importance", ascending=False)
    )


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------


def main() -> None:
    import streamlit as st

    from trialmind.config import settings

    st.set_page_config(page_title="TrialMind", layout="wide")
    st.title("TrialMind — Clinical Trial Risk Dashboard")
    st.caption(
        "Agentic AI pipeline for clinical trial risk modeling. Synthetic "
        "clinical-operations data — no PHI."
    )

    @st.cache_data(show_spinner="Running risk analysis (trains models on first load)…")
    def _analysis(top_k: int) -> Dict:
        from trialmind.analysis import build_analysis

        return build_analysis(top_k=top_k)

    top_k = st.sidebar.slider("High-risk sites to surface", 3, 10, 5)
    analysis = _analysis(top_k)

    sm = analysis["site_metrics"]
    dm = analysis["dropout_metrics"]
    drop = analysis["dropout"]

    # Headline metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sites", analysis["n_sites"])
    c2.metric("High-risk sites", analysis["n_high_risk_sites"])
    c3.metric("Patients", drop["n_patients"])
    c4.metric("Flagged for dropout", drop["n_high_risk"])

    c5, c6 = st.columns(2)
    c5.metric("Site-risk model ROC-AUC", f"{sm.get('roc_auc', float('nan')):.3f}")
    c6.metric("Dropout model ROC-AUC", f"{dm.get('roc_auc', float('nan')):.3f}")

    st.divider()

    # High-risk sites
    left, right = st.columns([1, 1])
    with left:
        st.subheader("High-risk sites")
        st.dataframe(sites_dataframe(analysis), use_container_width=True)

    with right:
        st.subheader("Why this site is high-risk (SHAP)")
        site_ids = [s["site_id"] for s in analysis["high_risk_sites"]]
        chosen = st.selectbox("Site", site_ids)
        site = next(s for s in analysis["high_risk_sites"] if s["site_id"] == chosen)
        st.bar_chart(shap_frame(site))
        st.caption(
            "SHAP value = how much each KPI pushed this site's risk up (positive) "
            "or down (negative)."
        )

    st.divider()

    st.subheader("Patient dropout — global drivers")
    st.bar_chart(importance_frame(analysis))

    st.divider()

    # LLM agents (optional, needs a key)
    st.subheader("Natural-language analysis (Claude Opus 4.8)")
    if settings.has_llm_credentials:
        if st.button("Generate explanations, recommendations & executive summary"):
            from trialmind.orchestrator import build_default_pipeline

            with st.spinner("Calling Claude…"):
                ctx = build_default_pipeline(top_k=top_k).run()
            st.markdown("#### Executive summary")
            st.write(ctx.get("exec_summary", ""))
            for s in ctx["analysis"]["high_risk_sites"]:
                sid = s["site_id"]
                with st.expander(f"Site {sid} — risk {s['predicted_prob']:.2f}"):
                    st.markdown("**Why**")
                    st.write(ctx.get("explanations", {}).get(sid, ""))
                    st.markdown("**Recommended actions**")
                    st.write(ctx.get("recommendations", {}).get(sid, ""))
    else:
        st.info(
            "Set ANTHROPIC_API_KEY in .env to enable the LLM agents "
            "(explainability, recommendation, executive summary). The analysis "
            "above runs without a key."
        )


if __name__ == "__main__":
    main()
