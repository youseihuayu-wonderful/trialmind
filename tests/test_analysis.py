"""Tests for the deterministic risk-analysis stage (no API key needed)."""

from __future__ import annotations

from trialmind.analysis import build_analysis
from trialmind.models import site_risk


def test_build_analysis_structure():
    a = build_analysis(top_k=3, n_sites=30)

    assert a["n_sites"] == 30
    assert len(a["high_risk_sites"]) == 3
    assert a["dropout"]["n_patients"] > 0
    assert a["dropout"]["n_high_risk"] >= 0

    # High-risk sites are returned in descending predicted-probability order.
    probs = [s["predicted_prob"] for s in a["high_risk_sites"]]
    assert probs == sorted(probs, reverse=True)


def test_site_records_have_grounded_shap_drivers():
    a = build_analysis(top_k=2, n_sites=30)
    site = a["high_risk_sites"][0]

    # Features present and limited to the allowed observable KPIs.
    assert set(site["features"]) == set(site_risk.FEATURE_COLUMNS)

    # SHAP contributions cover every feature and are sorted by absolute impact.
    contribs = site["shap_contributions"]
    assert {c["feature"] for c in contribs} == set(site_risk.FEATURE_COLUMNS)
    impacts = [abs(c["shap"]) for c in contribs]
    assert impacts == sorted(impacts, reverse=True)


def test_dropout_importance_is_well_formed():
    a = build_analysis(top_k=2, n_sites=30)
    imp = a["dropout"]["global_importance"]
    assert set(imp.keys()) == set(
        __import__("trialmind.models.dropout", fromlist=["FEATURE_COLUMNS"]).FEATURE_COLUMNS
    )
    assert abs(sum(imp.values()) - 1.0) < 0.05  # importances ~ sum to 1
