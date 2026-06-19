"""Tests for the dashboard's pure (non-Streamlit) helpers."""

from __future__ import annotations

from trialmind.dashboard import importance_frame, shap_frame, sites_dataframe

_ANALYSIS = {
    "n_sites": 40,
    "n_high_risk_sites": 2,
    "high_risk_sites": [
        {
            "site_id": 30,
            "country": "USA",
            "region": "North America",
            "predicted_prob": 0.95,
            "features": {"enrollment_attainment": 0.6},
            "shap_contributions": [
                {"feature": "enrollment_attainment", "value": 0.6, "shap": 0.22},
                {"feature": "query_backlog", "value": 40, "shap": 0.10},
            ],
        },
        {
            "site_id": 9,
            "country": "Germany",
            "region": "Europe",
            "predicted_prob": 0.90,
            "features": {"enrollment_attainment": 0.7},
            "shap_contributions": [
                {"feature": "dropout_rate", "value": 0.3, "shap": 0.18},
            ],
        },
    ],
    "dropout": {
        "global_importance": {"travel_distance_km": 0.3, "age": 0.15},
    },
}


def test_sites_dataframe():
    df = sites_dataframe(_ANALYSIS)
    assert list(df["site_id"]) == [30, 9]
    assert df.loc[0, "lead_driver"] == "enrollment_attainment"


def test_shap_frame():
    df = shap_frame(_ANALYSIS["high_risk_sites"][0])
    assert "shap" in df.columns
    assert df.loc["enrollment_attainment", "shap"] == 0.22


def test_importance_frame_sorted():
    df = importance_frame(_ANALYSIS)
    # Sorted by importance descending.
    assert list(df.index) == ["travel_distance_km", "age"]
