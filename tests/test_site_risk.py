"""Tests for the site-risk classification model.

Runs the full pipeline on a small, fast config against whatever database the
``DATABASE_URL`` env var points at, and asserts the model learns real signal and
that all evaluation artifacts are written.
"""

from __future__ import annotations

import json

import pytest

from trialmind.models.evaluation import artifacts_subdir
from trialmind.models.site_risk import (
    FEATURE_COLUMNS,
    feature_importance,
    load_site_features,
    train_site_risk,
)


@pytest.fixture(scope="module")
def trained():
    """Train once on a small config; data is generated inside train_site_risk."""
    model, metrics = train_site_risk(seed=42, n_sites=120)
    return model, metrics


def test_metrics_quality(trained):
    _, metrics = trained
    assert metrics["roc_auc"] > 0.7
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    for key in ("precision", "recall", "f1"):
        assert 0.0 <= metrics[key] <= 1.0


def test_features_are_non_leaky(trained):
    """Latent quality must never appear among the model's features."""
    _, metrics = trained
    assert metrics["features"] == FEATURE_COLUMNS
    assert "latent_quality" not in metrics["features"]


def test_loaded_features_shape(trained):
    df = load_site_features()
    assert len(df) == 120
    assert set(FEATURE_COLUMNS).issubset(df.columns)
    assert "high_risk" in df.columns
    assert "latent_quality" not in df.columns


def test_feature_importance(trained):
    model, _ = trained
    importance = feature_importance(model, FEATURE_COLUMNS)
    assert set(importance) == set(FEATURE_COLUMNS)
    assert all(v >= 0 for v in importance.values())


def test_artifacts_written(trained):
    out_dir = artifacts_subdir("site_risk")
    for name in ("roc.png", "pr.png", "confusion.png", "shap_summary.png"):
        path = out_dir / name
        assert path.exists() and path.stat().st_size > 0, f"missing artifact: {name}"

    metrics_path = out_dir / "metrics.json"
    assert metrics_path.exists()
    saved = json.loads(metrics_path.read_text())
    assert saved["roc_auc"] > 0.7
