"""Tests for the patient-dropout prediction model.

Trains on a fresh synthetic dataset and checks that the real dropout signal
(travel burden, prior no-shows, digital engagement) is recoverable, and that
the expected evaluation artifacts are written.
"""

from __future__ import annotations

import json

import pytest

from trialmind.db import create_all, drop_all
from trialmind.models.dropout import (
    FEATURE_COLUMNS,
    load_patient_features,
    train_dropout,
)
from trialmind.models.evaluation import artifacts_subdir


@pytest.fixture()
def fresh_db():
    drop_all()
    create_all()
    yield
    drop_all()


def test_load_patient_features_shape(fresh_db):
    from trialmind.synthetic import GenerationConfig, generate

    generate(GenerationConfig(n_sites=20, seed=7), persist=True)
    X, y = load_patient_features()
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(y) > 0
    # Leakage-safe: no visit-derived or outcome-derived columns present.
    assert "dropped_out" not in X.columns
    assert "dropout_rate" not in X.columns
    assert "latent_quality" not in X.columns


def test_train_dropout_metrics_and_artifacts(fresh_db):
    model, metrics = train_dropout(seed=42, n_sites=40)

    # The dropout signal is real and should be clearly recoverable.
    assert metrics["roc_auc"] > 0.65

    # Metrics file exists and round-trips.
    out_dir = artifacts_subdir("dropout")
    metrics_path = out_dir / "metrics.json"
    assert metrics_path.exists()
    saved = json.loads(metrics_path.read_text())
    assert saved["roc_auc"] == pytest.approx(metrics["roc_auc"])

    # All expected plot artifacts were created.
    for name in ("roc.png", "pr.png", "confusion.png", "shap_summary.png"):
        artifact = out_dir / name
        assert artifact.exists(), f"missing artifact: {name}"
        assert artifact.stat().st_size > 0
