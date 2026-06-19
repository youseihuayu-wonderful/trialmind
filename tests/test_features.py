"""Tests for the feature-engineering step.

Runs against a fresh in-memory-style database populated by the generator, and
checks that the engineered features separate high-risk from low-risk sites in the
expected directions (so the label is recoverable from the observable KPIs).
"""

from __future__ import annotations

import pytest

from trialmind.db import create_all, drop_all
from trialmind.features import compute_features
from trialmind.synthetic import GenerationConfig, generate


@pytest.fixture()
def populated_db():
    drop_all()
    create_all()
    generate(GenerationConfig(n_sites=60, seed=11), persist=True)
    yield
    drop_all()


def test_features_table_shape(populated_db):
    df = compute_features(seed=11)
    assert len(df) == 60
    expected = {
        "enrollment_attainment",
        "query_backlog",
        "protocol_deviation_rate",
        "dropout_rate",
        "avg_visit_delay_days",
        "high_risk",
    }
    assert expected.issubset(set(df.columns))


def test_label_balance_is_reasonable(populated_db):
    df = compute_features(seed=11)
    rate = df["high_risk"].mean()
    assert 0.15 < rate < 0.5  # near the configured 30% quantile, with noise


def test_high_risk_sites_look_worse(populated_db):
    df = compute_features(seed=11)
    high = df[df["high_risk"]]
    low = df[~df["high_risk"]]
    # High-risk sites should, on average, enroll less and drop out more.
    assert high["enrollment_attainment"].mean() < low["enrollment_attainment"].mean()
    assert high["dropout_rate"].mean() > low["dropout_rate"].mean()


def test_no_missing_values(populated_db):
    df = compute_features(seed=11)
    assert not df.isnull().any().any()
