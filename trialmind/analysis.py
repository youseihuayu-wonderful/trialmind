"""Deterministic risk analysis that feeds the LLM agents.

Produces the structured inputs the natural-language agents reason over:

* the top high-risk sites, each with its KPI values and the SHAP drivers behind
  its predicted risk (local explanation), and
* a patient-dropout summary (how many patients are flagged, the global drivers,
  and a few example high-risk profiles).

Headline model metrics come from the proper validated training runs
(``train_site_risk`` / ``train_dropout``, cached as ``metrics.json``); the
per-site SHAP drivers are computed in-sample on the current demo database, which
is what the explanations describe.
"""

from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier

from trialmind.db import create_all, drop_all, engine
from trialmind.features import compute_features
from trialmind.models import dropout, site_risk
from trialmind.models.evaluation import artifacts_subdir
from trialmind.synthetic import GenerationConfig, generate


def _load_or_compute_metrics(name: str, train_fn) -> Dict:
    """Return cached metrics.json for a model, training once if it is missing."""
    path = artifacts_subdir(name) / "metrics.json"
    if path.exists():
        return json.loads(path.read_text())
    _model, metrics = train_fn()
    return metrics


def _ensure_demo_data(seed: int, n_sites: int) -> None:
    """Rebuild a clean demo database for in-sample scoring/explanations."""
    drop_all()
    create_all()
    generate(GenerationConfig(n_sites=n_sites, seed=seed), persist=True)
    compute_features(seed=seed)


def _site_scoring(top_k: int) -> Dict:
    """Fit the site model on the demo data and extract top high-risk sites + SHAP."""
    df = site_risk.load_site_features()
    feats = site_risk.FEATURE_COLUMNS
    X = df[feats]
    y = df[site_risk.LABEL_COLUMN]

    model = site_risk._build_model(seed=42).fit(X, y)
    probs = model.predict_proba(X)[:, 1]

    explainer = shap.TreeExplainer(model)
    shap_vals = site_risk._positive_class_shap(explainer.shap_values(X))

    meta = pd.read_sql("SELECT site_id, country, region FROM sites", engine).set_index(
        "site_id"
    )

    order = np.argsort(probs)[::-1][:top_k]
    records: List[Dict] = []
    for i in order:
        sid = int(X.index[i])
        contribs = sorted(
            (
                {
                    "feature": f,
                    "value": round(float(X.iloc[i][f]), 4),
                    "shap": round(float(shap_vals[i][j]), 4),
                }
                for j, f in enumerate(feats)
            ),
            key=lambda d: abs(d["shap"]),
            reverse=True,
        )
        records.append(
            {
                "site_id": sid,
                "predicted_prob": round(float(probs[i]), 4),
                "country": str(meta.loc[sid, "country"]),
                "region": str(meta.loc[sid, "region"]),
                "features": {f: round(float(X.iloc[i][f]), 4) for f in feats},
                "shap_contributions": contribs,
            }
        )

    return {
        "sites": records,
        "n_sites": int(len(df)),
        "n_high_risk": int(y.sum()),
    }


def _dropout_scoring(threshold: float, n_examples: int) -> Dict:
    """Fit the dropout model on the demo data and summarise flagged patients."""
    X, y = dropout.load_patient_features()
    model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
    ).fit(X, y)
    probs = model.predict_proba(X)[:, 1]

    importance = dropout.feature_importance(model, list(X.columns))
    order = np.argsort(probs)[::-1][:n_examples]
    examples = [
        {
            "predicted_prob": round(float(probs[i]), 4),
            "features": {c: round(float(X.iloc[i][c]), 2) for c in X.columns},
        }
        for i in order
    ]
    return {
        "n_patients": int(len(X)),
        "n_high_risk": int((probs >= threshold).sum()),
        "threshold": threshold,
        "global_importance": {k: round(v, 4) for k, v in importance.items()},
        "examples": examples,
    }


def build_analysis(
    top_k: int = 4,
    dropout_threshold: float = 0.5,
    n_examples: int = 3,
    seed: int = 42,
    n_sites: int = 40,
) -> Dict:
    """Run the full deterministic analysis and return the structured context."""
    site_metrics = _load_or_compute_metrics("site_risk", site_risk.train_site_risk)
    dropout_metrics = _load_or_compute_metrics("dropout", dropout.train_dropout)

    _ensure_demo_data(seed=seed, n_sites=n_sites)
    site = _site_scoring(top_k)
    drop = _dropout_scoring(dropout_threshold, n_examples)

    return {
        "site_metrics": site_metrics,
        "dropout_metrics": dropout_metrics,
        "n_sites": site["n_sites"],
        "n_high_risk_sites": site["n_high_risk"],
        "high_risk_sites": site["sites"],
        "dropout": drop,
    }
