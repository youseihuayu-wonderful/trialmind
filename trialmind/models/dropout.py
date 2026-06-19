"""Patient-dropout prediction model.

Unit of prediction is the *patient*; the target is ``patients.dropped_out``.
To avoid leakage we use only enrollment-time / patient-intrinsic features plus
site context known at enrollment (capacity, staffing). We deliberately exclude
any visit-derived signal and the site's realized ``dropout_rate`` / hidden
``latent_quality``, all of which would leak the outcome.

A tree-based classifier (RandomForest) is used so ``shap.TreeExplainer`` works
cleanly for explanations. Artifacts (ROC, PR, confusion, SHAP summary, metrics)
are written to the shared ``artifacts/dropout`` directory.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from trialmind.db import create_all, drop_all, engine
from trialmind.models.evaluation import (
    artifacts_subdir,
    evaluate_binary,
    plot_confusion,
    plot_pr,
    plot_roc,
    plot_shap_summary,
    save_metrics,
)
from trialmind.synthetic import GenerationConfig, generate

# Enrollment-time / patient-intrinsic features plus site context known at
# enrollment. NOTHING here is derived from visits or from the realized outcome.
FEATURE_COLUMNS: List[str] = [
    "age",
    "travel_distance_km",
    "prior_noshow_count",
    "digital_engagement_score",
    "site_capacity",
    "staff_count",
]


def load_patient_features() -> Tuple[pd.DataFrame, pd.Series]:
    """Load patient features (X) and the dropout label (y) from the database.

    Patients are joined to their site to pull the enrollment-time site context
    (``site_capacity``, ``staff_count``). Returns ``(X, y)`` where ``X`` has the
    columns in :data:`FEATURE_COLUMNS` and ``y`` is the boolean dropout label as
    integers.
    """
    query = """
        SELECT
            p.age                      AS age,
            p.travel_distance_km       AS travel_distance_km,
            p.prior_noshow_count       AS prior_noshow_count,
            p.digital_engagement_score AS digital_engagement_score,
            s.site_capacity            AS site_capacity,
            s.staff_count              AS staff_count,
            p.dropped_out              AS dropped_out
        FROM patients AS p
        JOIN sites AS s ON s.site_id = p.site_id
    """
    df = pd.read_sql(query, engine)
    X = df[FEATURE_COLUMNS].copy()
    y = df["dropped_out"].astype(int)
    return X, y


def feature_importance(model, feature_names: List[str]) -> Dict[str, float]:
    """Return a ``{feature: importance}`` dict sorted by descending importance."""
    importances = model.feature_importances_
    pairs = sorted(
        zip(feature_names, (float(v) for v in importances)),
        key=lambda kv: kv[1],
        reverse=True,
    )
    return dict(pairs)


def train_dropout(seed: int = 42, n_sites: int = 40) -> Tuple[RandomForestClassifier, Dict]:
    """Generate data, train the dropout model, evaluate, and save artifacts.

    Returns the fitted model and the metrics dict.
    """
    # 1. Populate the database with a fresh synthetic dataset.
    drop_all()
    create_all()
    generate(GenerationConfig(n_sites=n_sites, seed=seed), persist=True)

    # 2. Load leakage-safe features and the label.
    X, y = load_patient_features()

    # 3. Stratified held-out split.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 4. Tree-based classifier (SHAP TreeExplainer friendly).
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # 5. Predict probabilities on the test set and score.
    y_score = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_binary(y_test, y_score, threshold=0.5)
    metrics["features"] = list(FEATURE_COLUMNS)
    metrics["feature_importance"] = feature_importance(model, FEATURE_COLUMNS)

    # 6. Artifacts.
    out_dir = artifacts_subdir("dropout")
    y_pred = (y_score >= 0.5).astype(int)

    plot_roc(y_test, y_score, out_dir / "roc.png", title="Dropout ROC curve")
    plot_pr(y_test, y_score, out_dir / "pr.png", title="Dropout Precision-Recall curve")
    plot_confusion(
        y_test,
        y_pred,
        out_dir / "confusion.png",
        labels=("retained", "dropped"),
        title="Dropout confusion matrix",
    )

    # SHAP summary on the test set.
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    # For binary RandomForest, shap_values may be a list [neg, pos] or a 3D array;
    # select the positive-class contributions.
    if isinstance(shap_values, list):
        pos_shap = shap_values[1]
    elif getattr(shap_values, "ndim", 2) == 3:
        pos_shap = shap_values[:, :, 1]
    else:
        pos_shap = shap_values
    plot_shap_summary(
        pos_shap, X_test, out_dir / "shap_summary.png", title="Dropout SHAP summary"
    )

    save_metrics(metrics, out_dir / "metrics.json")
    metrics["artifacts_dir"] = str(out_dir)

    return model, metrics
