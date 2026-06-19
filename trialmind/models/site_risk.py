"""Site-risk classification model.

Predicts whether a research site is ``high_risk`` from five observable
operational KPIs (never from the hidden ``latent_quality``, which would leak the
label). A tree-based classifier is used so SHAP's ``TreeExplainer`` works cleanly.

Because the number of sites is modest, headline metrics are computed from
**stratified 5-fold cross-validated** out-of-fold probabilities rather than a
single train/test split; a final model is then refit on all data for SHAP
explanations and downstream scoring.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from trialmind.db import create_all, drop_all, engine
from trialmind.features import compute_features
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

# The five observable KPI columns the model is allowed to use as features.
FEATURE_COLUMNS: List[str] = [
    "enrollment_attainment",
    "query_backlog",
    "protocol_deviation_rate",
    "dropout_rate",
    "avg_visit_delay_days",
]
LABEL_COLUMN = "high_risk"


def load_site_features() -> pd.DataFrame:
    """Read the persisted ``site_features`` table into a DataFrame.

    Returns a DataFrame indexed by ``site_id`` with the five feature columns plus
    the ``high_risk`` label, read directly from the configured database engine.
    """
    columns = ["site_id"] + FEATURE_COLUMNS + [LABEL_COLUMN]
    df = pd.read_sql(f"SELECT {', '.join(columns)} FROM site_features", engine)
    df = df.set_index("site_id")
    # SQLite stores booleans as 0/1; normalise the label to int.
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    return df


def _build_model(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def feature_importance(model, feature_names: List[str]) -> Dict[str, float]:
    """Return a {feature: importance} dict sorted by descending importance."""
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        raise ValueError("Model does not expose feature_importances_.")
    pairs = sorted(
        zip(feature_names, (float(v) for v in importances)),
        key=lambda kv: kv[1],
        reverse=True,
    )
    return {name: imp for name, imp in pairs}


def _seed_database(seed: int, n_sites: int) -> None:
    """Generate fresh synthetic data and compute/persist site features."""
    drop_all()
    create_all()
    generate(GenerationConfig(n_sites=n_sites, seed=seed), persist=True)
    compute_features(seed=seed, persist=True)


def train_site_risk(
    seed: int = 42,
    n_sites: int = 150,
    n_splits: int = 5,
    threshold: float = 0.5,
    regenerate: bool = True,
) -> Tuple[RandomForestClassifier, Dict]:
    """Run the full site-risk pipeline.

    Steps: (optionally) generate synthetic data and features, cross-validate to
    get honest out-of-fold probabilities, evaluate, refit a final model on all
    data, compute SHAP values, and save all artifacts.

    Returns ``(final_model, metrics)``.
    """
    if regenerate:
        _seed_database(seed=seed, n_sites=n_sites)

    df = load_site_features()
    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN].to_numpy()

    # Out-of-fold probabilities via stratified k-fold cross-validation.
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_proba = cross_val_predict(
        _build_model(seed),
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    oof_pred = (oof_proba >= threshold).astype(int)

    metrics = evaluate_binary(y, oof_proba, threshold=threshold)

    # Final model on all data, used for SHAP and as the returned artifact.
    final_model = _build_model(seed)
    final_model.fit(X, y)
    metrics["feature_importance"] = feature_importance(final_model, FEATURE_COLUMNS)
    metrics["n_sites"] = int(len(df))
    metrics["features"] = list(FEATURE_COLUMNS)
    metrics["model"] = type(final_model).__name__
    metrics["cv_folds"] = int(n_splits)

    out_dir = artifacts_subdir("site_risk")
    save_metrics(metrics, out_dir / "metrics.json")
    plot_roc(y, oof_proba, out_dir / "roc.png", title="Site risk - ROC (out-of-fold)")
    plot_pr(y, oof_proba, out_dir / "pr.png", title="Site risk - PR (out-of-fold)")
    plot_confusion(
        y,
        oof_pred,
        out_dir / "confusion.png",
        title="Site risk - confusion (out-of-fold)",
    )

    _save_shap_summary(final_model, X, out_dir / "shap_summary.png")

    return final_model, metrics


def _save_shap_summary(model, X: pd.DataFrame, path) -> None:
    """Compute positive-class SHAP values for the tree model and save the plot."""
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    positive = _positive_class_shap(shap_values)
    plot_shap_summary(positive, X, path, title="Site risk - SHAP summary")


def _positive_class_shap(shap_values) -> np.ndarray:
    """Extract the positive-class SHAP array across SHAP/sklearn versions.

    ``TreeExplainer.shap_values`` may return a list (one array per class) or a
    single 3-D array ``(n_samples, n_features, n_classes)`` depending on versions.
    """
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1])
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr
