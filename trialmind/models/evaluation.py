"""Shared evaluation utilities for the predictive models.

Both the site-risk and patient-dropout models report the same metrics
(ROC-AUC, PR-AUC, confusion matrix, precision/recall/F1) and save the same
plot family (ROC curve, PR curve, confusion matrix, SHAP summary) so results
are comparable and dashboard-ready.

All plotting uses the non-interactive Agg backend (no display required).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from trialmind.config import settings  # noqa: E402

ARTIFACTS_DIR = settings.data_dir / "artifacts"


def artifacts_subdir(name: str) -> Path:
    """Return (creating if needed) an artifacts subdirectory for a model."""
    path = ARTIFACTS_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def evaluate_binary(y_true, y_score, threshold: float = 0.5) -> dict:
    """Compute the standard binary-classification metric bundle.

    ``y_score`` is the predicted probability of the positive class.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n": int(y_true.shape[0]),
        "positives": int(y_true.sum()),
    }


def save_metrics(metrics: dict, path: Path) -> None:
    """Write a metrics dict to JSON."""
    path.write_text(json.dumps(metrics, indent=2))


def plot_roc(y_true, y_score, path: Path, title: str = "ROC curve") -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_pr(y_true, y_score, path: Path, title: str = "Precision-Recall curve") -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, label=f"PR-AUC = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(
    y_true, y_pred, path: Path, labels: Sequence[str] = ("low", "high"),
    title: str = "Confusion matrix",
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_shap_summary(shap_values, features, path: Path, title: Optional[str] = None) -> None:
    """Save a SHAP summary (beeswarm) plot.

    ``shap_values`` is the positive-class SHAP array; ``features`` is the matching
    feature DataFrame (columns become the y-axis labels).
    """
    import shap  # imported lazily to keep import cost out of non-SHAP paths

    plt.figure()
    shap.summary_plot(shap_values, features, show=False)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
