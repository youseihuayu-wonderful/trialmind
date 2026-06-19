"""Train and evaluate the patient-dropout prediction model.

Run with your own database to avoid colliding with other jobs, e.g.::

    DATABASE_URL=sqlite:///data/dropout.db python scripts/train_dropout.py
"""

from __future__ import annotations

from trialmind.models.dropout import train_dropout


def main() -> None:
    model, metrics = train_dropout(seed=42)

    print("Patient-dropout model — held-out test metrics")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC    : {metrics['pr_auc']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  Test n    : {metrics['n']} (positives={metrics['positives']})")
    print()
    print("Feature importance:")
    for name, value in metrics["feature_importance"].items():
        print(f"  {name:26s} {value:.4f}")
    print()
    print(f"Artifacts saved to: {metrics['artifacts_dir']}")


if __name__ == "__main__":
    main()
