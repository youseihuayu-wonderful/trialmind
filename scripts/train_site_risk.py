"""Train and evaluate the site-risk classification model.

Generates synthetic data, engineers features, cross-validates, refits a final
model, and writes evaluation artifacts. Run with::

    DATABASE_URL=sqlite:///data/site_risk.db .venv/bin/python scripts/train_site_risk.py
"""

from __future__ import annotations

from trialmind.models.evaluation import artifacts_subdir
from trialmind.models.site_risk import train_site_risk


def main() -> None:
    model, metrics = train_site_risk(seed=42)

    out_dir = artifacts_subdir("site_risk")
    print("Site-risk model trained.")
    print(f"  sites:      {metrics['n']}  (positives: {metrics['positives']})")
    print(f"  model:      {metrics['model']}  ({metrics['cv_folds']}-fold CV)")
    print(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:     {metrics['pr_auc']:.4f}")
    print(f"  precision:  {metrics['precision']:.4f}")
    print(f"  recall:     {metrics['recall']:.4f}")
    print(f"  F1:         {metrics['f1']:.4f}")
    print("  feature importance:")
    for name, imp in metrics["feature_importance"].items():
        print(f"    {name:26s} {imp:.4f}")
    print(f"  artifacts saved to: {out_dir}")
    print("    roc.png, pr.png, confusion.png, shap_summary.png, metrics.json")


if __name__ == "__main__":
    main()
