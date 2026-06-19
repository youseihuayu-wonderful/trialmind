"""Compute site-level features and labels into the database.

Usage:
    python scripts/run_features.py

Requires that synthetic data has already been generated.
"""

from __future__ import annotations

from trialmind.features import compute_features


def main() -> None:
    df = compute_features()
    n_high = int(df["high_risk"].sum())
    print("Feature engineering complete:")
    print(f"  sites featured : {len(df)}")
    print(f"  high-risk sites: {n_high} ({n_high / len(df):.1%})")
    print()
    print("Feature summary (mean by risk class):")
    summary = df.groupby("high_risk")[
        [
            "enrollment_attainment",
            "query_backlog",
            "protocol_deviation_rate",
            "dropout_rate",
            "avg_visit_delay_days",
        ]
    ].mean()
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
