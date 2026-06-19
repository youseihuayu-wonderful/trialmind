"""Feature engineering: aggregate raw rows into site-level features + risk label.

Reads ``sites`` / ``patients`` / ``visits`` with pandas, computes the operational
KPIs the site-risk model uses, and assigns the ``high_risk`` label.

Label design (no leakage): ``high_risk`` is derived from each site's hidden
``latent_quality`` (the bottom quantile of quality is "high risk"), with a small
amount of label noise to reflect real-world labeling uncertainty. The model is
trained only on the *observable* KPIs below and must recover this label from
them — so a perfect score is impossible and the metrics are meaningful.

Engineered features (per site):
    enrollment_attainment     enrolled / enrollment_target
    query_backlog             total data queries raised
    protocol_deviation_rate   deviations / total visits
    dropout_rate              dropped patients / enrolled
    avg_visit_delay_days      mean (actual - scheduled) over attended visits
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trialmind.db import SiteFeatures, engine, session_scope


def _load_frames():
    sites = pd.read_sql("SELECT * FROM sites", engine)
    patients = pd.read_sql("SELECT * FROM patients", engine)
    visits = pd.read_sql("SELECT * FROM visits", engine)
    return sites, patients, visits


def compute_features(
    label_quantile: float = 0.30,
    label_noise: float = 0.08,
    seed: int = 42,
    persist: bool = True,
) -> pd.DataFrame:
    """Compute the per-site feature table (and optionally persist it).

    Returns a DataFrame indexed by ``site_id`` with the feature columns and the
    ``high_risk`` label.
    """
    sites, patients, visits = _load_frames()
    if sites.empty or patients.empty:
        raise RuntimeError(
            "No data found. Run scripts/generate_data.py before feature engineering."
        )

    # Patient-level aggregation per site.
    pat_agg = (
        patients.groupby("site_id")
        .agg(enrolled=("patient_id", "size"), dropped=("dropped_out", "sum"))
        .reset_index()
    )

    # Attach site_id to each visit (via its patient), then aggregate per site.
    visits = visits.merge(
        patients[["patient_id", "site_id"]], on="patient_id", how="left"
    )
    visits["scheduled_date"] = pd.to_datetime(visits["scheduled_date"])
    visits["actual_date"] = pd.to_datetime(visits["actual_date"])
    attended = visits[visits["attended"] == 1].copy()
    attended["delay_days"] = (
        attended["actual_date"] - attended["scheduled_date"]
    ).dt.days

    vis_agg = (
        visits.groupby("site_id")
        .agg(
            total_visits=("visit_id", "size"),
            query_backlog=("query_raised", "sum"),
            deviations=("protocol_deviation", "sum"),
        )
        .reset_index()
    )
    delay_agg = (
        attended.groupby("site_id")
        .agg(avg_visit_delay_days=("delay_days", "mean"))
        .reset_index()
    )

    df = sites.merge(pat_agg, on="site_id", how="left")
    df = df.merge(vis_agg, on="site_id", how="left")
    df = df.merge(delay_agg, on="site_id", how="left")

    df["enrollment_attainment"] = (df["enrolled"] / df["enrollment_target"]).clip(0, 2)
    df["protocol_deviation_rate"] = df["deviations"] / df["total_visits"]
    df["dropout_rate"] = df["dropped"] / df["enrolled"]
    df["avg_visit_delay_days"] = df["avg_visit_delay_days"].fillna(0.0)
    df["query_backlog"] = df["query_backlog"].astype(int)

    # Non-leaky label: bottom-quantile latent quality is high risk, plus label noise.
    threshold = df["latent_quality"].quantile(label_quantile)
    base_label = df["latent_quality"] < threshold
    rng = np.random.default_rng(seed)
    flip = rng.random(len(df)) < label_noise
    df["high_risk"] = np.where(flip, ~base_label, base_label)

    feature_cols = [
        "site_id",
        "enrollment_attainment",
        "query_backlog",
        "protocol_deviation_rate",
        "dropout_rate",
        "avg_visit_delay_days",
        "high_risk",
    ]
    result = df[feature_cols].set_index("site_id")

    if persist:
        rows = [
            SiteFeatures(
                site_id=int(site_id),
                enrollment_attainment=float(r.enrollment_attainment),
                query_backlog=int(r.query_backlog),
                protocol_deviation_rate=float(r.protocol_deviation_rate),
                dropout_rate=float(r.dropout_rate),
                avg_visit_delay_days=float(r.avg_visit_delay_days),
                high_risk=bool(r.high_risk),
            )
            for site_id, r in result.iterrows()
        ]
        with session_scope() as session:
            session.query(SiteFeatures).delete()
            session.flush()
            session.add_all(rows)

    return result
