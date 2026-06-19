"""Synthetic clinical-operations data generator.

Produces realistic ``sites`` / ``patients`` / ``visits`` rows whose observable
signals are driven by hidden latent variables, so the downstream models learn
*real* causal structure rather than noise:

* Each site has a hidden ``site_quality`` ~ N(0, 1). Low quality drives lower
  enrollment attainment, more data queries, more protocol deviations, longer
  visit delays, and higher patient dropout.
* Each patient's ``dropped_out`` label is sampled from a logistic model of
  travel burden, prior no-show behavior, digital engagement, age, and the
  quality of their site.

No PHI — everything is generated. Determinism is controlled by ``seed``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import List, Tuple

import numpy as np

from trialmind.db import Patient, Site, Visit, session_scope

# Region -> candidate countries, for plausible site geography.
_REGIONS = {
    "North America": ["USA", "Canada"],
    "Europe": ["Germany", "France", "UK", "Spain"],
    "Asia Pacific": ["Japan", "Australia", "South Korea"],
    "Latin America": ["Brazil", "Argentina"],
}


@dataclass
class GenerationConfig:
    """Parameters controlling the synthetic generator."""

    n_sites: int = 40
    seed: int = 42
    visits_per_patient: int = 8
    visit_interval_days: int = 28
    study_start: date = date(2026, 1, 1)
    activation_window_days: int = 120
    regions: dict = field(default_factory=lambda: dict(_REGIONS))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _generate_site(rng: np.random.Generator, site_id: int, cfg: GenerationConfig) -> Tuple[Site, float]:
    """Create one site plus its hidden quality latent."""
    region = rng.choice(list(cfg.regions.keys()))
    country = rng.choice(cfg.regions[region])
    site_quality = float(rng.normal(0.0, 1.0))

    capacity = int(rng.integers(40, 200))
    # Staffing roughly tracks capacity; better-quality sites are a bit better staffed.
    staff = max(2, int(round(capacity / 20 + 1.5 * site_quality + rng.normal(0, 1))))
    enrollment_target = int(round(capacity * float(rng.uniform(0.7, 1.0))))
    activation = cfg.study_start + timedelta(
        days=int(rng.integers(0, cfg.activation_window_days))
    )

    site = Site(
        site_id=site_id,
        country=str(country),
        region=str(region),
        site_capacity=capacity,
        staff_count=staff,
        activation_date=activation,
        enrollment_target=enrollment_target,
        latent_quality=round(site_quality, 4),
    )
    return site, site_quality


def _patient_dropout_prob(
    travel_km: float, prior_noshow: int, engagement: float, age: int, site_quality: float
) -> float:
    """Logistic ground-truth dropout probability (the signal the model learns)."""
    logit = (
        -1.6
        + 0.020 * (travel_km - 30.0)        # farther to travel -> more dropout
        + 0.35 * prior_noshow               # past no-shows -> more dropout
        - 2.5 * (engagement - 0.5)          # higher engagement -> less dropout
        + 0.012 * (age - 55)                # older patients slightly more
        - 0.55 * site_quality               # better site -> less dropout
    )
    return _sigmoid(logit)


def _generate_patient(
    rng: np.random.Generator,
    patient_id: int,
    site: Site,
    site_quality: float,
    cfg: GenerationConfig,
) -> Patient:
    """Create one patient with a ground-truth dropout label."""
    age = int(np.clip(rng.normal(55, 12), 18, 90))
    travel_km = float(np.clip(rng.exponential(30.0), 0.5, 300.0))
    prior_noshow = int(rng.poisson(0.6))
    engagement = float(np.clip(rng.beta(2.0, 2.0), 0.01, 0.99))

    prob = _patient_dropout_prob(travel_km, prior_noshow, engagement, age, site_quality)
    dropped_out = bool(rng.random() < prob)

    # Enroll some time after the site activates.
    enrollment_date = site.activation_date + timedelta(days=int(rng.integers(0, 90)))

    return Patient(
        patient_id=patient_id,
        site_id=site.site_id,
        enrollment_date=enrollment_date,
        age=age,
        travel_distance_km=round(travel_km, 1),
        prior_noshow_count=prior_noshow,
        digital_engagement_score=round(engagement, 3),
        dropped_out=dropped_out,
    )


def _generate_visits(
    rng: np.random.Generator,
    patient: Patient,
    site_quality: float,
    cfg: GenerationConfig,
    next_visit_id: int,
) -> List[Visit]:
    """Create the visit schedule for one patient, reflecting site quality and dropout."""
    visits: List[Visit] = []

    # If the patient drops out, they stop attending after a random visit index.
    if patient.dropped_out:
        dropout_index = int(rng.integers(1, cfg.visits_per_patient))
    else:
        dropout_index = cfg.visits_per_patient + 1  # never

    noshow_base = -2.5 + 0.30 * patient.prior_noshow_count - 1.5 * (
        patient.digital_engagement_score - 0.5
    )
    deviation_prob = _sigmoid(-2.8 - 0.6 * site_quality)
    query_prob = _sigmoid(-2.5 - 0.7 * site_quality)

    for i in range(cfg.visits_per_patient):
        scheduled = patient.enrollment_date + timedelta(days=i * cfg.visit_interval_days)

        if i >= dropout_index:
            attended = False
        else:
            attended = rng.random() >= _sigmoid(noshow_base)

        if attended:
            delay = max(0, int(round(rng.normal(2.0 - site_quality, 2.0))))
            actual = scheduled + timedelta(days=delay)
        else:
            actual = None

        visits.append(
            Visit(
                visit_id=next_visit_id + i,
                patient_id=patient.patient_id,
                scheduled_date=scheduled,
                actual_date=actual,
                attended=attended,
                protocol_deviation=bool(rng.random() < deviation_prob),
                query_raised=bool(rng.random() < query_prob),
            )
        )
    return visits


def generate(config: GenerationConfig | None = None, persist: bool = True) -> dict:
    """Generate a full synthetic dataset.

    Returns a summary dict of counts and headline rates. When ``persist`` is True
    the rows are written to the configured database (replacing any existing data).
    """
    cfg = config or GenerationConfig()
    rng = np.random.default_rng(cfg.seed)

    sites: List[Site] = []
    patient_id = 1
    visit_id = 1
    n_patients = 0
    n_dropped = 0

    for site_id in range(1, cfg.n_sites + 1):
        site, quality = _generate_site(rng, site_id, cfg)

        # Enrolled count varies with site quality (under-enrolling sites are a real signal).
        attainment = float(np.clip(rng.normal(0.8 + 0.15 * quality, 0.15), 0.3, 1.2))
        enrolled = max(5, min(site.site_capacity, int(round(site.enrollment_target * attainment))))

        for _ in range(enrolled):
            patient = _generate_patient(rng, patient_id, site, quality, cfg)
            patient.visits = _generate_visits(rng, patient, quality, cfg, visit_id)
            site.patients.append(patient)

            visit_id += cfg.visits_per_patient
            patient_id += 1
            n_patients += 1
            n_dropped += int(patient.dropped_out)

        sites.append(site)

    summary = {
        "sites": len(sites),
        "patients": n_patients,
        "visits": visit_id - 1,
        "dropout_rate": round(n_dropped / n_patients, 4) if n_patients else 0.0,
        "seed": cfg.seed,
    }

    if persist:
        with session_scope() as session:
            # Replace any existing data so re-runs are reproducible.
            session.query(Visit).delete()
            session.query(Patient).delete()
            session.query(Site).delete()
            session.flush()
            session.add_all(sites)

    return summary
