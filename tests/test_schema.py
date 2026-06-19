"""Smoke test for the ORM schema against an in-memory SQLite database.

Verifies the relational model wires up correctly and rows round-trip through the
site -> patient -> visit hierarchy plus the derived site_features row.
"""

from __future__ import annotations

from datetime import date

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from trialmind.db.base import Base
from trialmind.db.models import Patient, Site, SiteFeatures, Visit


def _in_memory_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)()


def test_site_patient_visit_roundtrip():
    session = _in_memory_session()

    site = Site(
        site_id=1,
        country="USA",
        region="Northeast",
        site_capacity=120,
        staff_count=6,
        activation_date=date(2026, 1, 15),
        enrollment_target=100,
        latent_quality=0.5,
    )
    patient = Patient(
        patient_id=1,
        site_id=1,
        enrollment_date=date(2026, 2, 1),
        age=54,
        travel_distance_km=42.5,
        prior_noshow_count=1,
        digital_engagement_score=0.73,
        dropped_out=False,
    )
    visit = Visit(
        visit_id=1,
        patient_id=1,
        scheduled_date=date(2026, 2, 8),
        actual_date=date(2026, 2, 9),
        attended=True,
        protocol_deviation=False,
        query_raised=False,
    )
    site.patients.append(patient)
    patient.visits.append(visit)
    session.add(site)
    session.commit()

    loaded = session.get(Site, 1)
    assert loaded is not None
    assert len(loaded.patients) == 1
    assert loaded.patients[0].visits[0].attended is True


def test_site_features_relationship():
    session = _in_memory_session()

    site = Site(
        site_id=2,
        country="Germany",
        region="EU",
        site_capacity=80,
        staff_count=4,
        activation_date=date(2026, 1, 20),
        enrollment_target=90,
        latent_quality=-0.8,
    )
    features = SiteFeatures(
        site_id=2,
        enrollment_attainment=0.62,
        query_backlog=14,
        protocol_deviation_rate=0.08,
        dropout_rate=0.19,
        avg_visit_delay_days=3.4,
        high_risk=True,
    )
    site.features = features
    session.add(site)
    session.commit()

    loaded = session.get(Site, 2)
    assert loaded.features is not None
    assert loaded.features.high_risk is True
    assert loaded.features.enrollment_attainment == 0.62
