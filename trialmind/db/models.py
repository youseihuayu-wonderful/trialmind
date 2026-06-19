"""SQLAlchemy ORM schema for TrialMind.

Relational model of clinical-trial operations data:

    sites  1───*  patients  1───*  visits
      │
      1───1  site_features   (derived aggregate features + risk label)

All columns map to real, computable quantities — the feature-engineering step
populates ``site_features`` from the raw ``sites`` / ``patients`` / ``visits`` rows.
"""

from __future__ import annotations

from datetime import date
from typing import List, Optional

from sqlalchemy import Boolean, Date, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from trialmind.db.base import Base


class Site(Base):
    """A research site participating in the trial."""

    __tablename__ = "sites"

    site_id: Mapped[int] = mapped_column(primary_key=True)
    country: Mapped[str] = mapped_column(String(64))
    region: Mapped[str] = mapped_column(String(64))
    site_capacity: Mapped[int] = mapped_column(Integer)          # max concurrent patients
    staff_count: Mapped[int] = mapped_column(Integer)
    activation_date: Mapped[date] = mapped_column(Date)
    enrollment_target: Mapped[int] = mapped_column(Integer)

    patients: Mapped[List["Patient"]] = relationship(
        back_populates="site", cascade="all, delete-orphan"
    )
    features: Mapped[Optional["SiteFeatures"]] = relationship(
        back_populates="site", uselist=False, cascade="all, delete-orphan"
    )


class Patient(Base):
    """An enrolled patient, attached to one site."""

    __tablename__ = "patients"

    patient_id: Mapped[int] = mapped_column(primary_key=True)
    site_id: Mapped[int] = mapped_column(ForeignKey("sites.site_id"), index=True)
    enrollment_date: Mapped[date] = mapped_column(Date)
    age: Mapped[int] = mapped_column(Integer)
    travel_distance_km: Mapped[float] = mapped_column(Float)     # travel burden
    prior_noshow_count: Mapped[int] = mapped_column(Integer)     # prior no-show behavior
    digital_engagement_score: Mapped[float] = mapped_column(Float)  # 0–1
    dropped_out: Mapped[bool] = mapped_column(Boolean)          # label for dropout model

    site: Mapped["Site"] = relationship(back_populates="patients")
    visits: Mapped[List["Visit"]] = relationship(
        back_populates="patient", cascade="all, delete-orphan"
    )


class Visit(Base):
    """A scheduled visit / event for a patient."""

    __tablename__ = "visits"

    visit_id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patients.patient_id"), index=True
    )
    scheduled_date: Mapped[date] = mapped_column(Date)
    actual_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    attended: Mapped[bool] = mapped_column(Boolean)
    protocol_deviation: Mapped[bool] = mapped_column(Boolean)
    query_raised: Mapped[bool] = mapped_column(Boolean)         # data-management query

    patient: Mapped["Patient"] = relationship(back_populates="visits")


class SiteFeatures(Base):
    """Site-level aggregated features + risk label, derived by feature engineering."""

    __tablename__ = "site_features"

    site_id: Mapped[int] = mapped_column(
        ForeignKey("sites.site_id"), primary_key=True
    )
    enrollment_attainment: Mapped[float] = mapped_column(Float)    # enrolled / target
    query_backlog: Mapped[int] = mapped_column(Integer)            # open queries
    protocol_deviation_rate: Mapped[float] = mapped_column(Float)
    dropout_rate: Mapped[float] = mapped_column(Float)
    avg_visit_delay_days: Mapped[float] = mapped_column(Float)
    high_risk: Mapped[bool] = mapped_column(Boolean)              # label for site model

    site: Mapped["Site"] = relationship(back_populates="features")
