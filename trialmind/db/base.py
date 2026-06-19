"""Database engine and session setup.

Works against SQLite (default, zero-setup) or PostgreSQL (set ``DATABASE_URL``).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from trialmind.config import settings


class Base(DeclarativeBase):
    """Declarative base class for all ORM models."""


def _make_engine(database_url: str):
    # check_same_thread is a SQLite-only connection arg; skip it for Postgres.
    connect_args = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(database_url, connect_args=connect_args, future=True)


engine = _make_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional scope around a series of operations."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_all() -> None:
    """Create every table defined on ``Base``'s metadata."""
    # Import models so they register with Base.metadata before create_all runs.
    from trialmind.db import models  # noqa: F401

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)


def drop_all() -> None:
    """Drop every table. Destructive — used by tests and re-seeding."""
    from trialmind.db import models  # noqa: F401

    Base.metadata.drop_all(engine)
