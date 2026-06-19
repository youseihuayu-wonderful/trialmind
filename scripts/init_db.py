"""Create the TrialMind database schema.

Usage:
    python scripts/init_db.py

Honors DATABASE_URL (defaults to a local SQLite file under data/).
"""

from __future__ import annotations

from trialmind.config import settings
from trialmind.db import create_all


def main() -> None:
    print(f"Initializing database at: {settings.database_url}")
    create_all()
    print("Schema created: sites, patients, visits, site_features")


if __name__ == "__main__":
    main()
