"""Generate synthetic clinical-operations data into the database.

Usage:
    python scripts/generate_data.py [--sites N] [--seed S]

Creates the schema if needed, then populates sites / patients / visits.
"""

from __future__ import annotations

import argparse

from trialmind.db import create_all
from trialmind.synthetic import GenerationConfig, generate


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic trial data.")
    parser.add_argument("--sites", type=int, default=40, help="number of sites")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    create_all()
    summary = generate(GenerationConfig(n_sites=args.sites, seed=args.seed))

    print("Synthetic data generated:")
    print(f"  sites        : {summary['sites']}")
    print(f"  patients     : {summary['patients']}")
    print(f"  visits       : {summary['visits']}")
    print(f"  dropout rate : {summary['dropout_rate']:.1%}")
    print(f"  seed         : {summary['seed']}")


if __name__ == "__main__":
    main()
