"""Central configuration for TrialMind.

Reads from environment variables (optionally loaded from a local ``.env`` file).
Nothing here is hard-coded to a secret; secrets come from the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv not installed yet; env vars still work
    pass

# Repository root (two levels up from this file: trialmind/trialmind/config.py).
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings."""

    database_url: str
    model: str
    anthropic_api_key: str | None
    data_dir: Path

    @property
    def has_llm_credentials(self) -> bool:
        """True when an Anthropic API key is available for the LLM agents."""
        return bool(self.anthropic_api_key)


def load_settings() -> Settings:
    """Build the Settings object from the current environment."""
    return Settings(
        database_url=os.environ.get(
            "DATABASE_URL", f"sqlite:///{DATA_DIR / 'trialmind.db'}"
        ),
        model=os.environ.get("TRIALMIND_MODEL", "claude-opus-4-8"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
        data_dir=DATA_DIR,
    )


settings = load_settings()
