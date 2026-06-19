"""Base agent interface for the TrialMind pipeline.

Every pipeline stage is an *agent* — a unit with a name and a ``run`` method that
takes a shared context dict and returns an updated context dict. Deterministic
stages (data quality, feature engineering, model scoring) subclass ``Agent``
directly; natural-language stages additionally use ``LLMAgent`` for Claude calls.

The context flows through the pipeline (see ``trialmind.orchestrator``); each agent
reads what it needs and writes its outputs back under its own key.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

Context = Dict[str, Any]


class Agent(ABC):
    """Abstract pipeline stage."""

    #: Human-readable stage name, also used as the context key for this agent's output.
    name: str = "agent"

    @abstractmethod
    def run(self, context: Context) -> Context:
        """Execute the stage and return the updated context.

        Implementations must be pure with respect to the returned context: read
        inputs from ``context``, write outputs back into it, and return it.
        """
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<{type(self).__name__} name={self.name!r}>"


class LLMAgent(Agent):
    """Base for agents that call Claude (explainability / recommendation / summary).

    Holds a lazily-created Anthropic client and the configured model id. Concrete
    subclasses implement ``run`` and use ``self.complete(...)`` for inference.
    """

    def __init__(self, model: str | None = None) -> None:
        from trialmind.config import settings

        self.model = model or settings.model
        self._api_key = settings.anthropic_api_key
        self._client = None  # created on first use

    @property
    def client(self):
        """Lazily construct and cache the Anthropic client."""
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set — required for LLM agents. "
                    "Copy .env.example to .env and set the key."
                )
            import anthropic

            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def complete(self, system: str, user: str, max_tokens: int = 4000) -> str:
        """Single-turn Claude call returning the concatenated text output.

        Uses adaptive thinking (required on Opus 4.8) and high effort.
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            messages=[{"role": "user", "content": user}],
        )
        return "".join(
            block.text for block in response.content if block.type == "text"
        )
