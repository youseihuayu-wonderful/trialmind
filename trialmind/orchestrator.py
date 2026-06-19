"""Pipeline orchestrator.

Runs a sequence of agents, threading a shared context dict through each. The
orchestration itself is plain, deterministic Python (easy to test and reason
about); individual agents may call Claude internally.

Agents are registered in pipeline order. As stages are implemented they are added
here. A human-review gate sits between recommendation generation and any action.
"""

from __future__ import annotations

from typing import List

from trialmind.agents.base import Agent, Context


class Pipeline:
    """An ordered collection of agents executed sequentially."""

    def __init__(self, agents: List[Agent] | None = None) -> None:
        self.agents: List[Agent] = list(agents or [])

    def add(self, agent: Agent) -> "Pipeline":
        """Append an agent to the pipeline (fluent)."""
        self.agents.append(agent)
        return self

    def run(self, context: Context | None = None) -> Context:
        """Execute every agent in order, threading the context through."""
        ctx: Context = dict(context or {})
        for agent in self.agents:
            ctx = agent.run(ctx)
        return ctx

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.agents)


def build_default_pipeline() -> Pipeline:
    """Construct the full TrialMind pipeline.

    Stages are added here as they are implemented:
      data-quality -> feature-engineering -> site-risk -> dropout
      -> explainability -> recommendation -> exec-summary
    """
    pipeline = Pipeline()
    # Stages are wired in as each agent is implemented (see project roadmap).
    return pipeline
