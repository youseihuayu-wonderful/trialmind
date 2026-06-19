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


def build_default_pipeline(top_k: int = 4) -> Pipeline:
    """Construct the full TrialMind pipeline.

    Deterministic analysis (data -> features -> models -> SHAP) runs first, then
    the three Claude agents reason over its structured output:

      analysis -> explainability -> recommendation -> exec-summary

    The explainability/recommendation/exec-summary stages require an Anthropic
    API key (see trialmind.config). A human-review gate sits conceptually between
    recommendation and any downstream action.
    """
    from trialmind.agents import (
        AnalysisAgent,
        ExecSummaryAgent,
        ExplainabilityAgent,
        RecommendationAgent,
    )

    return Pipeline(
        [
            AnalysisAgent(top_k=top_k),
            ExplainabilityAgent(),
            RecommendationAgent(),
            ExecSummaryAgent(),
        ]
    )
