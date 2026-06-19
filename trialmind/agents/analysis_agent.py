"""Deterministic analysis stage, wrapped as a pipeline agent.

Runs ``trialmind.analysis.build_analysis`` and writes the structured result into
the pipeline context under ``"analysis"`` for the downstream LLM agents.
"""

from __future__ import annotations

from trialmind.agents.base import Agent, Context
from trialmind.analysis import build_analysis


class AnalysisAgent(Agent):
    name = "analysis"

    def __init__(self, top_k: int = 4) -> None:
        self.top_k = top_k

    def run(self, context: Context) -> Context:
        context["analysis"] = build_analysis(top_k=self.top_k)
        return context
