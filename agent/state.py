"""LangGraph state definition.

All values are plain JSON-compatible data (Pydantic models are stored via ``model_dump()``) so the
whole state — including the URL registry — can be checkpointed and resumed after a
human-in-the-loop interrupt.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class ResearchState(TypedDict, total=False):
    """Shared state flowing through ``assess_clarity -> ... -> synthesize``."""

    # --- input / ambiguity gate ---------------------------------------------------------------
    query: str                                  # original user prompt
    clarified_query: str                        # prompt enriched with the user's clarifications
    clarification_rounds: int                   # how many times we asked the user
    pending_clarification: dict[str, Any] | None  # ClarificationRequest awaiting an answer
    clarification_log: Annotated[list[dict[str, Any]], operator.add]  # Q&A history

    # --- ReAct loop ---------------------------------------------------------------------------
    iteration: int                              # completed plan->research->evaluate cycles
    plan: dict[str, Any]                        # current ResearchPlan
    registry: dict[str, Any]                    # URLRegistry.snapshot()
    research_notes: Annotated[list[str], operator.add]   # researcher summaries per iteration
    skipped_scrapes: Annotated[list[dict[str, str]], operator.add]
    evaluation: dict[str, Any] | None           # latest EvidenceEvaluation
    evaluation_history: Annotated[list[dict[str, Any]], operator.add]

    # --- output -------------------------------------------------------------------------------
    report: dict[str, Any] | None               # ResearchReport


def initial_state(query: str) -> ResearchState:
    """Fresh state for a new research request."""
    return ResearchState(
        query=query,
        clarified_query=query,
        clarification_rounds=0,
        pending_clarification=None,
        iteration=0,
        registry={},
        evaluation=None,
        report=None,
    )
