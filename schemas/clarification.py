"""Schemas for the Ambiguity Gate (human-in-the-loop pre-processor)."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ClarityAssessment(BaseModel):
    """LLM verdict on whether a research prompt is specific enough to act on."""

    is_ambiguous: bool = Field(
        description="True if the query is too vague, too short, or missing constraints that "
        "would materially change the research (scope, timeframe, audience, region, use-case)."
    )
    reasoning: str = Field(description="One or two sentences explaining the verdict.")
    missing_constraints: list[str] = Field(
        default_factory=list,
        description="Constraints that are missing, e.g. 'timeframe', 'industry', 'budget'.",
    )
    questions: list[str] = Field(
        default_factory=list,
        description="2-3 brief, targeted follow-up questions. Empty when not ambiguous.",
    )

    @field_validator("questions")
    @classmethod
    def _cap_questions(cls, v: list[str]) -> list[str]:
        """Keep at most three non-empty questions (LLMs sometimes over-produce)."""
        return [q.strip() for q in v if q and q.strip()][:3]


class ClarificationRequest(BaseModel):
    """Payload surfaced to the UI when the graph pauses via ``interrupt()``.

    The UI shows ``questions`` to the user and resumes the graph with their free-text reply.
    """

    original_query: str
    reasoning: str
    questions: list[str] = Field(min_length=1, max_length=3)
    round: int = Field(default=1, ge=1, description="Clarification round (1-based).")
