"""Schemas for the Query Decomposer (planner)."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class SubQuery(BaseModel):
    """A single targeted web-search query produced by the planner."""

    query: str = Field(description="A concise, search-engine-friendly query (5-12 words).")
    rationale: str = Field(description="Which facet of the research question this covers.")


class ResearchPlan(BaseModel):
    """Decomposition of a research question into 2-4 targeted search queries."""

    objective: str = Field(description="One-sentence restatement of what the report must answer.")
    sub_queries: list[SubQuery] = Field(
        min_length=1,
        max_length=6,
        description="2-4 targeted, non-overlapping search queries.",
    )

    @field_validator("sub_queries")
    @classmethod
    def _dedupe_and_cap(cls, v: list[SubQuery]) -> list[SubQuery]:
        """Drop case-insensitive duplicates and keep at most four queries."""
        seen: set[str] = set()
        unique: list[SubQuery] = []
        for sq in v:
            key = " ".join(sq.query.lower().split())
            if key and key not in seen:
                seen.add(key)
                unique.append(sq)
        return unique[:4]
