"""Schemas for retrieved evidence and the reflection (evaluator) step."""

from __future__ import annotations

from urllib.parse import urlparse

from pydantic import BaseModel, Field, computed_field, field_validator


class SearchResult(BaseModel):
    """Lightweight discovery-tier result returned by ``search_web``."""

    title: str
    url: str = Field(description="Normalized URL (tracking params and fragments stripped).")
    snippet: str
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Search-engine relevance score.")
    query: str = Field(default="", description="Sub-query that surfaced this result.")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def domain(self) -> str:
        """Bare domain (``www.`` stripped) for badges and diversity checks."""
        host = urlparse(self.url).netloc.lower()
        return host[4:] if host.startswith("www.") else host


class ScrapedPage(BaseModel):
    """Full-page extraction result from ``scrape_page``."""

    url: str
    title: str = ""
    markdown: str = ""
    ok: bool = True
    error: str | None = None
    chars: int = 0


class Evidence(BaseModel):
    """A single piece of evidence: a search hit optionally enriched with full page text."""

    url: str
    title: str
    snippet: str
    score: float = 0.0
    query: str = ""
    content: str | None = Field(default=None, description="Scraped Markdown, if the page was read.")

    @property
    def text(self) -> str:
        """Best available text for grounding: scraped content, else the snippet."""
        return self.content or self.snippet

    @property
    def domain(self) -> str:
        host = urlparse(self.url).netloc.lower()
        return host[4:] if host.startswith("www.") else host


class EvidenceEvaluation(BaseModel):
    """Reflection verdict: is the gathered evidence sufficient to answer the question?"""

    is_sufficient: bool = Field(description="True if evidence answers all facets of the question.")
    coverage_summary: str = Field(description="What is already well covered, in 1-3 sentences.")
    gaps: list[str] = Field(default_factory=list, description="Specific missing information.")
    conflicting_claims: list[str] = Field(
        default_factory=list, description="Claims where sources disagree, with both positions."
    )
    follow_up_queries: list[str] = Field(
        default_factory=list, description="New search queries that would close the gaps."
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class KeyPoint(BaseModel):
    """One claim from a source, backed by a verbatim quote."""

    claim: str = Field(description="The claim in your own words (one sentence).")
    quote: str = Field(description="A VERBATIM passage (8-40 words) copied exactly from the source that supports it.")


class SourceNotes(BaseModel):
    """Step 1 of two-step synthesis: the parts of one source that matter for the question."""

    relevant: bool = Field(description="False if the source contains nothing useful for the question.")
    key_points: list[KeyPoint] = Field(default_factory=list, description="2-5 most important points.")

    @field_validator("key_points")
    @classmethod
    def _cap(cls, v: list[KeyPoint]) -> list[KeyPoint]:
        return [kp for kp in v if kp.quote.strip()][:5]
