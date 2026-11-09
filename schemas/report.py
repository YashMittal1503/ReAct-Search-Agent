"""Schemas for the Grounded Synthesizer and its strictly validated final report.

Two models are deliberately separated:

* :class:`SynthesisDraft` is what the LLM produces. It uses plain ``[n]`` markers and
  never writes URLs, so the model cannot hallucinate links.
* :class:`ResearchReport` is the final, user-facing artefact. Code (not the LLM) builds it
  from the draft and the evidence store, and Pydantic validators enforce that every inline
  citation is a clickable ``[[n]](url)`` hyperlink pointing at a known source.
"""

from __future__ import annotations

import re
from urllib.parse import unquote, urlparse

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

# ``[[3]](https://...)`` — the only citation form allowed in a final report.
LINKED_CITATION_RE = re.compile(r"\[\[(\d+)\]\]\((\S+?)\)")
# ``[3]`` that is *not* already part of ``[[3]]`` — i.e. an unlinked citation.
BARE_CITATION_RE = re.compile(r"(?<!\[)\[(\d+)\](?!\])")


# ---------------------------------------------------------------------------
# LLM-facing draft
# ---------------------------------------------------------------------------
class CitedClaim(BaseModel):
    """Maps a source id used in the draft to the verbatim passage that supports it."""

    source_id: int = Field(ge=1, description="Evidence number as given in the prompt, e.g. 3.")
    supporting_quote: str = Field(
        description="A short VERBATIM sentence or phrase (8-40 words) copied from that source."
    )


class SynthesisDraft(BaseModel):
    """Structured output requested from the synthesizer LLM."""

    answer: str = Field(
        description="Comprehensive Markdown report. Cite evidence inline with bracketed numbers "
        "like [1] or [2][3] immediately after the claim. Do NOT write URLs."
    )
    confidence_score: float = Field(ge=0.0, le=1.0)
    contradictions_noted: list[str] = Field(default_factory=list)
    citations: list[CitedClaim] = Field(
        default_factory=list, description="One entry per source id used in the answer."
    )


# ---------------------------------------------------------------------------
# Final validated report
# ---------------------------------------------------------------------------
class Source(BaseModel):
    """A verified source rendered in the "Verified Sources" panel."""

    id: int = Field(ge=1)
    title: str
    url: str
    deep_link_url: str = Field(
        description="URL with a ``#:~:text=start,end`` fragment that highlights the quote."
    )
    supporting_quote: str
    quote_verified: bool = Field(
        default=True,
        description="True if the quote was found (near-)verbatim in the retrieved source text.",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def domain(self) -> str:
        """Bare domain for the UI badge."""
        host = urlparse(self.url).netloc.lower()
        return host[4:] if host.startswith("www.") else host

    @field_validator("url")
    @classmethod
    def _http_only(cls, v: str) -> str:
        if urlparse(v).scheme not in {"http", "https"}:
            raise ValueError(f"source url must be http(s): {v!r}")
        return v

    @model_validator(mode="after")
    def _deep_link_matches_url(self) -> "Source":
        # Compare percent-decoded bases: the deep link may escape "(" / ")" for Markdown safety.
        if unquote(self.deep_link_url.split("#", 1)[0]) != unquote(self.url.split("#", 1)[0]):
            raise ValueError("deep_link_url must extend the source url")
        return self


class ResearchReport(BaseModel):
    """Final grounded research report with clickable inline citations."""

    answer: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    contradictions_noted: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)

    @model_validator(mode="after")
    def _citations_are_linked_and_resolvable(self) -> "ResearchReport":
        """Enforce the citation contract.

        1. Source ids are unique.
        2. No bare ``[n]`` citations remain; each is a ``[[n]](url)`` hyperlink.
        3. Each linked citation refers to an existing source and points at its url/deep link.
        """
        by_id = {s.id: s for s in self.sources}
        if len(by_id) != len(self.sources):
            raise ValueError("duplicate source ids")

        bare = BARE_CITATION_RE.findall(self.answer)
        if bare:
            raise ValueError(f"unlinked citations found: {sorted(set(bare))}")

        for raw_id, url in LINKED_CITATION_RE.findall(self.answer):
            src = by_id.get(int(raw_id))
            if src is None:
                raise ValueError(f"citation [{raw_id}] has no matching source")
            if url not in {src.url, src.deep_link_url}:
                raise ValueError(f"citation [{raw_id}] url does not match its source")
        return self

    @property
    def cited_ids(self) -> list[int]:
        """Source ids in order of first appearance in the answer."""
        seen: list[int] = []
        for raw_id, _ in LINKED_CITATION_RE.findall(self.answer):
            if int(raw_id) not in seen:
                seen.append(int(raw_id))
        return seen
