"""LLM-as-judge for answer quality (structured outputs, provider-agnostic via the router).

Two judgements:

* **Nugget coverage**: for each golden nugget, is the claim *supported*, *partially* supported or
  *missing* in the report? Judged semantically; wording doesn't need to match.
* **Citation support**: for each (claim sentence, cited source passage) pair, does the passage
  actually support the claim? This is citation *precision*: do citations point at real support?

Judging is done at low temperature with strict rubrics, and every verdict carries a short
justification, so disagreements can be audited in ``results.jsonl``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from evals.dataset import Nugget

StructuredFn = Callable[..., Awaitable[Any]]  # (schema, messages, fast, *, max_tokens=None) -> schema

NUGGET_SYSTEM = """You are a strict evaluator of research reports.
For each numbered NUGGET (a claim that a good answer should contain), decide whether the REPORT
conveys it:
- "supported": the report clearly states the claim (any wording; equivalent numbers or ranges are fine).
- "partial": the report touches on it but is vague, incomplete, or gives a materially different figure.
- "missing": the report does not convey it, or contradicts it.
Judge only the report text. Do not use outside knowledge to fill gaps. Quote the report briefly
(max 25 words) as evidence when the verdict is supported or partial."""

CITATION_SYSTEM = """You verify citations in a research report.
For each numbered PAIR you get a CLAIM (a sentence from the report) and the SOURCE PASSAGE it cites.
Decide whether the source passage supports the claim:
- "yes": the passage directly supports the claim's key assertion.
- "partial": the passage supports part of the claim, or supports it only loosely.
- "no": the passage does not support the claim, or contradicts it.
Judge only against the given passage. Give a one-sentence reason."""


class NuggetVerdict(BaseModel):
    nugget_id: str
    verdict: Literal["supported", "partial", "missing"]
    evidence: str = Field(default="", description="Short quote from the report (empty if missing).")


class NuggetJudgement(BaseModel):
    verdicts: list[NuggetVerdict]


class CitationVerdict(BaseModel):
    index: int = Field(description="Pair number as given in the prompt.")
    supported: Literal["yes", "partial", "no"]
    reason: str = ""


class CitationJudgement(BaseModel):
    verdicts: list[CitationVerdict]


class CitationPair(BaseModel):
    """A claim sentence and the source passage it cites."""

    source_id: int
    claim: str
    source_title: str
    passage: str


class Judge:
    """Thin wrapper around a structured-output function (real router or a test fake)."""

    def __init__(self, structured: StructuredFn, *, max_tokens: int = 2_048) -> None:
        self.structured = structured
        self.max_tokens = max_tokens

    async def nuggets(self, question: str, answer: str, nuggets: list[Nugget]) -> dict[str, NuggetVerdict]:
        """Verdict per nugget id. Nuggets the judge skips are recorded as 'missing'."""
        if not nuggets:
            return {}
        listing = "\n".join(f"{n.id}: {n.text}" for n in nuggets)
        result: NuggetJudgement = await self.structured(
            NuggetJudgement,
            [SystemMessage(NUGGET_SYSTEM),
             HumanMessage(f"QUESTION:\n{question}\n\nNUGGETS:\n{listing}\n\nREPORT:\n{answer}")],
            False, max_tokens=self.max_tokens,
        )
        by_id = {v.nugget_id: v for v in result.verdicts if v.nugget_id in {n.id for n in nuggets}}
        return {n.id: by_id.get(n.id, NuggetVerdict(nugget_id=n.id, verdict="missing")) for n in nuggets}

    async def citations(self, pairs: list[CitationPair]) -> list[CitationVerdict]:
        """One verdict per pair (in order). Pairs the judge skips count as 'no'."""
        if not pairs:
            return []
        listing = "\n\n".join(
            f"PAIR {i}\nCLAIM: {p.claim}\nSOURCE [{p.source_id}] {p.source_title}:\n{p.passage}"
            for i, p in enumerate(pairs, start=1)
        )
        result: CitationJudgement = await self.structured(
            CitationJudgement, [SystemMessage(CITATION_SYSTEM), HumanMessage(listing)],
            False, max_tokens=self.max_tokens,
        )
        by_index = {v.index: v for v in result.verdicts}
        return [by_index.get(i, CitationVerdict(index=i, supported="no", reason="not judged"))
                for i in range(1, len(pairs) + 1)]
