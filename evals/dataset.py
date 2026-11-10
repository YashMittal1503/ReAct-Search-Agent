"""Golden dataset schema and loader.

Each line of ``evals/golden/questions.jsonl`` is one :class:`GoldenQuestion`. The core of each item is
its list of **nuggets**: atomic facts, caveats or known contradictions that a good answer must
contain (the "nugget evaluation" approach used in TREC question answering). Nuggets are phrased as
claims a judge can check against the report, not as exact wording.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

DEFAULT_DATASET = Path(__file__).parent / "golden" / "questions.jsonl"

NuggetKind = Literal["fact", "caveat", "contradiction"]


class Nugget(BaseModel):
    """One unit of required information."""

    id: str
    text: str = Field(description="The claim a good answer must contain (judged semantically).")
    kind: NuggetKind = "fact"
    vital: bool = Field(default=True, description="Vital nuggets weigh 1.0; non-vital (nice to have) 0.5.")


class GoldenQuestion(BaseModel):
    """One evaluation item."""

    id: str
    question: str
    domain: str
    nuggets: list[Nugget] = Field(default_factory=list)
    expected_clarification: bool = Field(
        default=False, description="True if the Ambiguity Gate SHOULD ask a clarifying question.")
    clarification_reply: str = Field(
        default="Proceed with sensible, clearly stated assumptions.",
        description="Simulated user reply used if the agent asks for clarification.")
    time_sensitive: bool = Field(default=False, description="Facts may drift; re-review periodically.")
    as_of: str = Field(description="Date (YYYY-MM) the nuggets were written for.")
    reviewed: bool = Field(default=False, description="Set to true once a human has verified the nuggets.")
    notes: str = ""

    @model_validator(mode="after")
    def _consistency(self) -> "GoldenQuestion":
        if not self.expected_clarification and not self.nuggets:
            raise ValueError(f"{self.id}: answerable questions need at least one nugget")
        ids = [n.id for n in self.nuggets]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{self.id}: duplicate nugget ids")
        if any(not n.id.startswith(f"{self.id}-") for n in self.nuggets):
            raise ValueError(f"{self.id}: nugget ids must be prefixed with the question id")
        return self


def load_dataset(path: str | Path = DEFAULT_DATASET, *, ids: list[str] | None = None,
                 domains: list[str] | None = None, limit: int | None = None) -> list[GoldenQuestion]:
    """Load and validate the golden set, optionally filtered."""
    items = [GoldenQuestion.model_validate_json(line)
             for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    dupes = [k for k, v in Counter(q.id for q in items).items() if v > 1]
    if dupes:
        raise ValueError(f"duplicate question ids: {dupes}")
    if ids:
        items = [q for q in items if q.id in set(ids)]
    if domains:
        items = [q for q in items if q.domain in set(domains)]
    return items[:limit] if limit else items


def dataset_stats(items: list[GoldenQuestion]) -> dict[str, object]:
    """Quick overview used by the CLI and the README."""
    return {
        "questions": len(items),
        "answerable": sum(not q.expected_clarification for q in items),
        "ambiguous": sum(q.expected_clarification for q in items),
        "nuggets": sum(len(q.nuggets) for q in items),
        "by_domain": dict(Counter(q.domain for q in items)),
        "time_sensitive": sum(q.time_sensitive for q in items),
        "reviewed": sum(q.reviewed for q in items),
    }


def dump_jsonl(items: list[GoldenQuestion], path: str | Path) -> None:
    Path(path).write_text("".join(json.dumps(q.model_dump(), ensure_ascii=False) + "\n" for q in items),
                          encoding="utf-8")
