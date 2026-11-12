"""Citation guard in synthesis (#24): uncited drafts get one cited revision; never made worse."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.citations import count_valid_citations
from agent.config import Settings
from agent.nodes import AgentDeps, ResearchNodes
from schemas.report import SynthesisDraft

MESSAGES = [SystemMessage("system"), HumanMessage("Numbered evidence: [1] … [2] … [3] … [4] …")]


def draft(answer: str) -> SynthesisDraft:
    return SynthesisDraft(answer=answer, confidence_score=0.7)


class ScriptedWriter:
    """Returns scripted drafts (or raises) and records the messages of every call."""

    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[list[Any]] = []
        self.max_tokens: list[int | None] = []

    async def __call__(self, schema: type, messages: list[Any], fast: bool, max_tokens: int | None = None) -> Any:
        self.calls.append(messages)
        self.max_tokens.append(max_tokens)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def nodes_with(writer: ScriptedWriter) -> ResearchNodes:
    deps = AgentDeps(settings=Settings(_env_file=None), llm_factory=lambda fast: None,  # type: ignore[arg-type,return-value]
                     structured=writer, searcher_factory=lambda: None, scraper_factory=lambda: None)  # type: ignore[arg-type,return-value]
    return ResearchNodes(deps)


def test_count_valid_citations() -> None:
    assert count_valid_citations("No citations at all.", 4) == 0
    assert count_valid_citations("A [1][2]. B [2, 4]. Out of range [7].", 4) == 3
    assert count_valid_citations("Linked [[3]](https://x.com/a). Sources section\n## Sources\n[1] x", 4) == 1


async def test_uncited_draft_gets_a_cited_revision() -> None:
    writer = ScriptedWriter(draft("Revised: A [1]. B [2]. C [3][4]."))
    result = await nodes_with(writer)._ensure_citations(draft("Correct facts but no markers."), MESSAGES, 4)
    assert result.answer.startswith("Revised")
    feedback = str(writer.calls[0][-1].content)
    assert "cited only 0 of the 4" in feedback and "Correct facts but no markers." in feedback
    assert writer.calls[0][:2] == MESSAGES  # original context is kept
    assert writer.max_tokens == [1_200]  # cap sized from the short draft, not the full 4,096


async def test_revision_cap_scales_with_draft_length_and_is_bounded() -> None:
    writer = ScriptedWriter(draft("x [1][2][3]"), draft("y [1][2][3]"))
    nodes = nodes_with(writer)
    await nodes._ensure_citations(draft("word " * 1_000), MESSAGES, 4)    # 5k chars -> 2,475 cap
    await nodes._ensure_citations(draft("word " * 20_000), MESSAGES, 4)   # huge draft -> capped at 4,096
    assert writer.max_tokens[0] == 2_475 and writer.max_tokens[1] == 4_096  # 5k chars: 1,250 x 1.5 + 600


async def test_revision_that_is_not_better_is_discarded() -> None:
    writer = ScriptedWriter(draft("Still uncited."))
    original = draft("One cite only [2].")
    assert await nodes_with(writer)._ensure_citations(original, MESSAGES, 4) is original


async def test_revision_failure_keeps_original() -> None:
    writer = ScriptedWriter(ValueError("bad JSON"))
    original = draft("Nothing cited.")
    assert await nodes_with(writer)._ensure_citations(original, MESSAGES, 4) is original


async def test_well_cited_or_small_evidence_needs_no_extra_call() -> None:
    writer = ScriptedWriter()  # any call would fail: no outcomes scripted
    nodes = nodes_with(writer)
    cited = draft("A [1]. B [2]. C [3].")
    assert await nodes._ensure_citations(cited, MESSAGES, 4) is cited
    single = draft("Only source [1].")  # one evidence item: one citation is enough
    assert await nodes._ensure_citations(single, MESSAGES, 1) is single
    assert writer.calls == []
