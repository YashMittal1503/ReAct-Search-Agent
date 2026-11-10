"""Evaluation harness (#7): dataset, scoring, snapshot/replay/e2e loop and compare. Offline, fakes only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent.config import Settings
from agent.graph import build_graph
from agent.nodes import AgentDeps
from evals.compare import compare, to_markdown
from evals.dataset import GoldenQuestion, Nugget, dataset_stats, load_dataset
from evals.judge import (
    CitationJudgement,
    CitationPair,
    CitationVerdict,
    Judge,
    NuggetJudgement,
    NuggetVerdict,
)
from evals.metrics import bootstrap_ci, citation_pairs, citation_precision, nugget_scores, summarize
from evals.run import Harness, run_mode, write_results
from schemas.report import ResearchReport, Source
from tests.test_graph_flow import QUOTE, FakeAgent, FakeLLM, FakeScraper, FakeSearcher

QUESTION = "Pros and cons of microservices vs monoliths for startups in 2026"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def test_golden_dataset_is_valid() -> None:
    items = load_dataset()
    stats = dataset_stats(items)
    assert stats["questions"] == 25 and stats["ambiguous"] == 2 and stats["nuggets"] >= 100
    assert len({q.id for q in items}) == 25
    assert all(q.nuggets for q in items if not q.expected_clarification)
    assert all(q.clarification_reply for q in items if q.expected_clarification)
    assert len(load_dataset(ids=["q01", "q07"])) == 2 and len(load_dataset(limit=3)) == 3


def test_dataset_validation_rules() -> None:
    with pytest.raises(ValueError, match="at least one nugget"):
        GoldenQuestion(id="x1", question="q", domain="d", as_of="2026-11")
    with pytest.raises(ValueError, match="prefixed"):
        GoldenQuestion(id="x1", question="q", domain="d", as_of="2026-11", nuggets=[Nugget(id="y-n1", text="t")])


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def test_nugget_scores_weighting() -> None:
    nuggets = [Nugget(id="q-n1", text="a"), Nugget(id="q-n2", text="b"),
               Nugget(id="q-n3", text="c", vital=False), Nugget(id="q-n4", text="d", kind="contradiction")]
    verdicts = {"q-n1": NuggetVerdict(nugget_id="q-n1", verdict="supported"),
                "q-n2": NuggetVerdict(nugget_id="q-n2", verdict="partial"),
                "q-n3": NuggetVerdict(nugget_id="q-n3", verdict="supported")}
    s = nugget_scores(nuggets, verdicts)  # weights 1,1,0.5,1 ; scores 1,0.5,1,0
    assert s["nugget_recall"] == round((1 + 0.5 + 0.5 + 0) / 3.5, 4)
    assert s["vital_recall"] == 0.5 and s["contradiction_coverage"] == 0.0
    assert nugget_scores([], {})["nugget_recall"] is None


def test_citation_pairs_and_precision() -> None:
    src = Source(id=1, title="T", url="https://a.com/x", deep_link_url="https://a.com/x", supporting_quote="the quote")
    report = ResearchReport(
        answer="Monoliths are simpler for small teams [[1]](https://a.com/x). Short [[1]](https://a.com/x).\n"
               "- Another claim about deployment speed [[1]](https://a.com/x)",
        confidence_score=0.5, sources=[src])
    pairs = citation_pairs(report, {"https://a.com/x": "intro text. the quote is here. outro"})
    assert [p.claim for p in pairs] == ["Monoliths are simpler for small teams.", "Another claim about deployment speed"]
    assert "the quote" in pairs[0].passage and "intro" in pairs[0].passage
    assert citation_precision([CitationVerdict(index=1, supported="yes"), CitationVerdict(index=2, supported="partial")]) == 0.75
    assert citation_precision([]) is None
    assert bootstrap_ci([1.0, 0.0, 1.0, 1.0]) == bootstrap_ci([1.0, 0.0, 1.0, 1.0])  # deterministic


# ---------------------------------------------------------------------------
# Harness loop with fakes
# ---------------------------------------------------------------------------
class FakeJudgeLLM(FakeLLM):
    """Pipeline fake + judge fake: first nugget supported, others missing; every citation supported."""

    async def __call__(self, schema: type, messages: list[Any], fast: bool, max_tokens: int | None = None) -> Any:
        if schema is NuggetJudgement:
            ids = [line.split(":")[0] for line in str(messages[-1].content).split("NUGGETS:\n")[1].split("\n\nREPORT")[0].splitlines()]
            return NuggetJudgement(verdicts=[NuggetVerdict(nugget_id=ids[0], verdict="supported", evidence="x")])
        if schema is CitationJudgement:
            n = str(messages[-1].content).count("PAIR ")
            return CitationJudgement(verdicts=[CitationVerdict(index=i, supported="yes") for i in range(1, n + 1)])
        return await super().__call__(schema, messages, fast, max_tokens)


def _harness(tmp_path: Path, synthesis_mode: str = "two_step") -> tuple[Harness, FakeJudgeLLM]:
    llm = FakeJudgeLLM()
    settings = Settings(max_iterations=1, max_scrapes_per_iteration=3, synthesis_mode=synthesis_mode, _env_file=None)
    deps = AgentDeps(settings=settings, llm_factory=lambda fast: None, structured=llm,  # type: ignore[arg-type,return-value]
                     searcher_factory=FakeSearcher, scraper_factory=FakeScraper,  # type: ignore[arg-type]
                     agent_factory=lambda **kw: FakeAgent(kw["tools"], False), today=lambda: "November 10, 2026")
    return Harness(settings, deps=deps, graph_factory=lambda: build_graph(deps), snapshot_dir=tmp_path / "snaps"), llm


GQ = GoldenQuestion(id="t1", question=QUESTION, domain="software", as_of="2026-11",
                    nuggets=[Nugget(id="t1-n1", text="Monoliths suit small teams"), Nugget(id="t1-n2", text="Other")])
AMBIG = GoldenQuestion(id="t2", question="microservices?", domain="ambiguous", as_of="2026-11",
                       expected_clarification=True, clarification_reply="Backend web apps, team of 8 engineers")


async def test_snapshot_then_replay_judges_frozen_evidence(tmp_path: Path) -> None:
    harness, _ = _harness(tmp_path)
    snap = await harness.snapshot(GQ)
    assert snap["status"] == "ok"
    data = json.loads(Path(snap["path"]).read_text(encoding="utf-8"))
    assert data["state"]["registry"]["pages"] and data["baseline_report"]["sources"]
    assert (await harness.snapshot(GQ))["status"] == "skipped"  # snapshots are reused unless forced

    row = await harness.replay(GQ)
    assert row["status"] == "ok"
    assert row["nugget_recall"] == 0.5 and row["citation_precision"] == 1.0 and row["citations_judged"] >= 1
    assert row["nugget_verdicts"]["t1-n2"]["verdict"] == "missing"  # skipped by judge -> missing
    assert "llm" in row and row["cited_sources"] >= 1

    # Same frozen evidence, legacy single-step variant: the synthesizer sees raw page excerpts.
    single, single_llm = _harness(tmp_path, "single_step")
    assert (await single.replay(GQ))["status"] == "ok"
    assert QUOTE in single_llm.synthesis_prompt and "Key points:" not in single_llm.synthesis_prompt


async def test_e2e_checks_clarification_gate_and_run_mode_skips(tmp_path: Path) -> None:
    harness, _ = _harness(tmp_path)
    rows = await run_mode("e2e", harness, [GQ, AMBIG])
    by_id = {r["id"]: r for r in rows}
    assert by_id["t2"]["clarification_requested"] is True and by_id["t2"]["status"] == "ok"
    assert by_id["t1"]["clarification_requested"] is False and by_id["t1"]["nugget_recall"] == 0.5

    summary = summarize(rows, [GQ, AMBIG])
    assert summary["gate_recall"] == 1.0 and summary["gate_false_positive_rate"] == 0.0
    assert summary["nugget_recall"]["mean"] == 0.5

    replay_rows = await run_mode("replay", harness, [GQ, AMBIG])  # ambiguous skipped; no snapshot yet
    assert [r["id"] for r in replay_rows] == ["t1"] and replay_rows[0]["status"] == "missing_snapshot"


# ---------------------------------------------------------------------------
# Compare + ship rule
# ---------------------------------------------------------------------------
def _write_run(tmp_path: Path, name: str, recalls: list[float], precisions: list[float], tokens: int) -> Path:
    questions = [GoldenQuestion(id=f"c{i}", question="q", domain="d", as_of="2026-11",
                                nuggets=[Nugget(id=f"c{i}-n1", text="t")]) for i in range(len(recalls))]
    rows = [{"id": q.id, "status": "ok", "nugget_recall": r, "citation_precision": p,
             "llm": {"input_tokens": tokens // len(recalls)}} for q, r, p in zip(questions, recalls, precisions)]
    out = tmp_path / name
    write_results(rows, questions, {"run_name": name, "variant": name, "mode": "replay"}, out)
    return out


def test_compare_ship_rule(tmp_path: Path) -> None:
    base = _write_run(tmp_path, "baseline", [0.8, 0.6, 0.7, 0.9, 0.5, 0.75], [0.9] * 6, 60_000)
    good = _write_run(tmp_path, "good", [0.8, 0.6, 0.7, 0.9, 0.5, 0.75], [0.9] * 6, 30_000)
    bad = _write_run(tmp_path, "bad", [0.6, 0.4, 0.5, 0.7, 0.3, 0.55], [0.9] * 6, 30_000)
    costly = _write_run(tmp_path, "costly", [0.8, 0.6, 0.7, 0.9, 0.5, 0.75], [0.9] * 6, 58_000)

    assert compare(base, good)["verdict"] == "SHIP"
    result = compare(base, bad)
    assert result["verdict"] == "DO NOT SHIP" and result["metrics"]["nugget_recall"]["paired_delta"] == -0.2
    assert compare(base, costly)["checks"]["token_saving"] == "FAIL"
    assert "## Verdict: SHIP" in to_markdown(compare(base, good))


async def test_judge_handles_empty_inputs() -> None:
    async def never(*a: Any, **k: Any) -> Any:
        raise AssertionError("should not be called")

    judge = Judge(never)
    assert await judge.nuggets("q", "a", []) == {} and await judge.citations([]) == []
    assert isinstance(CitationPair(source_id=1, claim="c", source_title="t", passage="p"), CitationPair)
