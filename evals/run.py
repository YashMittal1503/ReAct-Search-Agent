"""Evaluation harness CLI.

Modes
-----
``snapshot``  Run the full pipeline once per answerable question and **freeze the evidence** (URL
              registry with scraped pages, evaluator verdict, research notes) to
              ``evals/snapshots/<qid>.json``. This is the expensive step (search + scraping + LLM),
              done once and shared by every variant.
``replay``    Re-run only the **synthesis** step of a variant on the frozen evidence, then judge it.
              Retrieval noise is removed, which is the fair way to A/B synthesis changes (two-step vs
              single-step, RAG, prompts).
``e2e``       Full pipeline per question, including the Ambiguity Gate (ambiguous questions check
              whether clarification was requested). Most realistic, but noisiest.

Examples::

    python -m evals.run snapshot --limit 5
    python -m evals.run replay --variant baseline
    python -m evals.run replay --variant single_step
    python -m evals.compare evals/results/<baseline-run> evals/results/<other-run>

Results go to ``evals/results/<UTC timestamp>-<mode>-<variant>/`` (``results.jsonl``,
``summary.json``, ``summary.md``). Snapshots and results contain scraped web content, so they're
git-ignored.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.config import Settings
from agent.nodes import AgentDeps, ResearchNodes
from agent.observability import collect_llm_calls, summarize_calls
from agent.runner import ResearchSession, RunResult, report_summary
from evals.dataset import DEFAULT_DATASET, GoldenQuestion, dataset_stats, load_dataset
from evals.judge import Judge
from evals.metrics import citation_pairs, citation_precision, nugget_scores, summarize, summary_markdown
from evals.variants import settings_for
from schemas.clarification import ClarificationRequest
from schemas.report import ResearchReport
from tools.url_registry import URLRegistry

logger = logging.getLogger("evals")

ROOT = Path(__file__).parent
SNAPSHOT_DIR = ROOT / "snapshots"
RESULTS_DIR = ROOT / "results"


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=ROOT, timeout=5).stdout.strip() or "?"
    except Exception:  # noqa: BLE001
        return "?"


# ---------------------------------------------------------------------------
# Building blocks (dependency-injectable for tests)
# ---------------------------------------------------------------------------
class Harness:
    """Runs questions through the pipeline and judges the output."""

    def __init__(self, settings: Settings, *, deps: AgentDeps | None = None, judge: Judge | None = None,
                 graph_factory: Any = None, snapshot_dir: Path = SNAPSHOT_DIR) -> None:
        self.settings = settings
        self.deps = deps or AgentDeps(settings=settings)
        self.judge = judge or Judge(self.deps.structured)  # same router -> provider-agnostic judging
        self.graph_factory = graph_factory
        self.snapshot_dir = snapshot_dir

    def _graph(self) -> Any:
        if self.graph_factory is not None:
            return self.graph_factory()
        from agent.graph import build_graph

        return build_graph(self.deps)

    async def _run_session(self, q: GoldenQuestion, metadata: dict[str, Any]) -> tuple[ResearchSession, RunResult | None, bool]:
        """Full pipeline; answers clarification requests with the golden reply. Returns (session, result, asked)."""
        session = ResearchSession(self._graph(), metadata=metadata)
        asked, result, message = False, None, q.question
        for _ in range(self.settings.max_clarification_rounds + 1):
            paused = False
            async for item in session.run(message):
                if isinstance(item, ClarificationRequest):
                    asked, paused, message = True, True, q.clarification_reply
                elif isinstance(item, RunResult):
                    result = item
            if not paused:
                break
        return session, result, asked

    async def judge_report(self, q: GoldenQuestion, report: ResearchReport,
                           source_texts: dict[str, str] | None) -> dict[str, Any]:
        verdicts = await self.judge.nuggets(q.question, report.answer, q.nuggets)
        pairs = citation_pairs(report, source_texts)
        cverdicts = await self.judge.citations(pairs)
        return {
            **nugget_scores(q.nuggets, verdicts),
            "citation_precision": citation_precision(cverdicts),
            "citations_judged": len(cverdicts),
            "nugget_verdicts": {k: v.model_dump() for k, v in verdicts.items()},
            "citation_verdicts": [{**p.model_dump(exclude={"passage"}), **v.model_dump()} for p, v in zip(pairs, cverdicts)],
        }

    # ------------------------------------------------------------------ modes
    async def snapshot(self, q: GoldenQuestion, *, force: bool = False) -> dict[str, Any]:
        path = self.snapshot_dir / f"{q.id}.json"
        if path.exists() and not force:
            return {"id": q.id, "status": "skipped", "path": str(path)}
        started = time.monotonic()
        with collect_llm_calls() as calls:
            session, result, asked = await self._run_session(q, {"eval": "snapshot", "question_id": q.id})
        state = (await session.graph.aget_state(session.config)).values
        if result is None or not state.get("registry"):
            return {"id": q.id, "status": "error", "error": "pipeline produced no report/evidence"}
        snap = {
            "id": q.id, "question": q.question, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git_commit(), "engine": self.settings.research_engine,
            "clarification_requested": asked,
            "state": {k: state.get(k) for k in ("query", "clarified_query", "registry", "evaluation",
                                                 "research_notes", "iteration")},
            "baseline_report": result.report.model_dump(mode="json"),
            "llm": summarize_calls(list(calls)), "duration_s": round(time.monotonic() - started, 1),
        }
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snap, ensure_ascii=False, default=str), encoding="utf-8")
        return {"id": q.id, "status": "ok", "path": str(path), "duration_s": snap["duration_s"]}

    async def replay(self, q: GoldenQuestion) -> dict[str, Any]:
        path = self.snapshot_dir / f"{q.id}.json"
        if not path.exists():
            return {"id": q.id, "status": "missing_snapshot"}
        snap = json.loads(path.read_text(encoding="utf-8"))
        state = {**snap["state"], "report": None}
        nodes = ResearchNodes(self.deps)
        started = time.monotonic()
        with collect_llm_calls() as calls:
            update = await nodes.synthesize(state)  # only the synthesis step, on frozen evidence
        report = ResearchReport.model_validate(update["report"])
        registry = URLRegistry.from_snapshot(state["registry"])
        texts = {ev.url: ev.text for ev in registry.evidence()}
        llm = summarize_calls(list(calls))
        duration = round(time.monotonic() - started, 2)
        judged = await self.judge_report(q, report, texts)
        fake = RunResult(report=report, iterations=state.get("iteration", 0), stats=registry.stats(),
                         queries=list(registry.queries_run))
        return {"id": q.id, "status": "ok", **report_summary(fake), **judged, "llm": llm, "duration_s": duration,
                "answer": report.answer}

    async def e2e(self, q: GoldenQuestion) -> dict[str, Any]:
        started = time.monotonic()
        with collect_llm_calls() as calls:
            _, result, asked = await self._run_session(q, {"eval": "e2e", "question_id": q.id})
        row: dict[str, Any] = {"id": q.id, "clarification_requested": asked, "llm": summarize_calls(list(calls)),
                               "duration_s": round(time.monotonic() - started, 2)}
        if result is None:
            return {**row, "status": "error", "error": "no report"}
        judged = await self.judge_report(q, result.report, None) if q.nuggets else {}
        return {**row, "status": "ok", **report_summary(result), **judged, "answer": result.report.answer}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
async def run_mode(mode: str, harness: Harness, questions: list[GoldenQuestion], *, force: bool = False,
                   on_row: Any = None) -> list[dict[str, Any]]:
    """Run questions sequentially (rate-limit friendly). Errors are recorded per question, never fatal."""
    rows = []
    for q in questions:
        if mode in ("snapshot", "replay") and q.expected_clarification:
            continue  # ambiguous questions only make sense end-to-end
        try:
            if mode == "snapshot":
                row = await harness.snapshot(q, force=force)
            elif mode == "replay":
                row = await harness.replay(q)
            else:
                row = await harness.e2e(q)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Question %s failed", q.id)
            row = {"id": q.id, "status": "error", "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        rows.append(row)
        if on_row:
            on_row(row)
    return rows


def write_results(rows: list[dict[str, Any]], questions: list[GoldenQuestion], meta: dict[str, Any],
                  out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows),
                                           encoding="utf-8")
    summary = summarize(rows, questions)
    (out_dir / "summary.json").write_text(json.dumps({"meta": meta, "summary": summary}, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(summary_markdown(summary, meta), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI wiring
    parser = argparse.ArgumentParser(prog="python -m evals.run", description=__doc__.split("\n\n")[0])
    parser.add_argument("mode", choices=["snapshot", "replay", "e2e", "stats"])
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--ids", nargs="*", help="Only these question ids (e.g. q01 q07)")
    parser.add_argument("--domains", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--force", action="store_true", help="snapshot: overwrite existing snapshots")
    args = parser.parse_args(argv)

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    questions = load_dataset(args.dataset, ids=args.ids, domains=args.domains, limit=args.limit)
    if args.mode == "stats":
        print(json.dumps(dataset_stats(questions), indent=2))
        return 0

    from agent.observability import configure_logging
    from agent.observability.tracing import configure_tracing

    settings = settings_for(args.variant)
    configure_logging(settings.log_dir, settings.log_level, console=False)
    configure_tracing()
    harness = Harness(settings)
    run_name = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{args.mode}-{args.variant}"
    meta = {"run_name": run_name, "mode": args.mode, "variant": args.variant, "git_commit": git_commit(),
            "dataset_size": len(questions), "judge": "router main chain",
            "chains": {r: [t.label for t in c] for r, c in (harness.deps.router.chains.items()
                                                             if harness.deps.router else [])}}

    def show(row: dict[str, Any]) -> None:
        bits = [f"{k}={row[k]}" for k in ("status", "nugget_recall", "citation_precision", "duration_s") if k in row]
        print(f"  {row['id']}: " + " ".join(bits) + (f" error={row.get('error')}" if row.get("error") else ""))

    print(f"{args.mode} · variant={args.variant} · {len(questions)} questions")
    rows = asyncio.run(run_mode(args.mode, harness, questions, force=args.force, on_row=show))
    if args.mode == "snapshot":
        print(f"Snapshots in {SNAPSHOT_DIR}")
        return 0
    out = RESULTS_DIR / run_name
    write_results(rows, questions, meta, out)
    print((out / "summary.md").read_text(encoding="utf-8"))
    print(f"Results: {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
