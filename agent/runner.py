"""High-level async session API over the compiled graph.

A :class:`ResearchSession` owns one LangGraph ``thread_id``. ``run()`` starts a new request or
resumes an interrupted one and yields a single, UI-friendly stream of:

* :class:`TraceEvent` — live Thought / Action / Observation steps,
* :class:`ClarificationRequest` — the graph paused at the Ambiguity Gate,
* :class:`RunResult` — the final validated :class:`ResearchReport` plus run statistics.

It also works as a CLI: ``python -m agent.runner "your research question"``.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from agent.observability import RunRecorder, collect_llm_calls, log_context, summarize_calls
from agent.observability.tracing import run_config, submit_feedback_in_background, trace_url, tracing_config
from agent.state import initial_state
from schemas.clarification import ClarificationRequest
from schemas.events import TraceEvent
from schemas.report import ResearchReport

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Final output of a research run."""

    report: ResearchReport
    iterations: int
    stats: dict[str, int] = field(default_factory=dict)
    queries: list[str] = field(default_factory=list)
    skipped_scrapes: list[dict[str, str]] = field(default_factory=list)
    run_id: str = ""
    llm: dict[str, Any] = field(default_factory=dict)  # summarize_calls(): calls, tokens, per-model breakdown
    duration_s: float = 0.0
    log_file: str = ""
    trace_url: str = ""  # LangSmith link (when tracing is enabled)


def report_summary(result: RunResult) -> dict[str, Any]:
    """Quality/size indicators of a finished report (persisted to the run index for evaluation)."""
    report = result.report
    verified = sum(s.quote_verified for s in report.sources)
    return {
        "confidence": report.confidence_score,
        "cited_sources": len(report.sources),
        "verified_quotes": verified,
        "verified_ratio": round(verified / len(report.sources), 3) if report.sources else 0.0,
        "contradictions": len(report.contradictions_noted),
        "iterations": result.iterations,
        "queries_run": len(result.queries),
        "unique_sources": result.stats.get("results", 0),
        "pages_read": result.stats.get("pages_ok", 0),
        "answer_chars": len(report.answer),
    }


RunItem = TraceEvent | ClarificationRequest | RunResult


class ResearchSession:
    """One conversation thread with the research graph (supports multi-turn clarification)."""

    def __init__(self, graph: CompiledStateGraph, thread_id: str | None = None, *,
                 recorder: RunRecorder | None = None, metadata: dict[str, Any] | None = None) -> None:
        """
        Args:
            graph: Compiled research graph (with a checkpointer).
            thread_id: Conversation id; resuming a clarification reuses it.
            recorder: Optional :class:`RunRecorder` that persists every run to ``logs/runs``.
            metadata: Extra fields recorded with each run (engine, model chains, ...).
        """
        self.graph = graph
        self.thread_id = thread_id or uuid.uuid4().hex
        self.recorder = recorder
        self.metadata = dict(metadata or {})
        self.config: dict[str, Any] = {"configurable": {"thread_id": self.thread_id}, "recursion_limit": 100}

    async def is_waiting_for_user(self) -> bool:
        """True if the thread is paused at the clarification interrupt."""
        snapshot = await self.graph.aget_state(self.config)
        return bool(snapshot.next) and any(t.interrupts for t in snapshot.tasks)

    async def run(self, message: str) -> AsyncIterator[RunItem]:
        """Start a new research request, or resume a paused one with the user's ``message``.

        Each call is one *run* with its own ``run_id``. Every log record emitted during the run is tagged
        with it, every LLM attempt is collected, and (if a recorder is set) the run is persisted to
        ``logs/runs/<date>/<run_id>.jsonl`` plus a summary line in ``logs/runs/index.jsonl``.
        """
        run_id = str(uuid.uuid4())
        resuming = await self.is_waiting_for_user()
        graph_input: Any = Command(resume={"answer": message}) if resuming else initial_state(message)
        run_meta = {"resume": resuming, **self.metadata}
        tracing_on = tracing_config().enabled
        handle = (self.recorder.start(run_id, thread_id=self.thread_id, query=message,
                                      metadata={**run_meta, "langsmith": tracing_on})
                  if self.recorder else None)
        # Root run for LangSmith: same id as the local run, thread metadata, tags.
        config = {**self.config, **run_config(run_id, thread_id=self.thread_id, query=message, metadata=run_meta)}
        started = time.monotonic()
        status: str = "error"
        summary: dict[str, Any] | None = None
        extra: dict[str, Any] = {}

        with log_context(run_id=run_id, thread_id=self.thread_id), collect_llm_calls() as calls:
            logger.info("Run started (%s)", "resume" if resuming else "new request", extra={"query": message[:300]})
            try:
                async for mode, chunk in self.graph.astream(graph_input, config,
                                                            stream_mode=["custom", "updates"]):
                    if mode == "custom" and isinstance(chunk, dict) and "trace" in chunk:
                        event = TraceEvent.model_validate(chunk["trace"])
                        if handle:
                            handle.trace(event.model_dump())
                        yield event
                    elif mode == "updates" and isinstance(chunk, dict) and "__interrupt__" in chunk:
                        status = "awaiting_clarification"
                        for intr in chunk["__interrupt__"]:
                            request = ClarificationRequest.model_validate(intr.value)
                            if handle:
                                handle.write("clarification", **request.model_dump())
                            yield request
                        return

                state = (await self.graph.aget_state(self.config)).values
                status = "completed_without_report"
                if state.get("report"):
                    from tools.url_registry import URLRegistry

                    registry = URLRegistry.from_snapshot(state.get("registry"))
                    result = RunResult(
                        report=ResearchReport.model_validate(state["report"]),
                        iterations=state.get("iteration", 0),
                        stats=registry.stats(),
                        queries=list(registry.queries_run),
                        skipped_scrapes=list(state.get("skipped_scrapes", [])),
                        run_id=run_id,
                        llm=summarize_calls(list(calls)),
                        duration_s=round(time.monotonic() - started, 2),
                        log_file=str(handle.path) if handle else "",
                        trace_url=(await _maybe_trace_url(run_id)) if tracing_on else "",
                    )
                    status, summary = "completed", report_summary(result)
                    if handle:
                        handle.write("report", **result.report.model_dump(mode="json"), queries=result.queries)
                    yield result
            except GeneratorExit:
                status = "cancelled"
                raise
            except BaseException as exc:
                extra = {"error": f"{type(exc).__name__}: {str(exc)[:500]}"}
                logger.exception("Run failed")
                raise
            finally:
                if tracing_on:
                    extra = {**extra, "langsmith_run_id": run_id}
                final = handle.finish(status, calls=list(calls), report=summary, extra=extra) if handle else {}
                if tracing_on and status == "completed":
                    # Attach quality/cost scores to the LangSmith root run (best-effort, non-blocking).
                    submit_feedback_in_background(
                        run_id, final or {**(summary or {}), **summarize_calls(list(calls))})
                logger.info("Run finished: %s in %.1fs", status, time.monotonic() - started,
                            extra={"status": status, **{k: final[k] for k in
                                                        ("llm_calls", "input_tokens", "output_tokens",
                                                         "failover_events") if k in final}})


async def _maybe_trace_url(run_id: str) -> str:
    """LangSmith link for the run, bounded to a few seconds (empty string on any problem)."""
    try:
        return await asyncio.wait_for(asyncio.to_thread(trace_url, run_id), 5) or ""
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
async def _cli(question: str) -> None:  # pragma: no cover - manual entry point
    from agent.graph import default_graph

    from pathlib import Path

    from agent.config import get_settings
    from agent.observability import configure_logging

    settings = get_settings()
    configure_logging(settings.log_dir, settings.log_level, console=False)
    from agent.observability.tracing import configure_tracing

    configure_tracing()
    session = ResearchSession(default_graph(), recorder=RunRecorder(Path(settings.log_dir) / "runs",
                                                                    enabled=settings.run_logs))
    message = question
    while True:
        paused = False
        async for item in session.run(message):
            if isinstance(item, TraceEvent):
                print(f"{item.icon} [{item.node}] {item.message}")
            elif isinstance(item, ClarificationRequest):
                print("\n❓ " + item.reasoning)
                for q in item.questions:
                    print("   • " + q)
                message = input("\nYour answer: ")
                paused = True
            elif isinstance(item, RunResult):
                print("\n" + "=" * 80 + "\n" + item.report.answer + "\n")
                for s in item.report.sources:
                    print(f"[{s.id}] {s.title} — {s.deep_link_url}\n     “{s.supporting_quote}”")
                print(f"\nconfidence={item.report.confidence_score} iterations={item.iterations} stats={item.stats}")
                print(f"run log: {item.log_file}" + (f"\nLangSmith: {item.trace_url}" if item.trace_url else ""))
        if not paused:
            break


if __name__ == "__main__":  # pragma: no cover
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    asyncio.run(_cli(" ".join(sys.argv[1:]) or input("Research question: ")))
