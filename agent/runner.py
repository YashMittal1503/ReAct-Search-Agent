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
import sys
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from agent.state import initial_state
from schemas.clarification import ClarificationRequest
from schemas.events import TraceEvent
from schemas.report import ResearchReport

@dataclass
class RunResult:
    """Final output of a research run."""

    report: ResearchReport
    iterations: int
    stats: dict[str, int] = field(default_factory=dict)
    queries: list[str] = field(default_factory=list)
    skipped_scrapes: list[dict[str, str]] = field(default_factory=list)


RunItem = TraceEvent | ClarificationRequest | RunResult


class ResearchSession:
    """One conversation thread with the research graph (supports multi-turn clarification)."""

    def __init__(self, graph: CompiledStateGraph, thread_id: str | None = None) -> None:
        self.graph = graph
        self.thread_id = thread_id or uuid.uuid4().hex
        self.config: dict[str, Any] = {"configurable": {"thread_id": self.thread_id}, "recursion_limit": 100}

    async def is_waiting_for_user(self) -> bool:
        """True if the thread is paused at the clarification interrupt."""
        snapshot = await self.graph.aget_state(self.config)
        return bool(snapshot.next) and any(t.interrupts for t in snapshot.tasks)

    async def run(self, message: str) -> AsyncIterator[RunItem]:
        """Start a new research request, or resume a paused one with the user's ``message``."""
        if await self.is_waiting_for_user():
            graph_input: Any = Command(resume={"answer": message})
        else:
            graph_input = initial_state(message)

        async for mode, chunk in self.graph.astream(graph_input, self.config, stream_mode=["custom", "updates"]):
            if mode == "custom" and isinstance(chunk, dict) and "trace" in chunk:
                yield TraceEvent.model_validate(chunk["trace"])
            elif mode == "updates" and isinstance(chunk, dict) and "__interrupt__" in chunk:
                for intr in chunk["__interrupt__"]:
                    yield ClarificationRequest.model_validate(intr.value)
                return

        state = (await self.graph.aget_state(self.config)).values
        if state.get("report"):
            from tools.url_registry import URLRegistry

            registry = URLRegistry.from_snapshot(state.get("registry"))
            yield RunResult(
                report=ResearchReport.model_validate(state["report"]),
                iterations=state.get("iteration", 0),
                stats=registry.stats(),
                queries=list(registry.queries_run),
                skipped_scrapes=list(state.get("skipped_scrapes", [])),
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
async def _cli(question: str) -> None:  # pragma: no cover - manual entry point
    from agent.graph import default_graph

    session = ResearchSession(default_graph())
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
        if not paused:
            break


if __name__ == "__main__":  # pragma: no cover
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    asyncio.run(_cli(" ".join(sys.argv[1:]) or input("Research question: ")))
