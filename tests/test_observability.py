"""Persistent logs, LLM call telemetry and run records (#15). Offline; writes only to tmp_path."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from agent.observability import (
    LLMCallRecord,
    RunRecorder,
    collect_llm_calls,
    configure_logging,
    log_context,
    summarize_calls,
)
from agent.runner import ResearchSession, RunResult
from schemas import ClarificationRequest
from tests.test_graph_flow import _collect, _graph
from tests.test_router import MSGS, Script, api_error, make_router


@pytest.fixture
def clean_logging():
    """Remove our handlers after the test so global logging is left as it was."""
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_deep_research_handler", False):
            root.removeHandler(handler)
            handler.close()


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_json_file_logging_with_run_context_and_extras(tmp_path: Path, clean_logging: None) -> None:
    log_path = configure_logging(tmp_path, "INFO", console=False)
    configure_logging(tmp_path, "INFO", console=False)  # idempotent: no duplicate handlers
    assert sum(getattr(h, "_deep_research_handler", False) for h in logging.getLogger().handlers) == 1

    log = logging.getLogger("agent.test")
    with log_context(run_id="run-1", thread_id="thread-9"):
        log.info("inside run", extra={"provider": "groq", "tokens": 42})
    log.info("outside run")
    logging.getLogger("httpx").info("noisy request line")  # quietened to WARNING

    records = _lines(log_path)
    assert [r["message"] for r in records] == ["inside run", "outside run"]
    inside, outside = records
    assert inside["run_id"] == "run-1" and inside["thread_id"] == "thread-9"
    assert inside["provider"] == "groq" and inside["tokens"] == 42 and inside["level"] == "INFO"
    assert "run_id" not in outside


async def test_router_records_every_attempt() -> None:
    script = Script()
    router = make_router(script)
    script.outcomes[("groq", "g1", "llama")] = [api_error(429, "Rate limit reached")]
    with collect_llm_calls() as calls:
        await router.agenerate("main", MSGS)
    assert [(c.provider, c.outcome, c.attempt) for c in calls] == [("groq", "rate_limit", 1), ("groq", "ok", 2)]
    assert all(c.key == "***" for c in calls)  # keys are masked; raw keys never reach logs
    ok = calls[-1]
    assert ok.input_tokens == 10 and ok.output_tokens == 5 and ok.est_tokens > 0 and ok.latency_ms >= 0
    assert "RateLimitError" in calls[0].error

    with collect_llm_calls() as outer_none:
        pass
    await router.agenerate("main", MSGS)  # no active collector: must not fail
    assert outer_none == []


def test_summarize_calls() -> None:
    calls = [
        LLMCallRecord("main", "groq", "a", "k1", "rate_limit", 10, 1),
        LLMCallRecord("main", "groq", "a", "k2", "ok", 100, 2, input_tokens=50, output_tokens=20),
        LLMCallRecord("fast", "gemini", "b", "k3", "ok", 200, 1, input_tokens=5, output_tokens=5),
    ]
    s = summarize_calls(calls)
    assert (s["llm_calls"], s["llm_attempts"], s["failed_attempts"]) == (2, 3, 1)
    assert (s["input_tokens"], s["output_tokens"], s["llm_time_s"]) == (55, 25, 0.31)
    assert s["failures_by_kind"] == {"rate_limit": 1}
    assert s["by_model"]["groq:a"] == {"calls": 2, "ok": 1, "failed": 1, "input_tokens": 50,
                                       "output_tokens": 20, "latency_ms": 110}


async def test_run_recorder_persists_clarification_and_completed_runs(tmp_path: Path) -> None:
    recorder = RunRecorder(tmp_path / "runs")
    session = ResearchSession(_graph(), recorder=recorder, metadata={"engine": "deep"})

    first = await _collect(session, "microservices?")
    assert isinstance(first[-1], ClarificationRequest)
    second = await _collect(session, "Backend web apps, team of 8 engineers")
    result = second[-1]
    assert isinstance(result, RunResult)

    index = recorder.read_index()
    assert [r["status"] for r in index] == ["awaiting_clarification", "completed"]
    assert index[0]["thread_id"] == index[1]["thread_id"] == session.thread_id
    done = index[1]
    assert done["run_id"] == result.run_id and done["iterations"] == 2
    assert done["cited_sources"] == len(result.report.sources) and 0 <= done["verified_ratio"] <= 1
    assert done["trace_events"] > 0 and done["duration_s"] >= 0

    events = [e["event"] for e in _lines(Path(result.log_file))]
    assert events[0] == "run_start" and events[-1] == "run_end"
    assert "trace" in events and "report" in events
    clar_events = [e["event"] for e in _lines(Path(index[0]["log_file"]))]
    assert "clarification" in clar_events
    start = _lines(Path(result.log_file))[0]
    assert start["engine"] == "deep" and start["resume"] is True


def test_recorder_failures_never_raise(tmp_path: Path) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    recorder = RunRecorder(blocker / "runs")  # mkdir under a file fails
    handle = recorder.start("r1", thread_id="t", query="q")
    handle.trace({"kind": "status", "message": "m"})
    assert handle.finish("completed", calls=[])["status"] == "completed"

    disabled = RunRecorder(tmp_path / "off", enabled=False)
    disabled.start("r2", thread_id="t", query="q").finish("completed", calls=[])
    assert not (tmp_path / "off").exists()
