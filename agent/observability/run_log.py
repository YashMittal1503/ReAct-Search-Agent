"""Persistent per-run records for debugging and evaluation.

Layout (all JSON lines, append-only)::

    logs/runs/2026-11-10/<run_id>.jsonl   # run_start, every trace event, every LLM call, run_end
    logs/runs/index.jsonl                 # one summary line per run (compare runs, feed evals)

Writes are best-effort: an I/O problem is logged and never interrupts a research run.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.observability.telemetry import LLMCallRecord, summarize_calls

logger = logging.getLogger(__name__)
INDEX_NAME = "index.jsonl"


class RunRecorder:
    """Writes one run file per research run plus a cross-run index."""

    def __init__(self, base_dir: str | Path = "logs/runs", *, enabled: bool = True) -> None:
        self.base_dir = Path(base_dir)
        self.enabled = enabled
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ public API
    def start(self, run_id: str, *, thread_id: str, query: str, metadata: dict[str, Any] | None = None) -> "RunHandle":
        path = self.base_dir / datetime.now(timezone.utc).strftime("%Y-%m-%d") / f"{run_id}.jsonl"
        handle = RunHandle(self, run_id, thread_id, path, query)
        handle.write("run_start", query=query, thread_id=thread_id, **(metadata or {}))
        return handle

    def read_index(self) -> list[dict[str, Any]]:
        """All run summaries (oldest first)."""
        path = self.base_dir / INDEX_NAME
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    # ------------------------------------------------------------------ internals
    def _append(self, path: Path, record: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            line = json.dumps(record, default=str, ensure_ascii=False)
            with self._lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except Exception:  # noqa: BLE001 - persistence must never break a run
            logger.warning("Could not write run log %s", path, exc_info=True)


class RunHandle:
    """Writer for a single run."""

    def __init__(self, recorder: RunRecorder, run_id: str, thread_id: str, path: Path, query: str) -> None:
        self.recorder, self.run_id, self.thread_id, self.path, self.query = recorder, run_id, thread_id, path, query
        self.started = time.time()
        self.trace_events = 0
        self.failover_events = 0
        self._finished = False

    def write(self, event: str, **data: Any) -> None:
        self.recorder._append(self.path, {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                                          "run_id": self.run_id, "event": event, **data})

    def trace(self, trace_event: dict[str, Any]) -> None:
        self.trace_events += 1
        if trace_event.get("kind") == "failover":
            self.failover_events += 1
        self.write("trace", **trace_event)

    def finish(self, status: str, *, calls: list[LLMCallRecord], report: dict[str, Any] | None = None,
               extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Persist LLM calls + run_end, append the index line, and return the summary."""
        if self._finished:
            return {}
        self._finished = True
        for call in calls:
            self.write("llm_call", **asdict(call))
        summary: dict[str, Any] = {
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "status": status,  # completed | awaiting_clarification | error
            "query": self.query[:300],
            "started_at": datetime.fromtimestamp(self.started, tz=timezone.utc).isoformat(timespec="seconds"),
            "duration_s": round(time.time() - self.started, 2),
            "trace_events": self.trace_events,
            "failover_events": self.failover_events,
            **summarize_calls(calls),
            **(report or {}),
            **(extra or {}),
            "log_file": str(self.path),
        }
        self.write("run_end", **summary)
        self.recorder._append(self.recorder.base_dir / INDEX_NAME, summary)
        return summary
