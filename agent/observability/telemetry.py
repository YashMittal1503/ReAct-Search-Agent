"""Per-attempt LLM call telemetry emitted by the model router.

Every provider attempt (successful or not) becomes an :class:`LLMCallRecord`. Records are:

* logged to the ``llm.calls`` logger as structured fields (so they land in ``logs/app.jsonl``);
* appended to the current run's collector, if one is active (:func:`collect_llm_calls`), which the
  :class:`~agent.observability.run_log.RunRecorder` persists and summarizes.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger("llm.calls")

_collector: ContextVar[list["LLMCallRecord"] | None] = ContextVar("llm_call_collector", default=None)


@dataclass
class LLMCallRecord:
    """One attempt against one provider/model/key."""

    role: str
    provider: str
    model: str
    key: str                      # masked
    outcome: str                  # "ok" or an ErrorKind value ("rate_limit", "transient", ...)
    latency_ms: float
    attempt: int                  # 1-based attempt number within the logical call
    est_tokens: int = 0           # router's pre-call estimate (prompt + completion cap)
    input_tokens: int = 0         # provider-reported usage (successful calls only)
    output_tokens: int = 0
    error: str = ""
    ts: float = field(default_factory=time.time)

    @property
    def ok(self) -> bool:
        return self.outcome == "ok"


@contextmanager
def collect_llm_calls() -> Iterator[list[LLMCallRecord]]:
    """Collect every :class:`LLMCallRecord` produced in this context (incl. child tasks)."""
    records: list[LLMCallRecord] = []
    token = _collector.set(records)
    try:
        yield records
    finally:
        try:
            _collector.reset(token)
        except ValueError:  # finalized from another context (async-generator cleanup)
            pass


def record_llm_call(record: LLMCallRecord) -> None:
    """Log the record and add it to the active collector. Never raises."""
    try:
        sink = _collector.get()
        if sink is not None:
            sink.append(record)
        level = logging.INFO if record.ok else logging.WARNING
        logger.log(level, "%s %s:%s key=%s %.0fms", record.outcome, record.provider, record.model, record.key,
                   record.latency_ms, extra={"event": "llm_call", **asdict(record)})
    except Exception:  # noqa: BLE001 - telemetry must never break a call
        pass


def summarize_calls(records: list[LLMCallRecord]) -> dict[str, Any]:
    """Totals and a per-model breakdown for a run."""
    by_model: dict[str, dict[str, Any]] = defaultdict(lambda: {"calls": 0, "ok": 0, "failed": 0, "input_tokens": 0,
                                                               "output_tokens": 0, "latency_ms": 0.0})
    for r in records:
        m = by_model[f"{r.provider}:{r.model}"]
        m["calls"] += 1
        m["ok" if r.ok else "failed"] += 1
        m["input_tokens"] += r.input_tokens
        m["output_tokens"] += r.output_tokens
        m["latency_ms"] += r.latency_ms
    ok = [r for r in records if r.ok]
    return {
        "llm_calls": len(ok),
        "llm_attempts": len(records),
        "failed_attempts": len(records) - len(ok),
        "input_tokens": sum(r.input_tokens for r in ok),
        "output_tokens": sum(r.output_tokens for r in ok),
        "llm_time_s": round(sum(r.latency_ms for r in records) / 1000, 2),
        "failures_by_kind": dict(sorted({k: sum(1 for r in records if r.outcome == k)
                                         for k in {r.outcome for r in records if not r.ok}}.items())),
        "by_model": {k: {**v, "latency_ms": round(v["latency_ms"])} for k, v in sorted(by_model.items())},
    }
