"""Observability: persistent structured logs, LLM call telemetry and per-run records.

* :mod:`~agent.observability.logging_setup`: console + rotating JSON-lines file, run-context tagging
* :mod:`~agent.observability.telemetry`: one record per LLM attempt (provider, model, key, latency, tokens)
* :mod:`~agent.observability.run_log`: one JSONL file per run + a cross-run index for evaluation

Kept dependency-free (stdlib only) so the low-level router can import it without cycles.
"""

from agent.observability.logging_setup import configure_logging, current_log_context, log_context
from agent.observability.run_log import RunRecorder
from agent.observability.telemetry import LLMCallRecord, collect_llm_calls, record_llm_call, summarize_calls

__all__ = [
    "LLMCallRecord",
    "RunRecorder",
    "collect_llm_calls",
    "configure_logging",
    "current_log_context",
    "log_context",
    "record_llm_call",
    "summarize_calls",
]
