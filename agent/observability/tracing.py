"""LangSmith tracing: opt-in configuration, rich root-run config, run URLs and evaluation feedback.

LangChain/LangGraph trace to LangSmith automatically once ``LANGSMITH_TRACING=true`` and
``LANGSMITH_API_KEY`` are in the *process environment*. pydantic-settings reads ``.env`` without
exporting it, so :func:`configure_tracing` exports the LangSmith variables explicitly. Without a key
nothing is enabled and no network calls are made.

What a trace contains:

* root run ``deep-research`` whose id **equals the local run id** (logs and LangSmith share one id),
  tagged with the engine and iteration cap, and with ``thread_id`` metadata so LangSmith *Threads*
  group the turns of one conversation (including clarification turns);
* every graph node, the deep agent, sub-agents and tool calls (automatic);
* one child run per **router attempt**, named ``provider:model`` with provider/model/masked key/
  attempt metadata, so failed attempts appear next to the one that succeeded.

After each completed run, quality and cost indicators are attached as **feedback** to the root run, so
they can be filtered and charted across runs in LangSmith.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_PROJECT = "deep-research-agent"
ROOT_RUN_NAME = "deep-research"
FEEDBACK_TIMEOUT_S = 15.0
_OFF = {"false", "0", "off", "no"}

#: Summary fields (from the run index) attached to the root run as LangSmith feedback scores.
FEEDBACK_KEYS = ("confidence", "verified_ratio", "cited_sources", "iterations", "failover_events",
                 "llm_calls", "failed_attempts", "total_tokens", "duration_s")


@dataclass(frozen=True)
class TracingConfig:
    enabled: bool
    project: str = DEFAULT_PROJECT
    endpoint: str | None = None


_config = TracingConfig(enabled=False)
_url_lock = threading.Lock()
_project_ids: dict[str, str] = {}
_background: set[asyncio.Task[Any]] = set()


def configure_tracing(env: Mapping[str, str] | None = None) -> TracingConfig:
    """Enable LangSmith tracing iff a key is configured (and tracing isn't explicitly turned off)."""
    global _config
    if env is None:
        from agent.resilience.providers import load_env

        env = load_env()
    key = env.get("LANGSMITH_API_KEY") or env.get("LANGCHAIN_API_KEY") or ""
    flag = (env.get("LANGSMITH_TRACING") or env.get("LANGCHAIN_TRACING_V2") or "true").strip().lower()
    project = env.get("LANGSMITH_PROJECT") or env.get("LANGCHAIN_PROJECT") or DEFAULT_PROJECT
    endpoint = env.get("LANGSMITH_ENDPOINT") or env.get("LANGCHAIN_ENDPOINT") or None

    if key and flag not in _OFF:
        os.environ["LANGSMITH_API_KEY"] = key
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_PROJECT"] = project
        if endpoint:
            os.environ["LANGSMITH_ENDPOINT"] = endpoint
        _config = TracingConfig(enabled=True, project=project, endpoint=endpoint)
        logger.info("LangSmith tracing enabled (project=%s)", project)
    else:
        # Make sure a stray LANGSMITH_TRACING=true without a key doesn't produce warnings / attempts.
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        _config = TracingConfig(enabled=False, project=project, endpoint=endpoint)
    return _config


def tracing_config() -> TracingConfig:
    return _config


def run_config(run_id: str, *, thread_id: str, query: str, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Root-run fields merged into the graph's ``RunnableConfig`` (harmless when tracing is off)."""
    meta = dict(metadata or {})
    tags = [f"engine:{meta['engine']}"] if meta.get("engine") else []
    if meta.get("resume"):
        tags.append("clarification-reply")
    return {
        "run_id": uuid.UUID(run_id),
        "run_name": ROOT_RUN_NAME,
        "tags": tags,
        "metadata": {"thread_id": thread_id, "query": query[:500],
                     **{k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}},
    }


def attempt_config(*, role: str, provider: str, model: str, key: str, attempt: int,
                   callbacks: Any = None) -> dict[str, Any]:
    """Child-run config for one router attempt.

    ``callbacks`` should be the parent chat-model run's ``run_manager.get_child()``, which nests each
    attempt under the ``ResilientChatModel`` run in the trace tree.
    """
    config: dict[str, Any] = {
        "run_name": f"{provider}:{model}",
        "tags": [f"provider:{provider}", f"role:{role}"],
        "metadata": {"provider": provider, "model": model, "key": key, "role": role, "attempt": attempt},
    }
    if callbacks is not None:
        config["callbacks"] = callbacks
    return config


def trace_url(run_id: str, client: Any = None) -> str | None:
    """LangSmith UI link for a run (``?poll=true`` waits for ingestion). ``None`` if unavailable.

    The project id is looked up once and cached; until the project exists (first ever trace) this
    returns ``None`` and is retried on the next run.
    """
    if not _config.enabled:
        return None
    try:
        from langsmith import Client

        client = client or Client()
        with _url_lock:
            project_id = _project_ids.get(_config.project)
            if project_id is None:
                project_id = str(client.read_project(project_name=_config.project).id)
                _project_ids[_config.project] = project_id
        return client.get_run_url(run=SimpleNamespace(id=run_id, session_id=project_id))
    except Exception:  # noqa: BLE001 - a missing link must never affect the run
        logger.debug("Could not build LangSmith URL", exc_info=True)
        return None


def _send_feedback(run_id: str, summary: Mapping[str, Any], client: Any = None) -> int:
    from langchain_core.tracers.langchain import wait_for_all_tracers
    from langsmith import Client

    wait_for_all_tracers()  # make sure the root run was submitted before attaching feedback
    client = client or Client()
    values = {**summary, "total_tokens": (summary.get("input_tokens") or 0) + (summary.get("output_tokens") or 0)}
    sent = 0
    for key in FEEDBACK_KEYS:
        value = values.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            client.create_feedback(run_id, key=key, score=value, comment="auto: deep-research run summary")
            sent += 1
    return sent


async def submit_feedback(run_id: str, summary: Mapping[str, Any], *, client: Any = None,
                          timeout: float = FEEDBACK_TIMEOUT_S) -> int:
    """Attach run-summary scores to the root run. Returns the number sent; never raises."""
    if not _config.enabled or not summary:
        return 0
    try:
        return await asyncio.wait_for(asyncio.to_thread(_send_feedback, run_id, summary, client), timeout)
    except Exception:  # noqa: BLE001 - feedback is best-effort
        logger.warning("LangSmith feedback for run %s failed", run_id, exc_info=True)
        return 0


def submit_feedback_in_background(run_id: str, summary: Mapping[str, Any]) -> None:
    """Fire-and-forget :func:`submit_feedback` on the running loop (keeps a reference until done)."""
    if not _config.enabled:
        return
    try:
        task = asyncio.get_running_loop().create_task(submit_feedback(run_id, summary))
    except RuntimeError:
        return
    _background.add(task)
    task.add_done_callback(_background.discard)
