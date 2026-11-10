"""LangSmith tracing integration (#16). Offline: no key -> no network; LangSmith client is faked."""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any

import pytest

from agent.observability import tracing
from agent.runner import ResearchSession, RunResult
from tests.test_graph_flow import _collect, _graph
from tests.test_router import MSGS, Script, api_error, make_router


@pytest.fixture
def fake_environ(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Isolate os.environ writes and reset the module config afterwards."""
    env: dict[str, str] = {}
    monkeypatch.setattr(tracing.os, "environ", env)
    monkeypatch.setattr(tracing, "_project_ids", {})
    yield env
    tracing._config = tracing.TracingConfig(enabled=False)


def test_tracing_disabled_without_key(fake_environ: dict[str, str]) -> None:
    cfg = tracing.configure_tracing({"LANGSMITH_TRACING": "true"})  # flag without a key
    assert not cfg.enabled and fake_environ["LANGSMITH_TRACING"] == "false"
    assert "LANGSMITH_API_KEY" not in fake_environ


def test_tracing_enabled_exports_env(fake_environ: dict[str, str]) -> None:
    cfg = tracing.configure_tracing({"LANGSMITH_API_KEY": "lsv2_x", "LANGSMITH_PROJECT": "demo"})
    assert cfg.enabled and cfg.project == "demo"
    exported = {k: v for k, v in fake_environ.items() if k.startswith("LANGSMITH_")}
    assert exported == {"LANGSMITH_API_KEY": "lsv2_x", "LANGSMITH_TRACING": "true", "LANGSMITH_PROJECT": "demo"}
    off = tracing.configure_tracing({"LANGSMITH_API_KEY": "lsv2_x", "LANGSMITH_TRACING": "false"})
    assert not off.enabled
    legacy = tracing.configure_tracing({"LANGCHAIN_API_KEY": "k", "LANGCHAIN_TRACING_V2": "true"})
    assert legacy.enabled and legacy.project == tracing.DEFAULT_PROJECT


def test_run_config_shape() -> None:
    run_id = str(uuid.uuid4())
    cfg = tracing.run_config(run_id, thread_id="t1", query="q" * 600,
                             metadata={"engine": "deep", "resume": True, "chains": {"main": ["x"]}})
    assert cfg["run_id"] == uuid.UUID(run_id) and cfg["run_name"] == "deep-research"
    assert cfg["tags"] == ["engine:deep", "clarification-reply"]
    assert cfg["metadata"]["thread_id"] == "t1" and len(cfg["metadata"]["query"]) == 500
    assert "chains" not in cfg["metadata"]  # only scalar metadata is forwarded


async def test_router_names_each_attempt_as_child_run() -> None:
    script = Script()
    router = make_router(script)
    script.outcomes[("groq", "llama")] = [api_error(429), api_error(429)]
    await router.agenerate("main", MSGS)
    configs = [kw["config"] for kw in script.kwargs]
    assert [c["run_name"] for c in configs] == ["groq:llama", "groq:llama", "gemini:flash"]
    assert [c["metadata"]["attempt"] for c in configs] == [1, 2, 3]
    assert configs[-1]["metadata"]["key"] == "***" and "provider:gemini" in configs[-1]["tags"]


class FakeLangSmith:
    def __init__(self, fail: bool = False) -> None:
        self.feedback: list[tuple[str, str, float]] = []
        self.project_reads = 0
        self.fail = fail

    def create_feedback(self, run_id: str, key: str, *, score: float, comment: str = "") -> None:
        if self.fail:
            raise RuntimeError("langsmith down")
        self.feedback.append((run_id, key, score))

    def read_project(self, *, project_name: str) -> Any:
        self.project_reads += 1
        return SimpleNamespace(id="proj-1")

    def get_run_url(self, *, run: Any) -> str:
        return f"https://smith.example/projects/p/{run.session_id}/r/{run.id}?poll=true"


async def test_feedback_sent_only_when_enabled_and_never_raises(fake_environ: dict[str, str]) -> None:
    summary = {"confidence": 0.8, "verified_ratio": 0.5, "cited_sources": 4, "input_tokens": 100,
               "output_tokens": 50, "status": "completed", "flag": True}
    client = FakeLangSmith()
    assert await tracing.submit_feedback("r1", summary, client=client) == 0  # disabled -> nothing sent

    tracing.configure_tracing({"LANGSMITH_API_KEY": "k"})
    assert await tracing.submit_feedback("r1", summary, client=client) == 4
    assert {k: v for _, k, v in client.feedback} == {"confidence": 0.8, "verified_ratio": 0.5,
                                                     "cited_sources": 4, "total_tokens": 150}
    assert await tracing.submit_feedback("r1", summary, client=FakeLangSmith(fail=True)) == 0


def test_trace_url_cached_project_lookup(fake_environ: dict[str, str]) -> None:
    client = FakeLangSmith()
    assert tracing.trace_url("abc", client=client) is None  # disabled
    tracing.configure_tracing({"LANGSMITH_API_KEY": "k"})
    assert tracing.trace_url("abc", client=client) == "https://smith.example/projects/p/proj-1/r/abc?poll=true"
    tracing.trace_url("def", client=client)
    assert client.project_reads == 1


async def test_run_tree_seen_by_tracers() -> None:
    """The same callback mechanism LangSmith uses records the root run and per-attempt child runs."""
    from langchain_core.tracers.context import collect_runs

    # Router attempts appear as children of the ResilientChatModel run; failed ones carry the error.
    # Real (Runnable) chat models are needed here so the attempts participate in callbacks.
    from langchain_core.language_models import BaseChatModel
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    from agent.resilience.providers import ModelTarget, load_provider_configs
    from agent.resilience.router import ModelRouter

    class RateLimited(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "rate-limited"

        def _generate(self, *a: Any, **kw: Any) -> Any:
            raise api_error(429, "Rate limit reached")

    def factory(spec: Any, key: str, model: str) -> BaseChatModel:
        return RateLimited() if key == "g1" else GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))

    configs = load_provider_configs({"GROQ_API_KEYS": "g1,g2", "GROQ_RPM": "0"})
    router = ModelRouter(configs, {"main": [ModelTarget("groq", "llama")]}, client_factory=factory)
    with collect_runs() as cb:
        await router.chat_model("main").ainvoke(MSGS)
    (parent,) = cb.traced_runs
    assert parent.name == "ResilientChatModel"
    children = sorted(parent.child_runs, key=lambda r: r.dotted_order)
    assert [c.name for c in children] == ["groq:llama", "groq:llama"]
    assert children[0].error and "RateLimitError" in children[0].error and children[1].error is None
    assert children[1].extra["metadata"]["attempt"] == 2

    # The graph's root run uses the local run id, our run name, tags and thread metadata, and every
    # graph node is parented to it. LangSmith rebuilds the tree from exactly these parent links.
    from langchain_core.callbacks import BaseCallbackHandler

    starts: list[dict[str, Any]] = []

    class Recorder(BaseCallbackHandler):
        def on_chain_start(self, serialized: Any, inputs: Any, *, run_id: uuid.UUID,
                           parent_run_id: uuid.UUID | None = None, **kw: Any) -> None:
            starts.append({"name": kw.get("name"), "id": run_id, "parent": parent_run_id,
                           "tags": kw.get("tags") or [], "metadata": kw.get("metadata") or {}})

    session = ResearchSession(_graph(max_iterations=1), metadata={"engine": "deep"})
    session.config["callbacks"] = [Recorder()]  # inheritable, like the LangSmith tracer
    items = await _collect(session, "Pros and cons of microservices vs monoliths for startups in 2026")
    root = next(s for s in starts if s["name"] == "deep-research")
    assert root["id"] == uuid.UUID(items[-1].run_id) and root["parent"] is None
    assert root["metadata"]["thread_id"] == session.thread_id and "engine:deep" in root["tags"]
    nodes = {s["name"]: s for s in starts if s["parent"] == root["id"]}
    assert set(nodes) >= {"assess_clarity", "plan", "research", "evaluate", "synthesize"}
    assert len({s["id"] for s in starts}) == len(starts)  # every run has its own id


async def test_session_passes_root_run_config_to_graph() -> None:
    graph = _graph(max_iterations=1)
    seen: dict[str, Any] = {}

    class Spy:
        def __getattr__(self, name: str) -> Any:
            return getattr(graph, name)

        def astream(self, graph_input: Any, config: dict[str, Any], **kw: Any) -> Any:
            seen.update(config)
            return graph.astream(graph_input, config, **kw)

    session = ResearchSession(Spy(), metadata={"engine": "react"})  # type: ignore[arg-type]
    items = await _collect(session, "Pros and cons of microservices vs monoliths for startups in 2026")
    result = items[-1]
    assert isinstance(result, RunResult) and result.trace_url == ""  # tracing off in tests
    assert seen["run_id"] == uuid.UUID(result.run_id) and seen["run_name"] == "deep-research"
    assert seen["metadata"]["thread_id"] == session.thread_id and seen["tags"] == ["engine:react"]
    assert seen["configurable"]["thread_id"] == session.thread_id  # checkpointer config preserved
