"""Tests for the cross-provider ModelRouter and its LangChain facade (fake clients, no network)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import httpx
import openai
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from agent.resilience.providers import ModelTarget, load_provider_configs
from agent.resilience.router import AllProvidersExhaustedError, ModelRouter, RouterEvent, router_events
from tools.search import TavilySearcher, build_tavily_pool


def api_error(status: int, message: str = "error", headers: dict | None = None) -> openai.APIStatusError:
    response = httpx.Response(status, headers=headers or {}, request=httpx.Request("POST", "https://api.test/v1"))
    cls = {429: openai.RateLimitError, 401: openai.AuthenticationError, 404: openai.NotFoundError,
           400: openai.BadRequestError, 500: openai.InternalServerError}.get(status, openai.APIStatusError)
    return cls(message, response=response, body=None)


class FakeClient:
    """Stands in for ChatOpenAI bound to one (provider, key, model)."""

    def __init__(self, script: "Script", provider: str, key: str, model: str) -> None:
        self.script, self.id = script, (provider, key, model)

    def _next(self, kwargs: dict[str, Any]) -> AIMessage:
        self.script.calls.append(self.id)
        self.script.kwargs.append(kwargs)
        queue = self.script.outcomes.get(self.id) or self.script.outcomes.get(self.id[::2])  # (provider, model)
        outcome = queue.pop(0) if queue else None
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            return outcome(kwargs)
        return outcome or AIMessage(content=f"answer from {self.id[0]}:{self.id[2]}",
                                    usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})

    async def ainvoke(self, messages: Any, stop: Any = None, **kwargs: Any) -> AIMessage:
        return self._next(kwargs)

    def invoke(self, messages: Any, stop: Any = None, **kwargs: Any) -> AIMessage:
        return self._next(kwargs)


class Script:
    def __init__(self) -> None:
        self.outcomes: dict[tuple, list] = defaultdict(list)
        self.calls: list[tuple[str, str, str]] = []
        self.kwargs: list[dict[str, Any]] = []

    def factory(self, spec: Any, key: str, model: str) -> FakeClient:
        return FakeClient(self, spec.name, key, model)


def make_router(script: Script, env: dict[str, str] | None = None, chain: list[str] | None = None, **kw: Any) -> ModelRouter:
    env = env or {"GROQ_API_KEYS": "g1,g2", "GEMINI_API_KEY": "ge1", "GROQ_RPM": "0", "GEMINI_RPM": "0"}
    configs = load_provider_configs(env)
    targets = [ModelTarget(*c.split(":", 1)) for c in (chain or ["groq:llama", "gemini:flash"])]
    return ModelRouter(configs, {"main": targets, "fast": targets}, client_factory=script.factory, **kw)


MSGS = [HumanMessage("hello")]


async def test_happy_path_uses_tier_one() -> None:
    script = Script()
    router = make_router(script)
    result = await router.agenerate("main", MSGS)
    assert result.generations[0].message.content == "answer from groq:llama"
    assert result.generations[0].message.response_metadata["router"]["provider"] == "groq"
    assert router.failovers == 0


async def test_rate_limited_key_rotates_to_next_key_same_tier() -> None:
    script = Script()
    router = make_router(script)
    script.outcomes[("groq", "g1", "llama")] = [api_error(429, "Rate limit reached", {"retry-after": "30"})]
    events: list[RouterEvent] = []
    with router_events(events.append):
        result = await router.agenerate("main", MSGS)
    assert script.calls == [("groq", "g1", "llama"), ("groq", "g2", "llama")]
    assert result.generations[0].message.content == "answer from groq:llama"
    g1 = router.pools["groq"].slots[0]
    assert g1.cooldown_remaining("llama") == pytest.approx(30, abs=1)
    assert any("rate-limited" in e.message for e in events)

    # Next call skips the cooling key entirely (no wasted request).
    script.calls.clear()
    await router.agenerate("main", MSGS)
    assert script.calls == [("groq", "g2", "llama")]


async def test_all_keys_limited_fails_over_across_providers() -> None:
    script = Script()
    router = make_router(script)
    script.outcomes[("groq", "llama")] = [api_error(429), api_error(429)]
    events: list[RouterEvent] = []
    with router_events(events.append):
        result = await router.agenerate("main", MSGS)
    assert result.generations[0].message.content == "answer from gemini:flash"
    assert [c[0] for c in script.calls] == ["groq", "groq", "gemini"]
    assert router.failovers == 2
    assert any("Served by gemini:flash" in e.message for e in events)


async def test_round_robin_distributes_consecutive_calls() -> None:
    script = Script()
    router = make_router(script)
    for _ in range(4):
        await router.agenerate("main", MSGS)
    assert [c[1] for c in script.calls] == ["g1", "g2", "g1", "g2"]


async def test_oversized_request_skips_model_without_penalizing_key() -> None:
    script = Script()
    router = make_router(script, chain=["groq:llama", "groq:big", "gemini:flash"])
    script.outcomes[("groq", "llama")] = [api_error(413, "Request too large ... tokens per minute (TPM)")]
    result = await router.agenerate("main", MSGS)
    assert result.generations[0].message.content == "answer from groq:big"  # same key, bigger model
    assert len([c for c in script.calls if c[2] == "llama"]) == 1  # didn't retry llama on key g2
    assert router.pools["groq"].slots[0].cooldown_remaining("llama") == 0


async def test_proactive_tpm_budget_routes_large_prompts_elsewhere() -> None:
    script = Script()
    env = {"GROQ_API_KEY": "g1", "GROQ_TPM": "1000", "GROQ_RPM": "0", "GEMINI_API_KEY": "ge1", "GEMINI_RPM": "0"}
    router = make_router(script, env=env)
    big = [HumanMessage("word " * 4000)]  # ~5k tokens > 1000 TPM capacity
    result = await router.agenerate("main", big)
    assert script.calls == [("gemini", "ge1", "flash")]  # groq never even called
    assert result.generations[0].message.content == "answer from gemini:flash"


async def test_proactive_rpm_budget_avoids_429_by_switching_tier() -> None:
    script = Script()
    env = {"GROQ_API_KEY": "g1", "GROQ_RPM": "6", "GEMINI_API_KEY": "ge1", "GEMINI_RPM": "0"}
    router = make_router(script, env=env)
    await router.agenerate("main", MSGS)  # consumes groq's single burst token
    await router.agenerate("main", MSGS)  # groq budget empty -> goes straight to gemini, no waiting
    assert [c[0] for c in script.calls] == ["groq", "gemini"]
    assert router.failovers == 0  # avoided, not absorbed


async def test_auth_error_disables_key_and_dead_model_is_remembered() -> None:
    script = Script()
    router = make_router(script, chain=["groq:gone", "groq:llama", "gemini:flash"])
    script.outcomes[("groq", "g1", "gone")] = [api_error(404, "model `gone` does not exist")]
    script.outcomes[("groq", "g2", "llama")] = [api_error(401, "Invalid API Key")]
    await router.agenerate("main", MSGS)
    health = router.health()
    assert "groq:gone" in health["dead_models"]
    assert router.pools["groq"].slots[1].status() == "disabled"
    script.calls.clear()
    await router.agenerate("main", MSGS)
    assert all(c[2] != "gone" for c in script.calls)


async def test_waits_for_soonest_key_when_everything_is_saturated() -> None:
    script = Script()
    env = {"GROQ_API_KEY": "g1", "GROQ_RPM": "600"}  # burst 100, refills 10/s
    router = make_router(script, env=env, chain=["groq:llama"], max_wait=5)
    slot = router.pools["groq"].slots[0]
    slot._limits("llama").rpm.tokens = 0  # drain the bucket
    events: list[RouterEvent] = []
    with router_events(events.append):
        await router.agenerate("main", MSGS)
    assert script.calls == [("groq", "g1", "llama")]
    assert any("waiting" in e.message for e in events)


async def test_exhaustion_and_fatal_errors() -> None:
    script = Script()
    router = make_router(script)
    script.outcomes[("groq", "llama")] = [api_error(500), api_error(500)]
    script.outcomes[("gemini", "flash")] = [api_error(429)]
    with pytest.raises(AllProvidersExhaustedError):
        await router.agenerate("main", MSGS)

    script2 = Script()
    router2 = make_router(script2)
    script2.outcomes[("groq", "llama")] = [KeyError("bug")]
    with pytest.raises(KeyError):
        await router2.agenerate("main", MSGS)

    with pytest.raises(AllProvidersExhaustedError, match="No LLM providers configured"):
        await ModelRouter({}).agenerate("main", MSGS)


def test_sync_generate_also_fails_over() -> None:
    script = Script()
    router = make_router(script)
    script.outcomes[("groq", "llama")] = [api_error(429), api_error(429)]
    assert router.generate("main", MSGS).generations[0].message.content == "answer from gemini:flash"


# ---------------------------------------------------------------------------
# LangChain facade: structured output, create_agent and deepagents compatibility
# ---------------------------------------------------------------------------
class Verdict(BaseModel):
    ok: bool
    reason: str


async def test_with_structured_output_through_failover() -> None:
    script = Script()
    router = make_router(script)

    def tool_reply(kwargs: dict) -> AIMessage:
        assert kwargs["tool_choice"] == "required"  # LangChain "any" -> OpenAI "required"
        assert kwargs["tools"][0]["function"]["name"] == "Verdict"
        return AIMessage(content="", tool_calls=[{"name": "Verdict", "args": {"ok": True, "reason": "r"}, "id": "c1"}])

    script.outcomes[("groq", "llama")] = [api_error(429), api_error(429)]
    script.outcomes[("gemini", "flash")] = [tool_reply]
    verdict = await router.chat_model("fast").with_structured_output(Verdict, method="function_calling").ainvoke(MSGS)
    assert verdict == Verdict(ok=True, reason="r")


async def test_create_agent_runs_tool_loop_on_resilient_model() -> None:
    from langchain.agents import create_agent

    @tool
    def lookup(q: str) -> str:
        """Look something up."""
        return f"result for {q}"

    script = Script()
    router = make_router(script)
    script.outcomes[("groq", "g1", "llama")] = [
        api_error(429),
    ]
    script.outcomes[("groq", "g2", "llama")] = [
        AIMessage(content="", tool_calls=[{"name": "lookup", "args": {"q": "x"}, "id": "t1"}]),
        AIMessage(content="final answer"),
    ]
    agent = create_agent(router.chat_model("main"), [lookup])
    out = await agent.ainvoke({"messages": [HumanMessage("go")]})
    assert out["messages"][-1].content == "final answer"
    assert any(m.content == "result for x" for m in out["messages"])


async def test_deepagents_accepts_resilient_model() -> None:
    from deepagents import create_deep_agent

    script = Script()
    router = make_router(script)
    script.outcomes[("groq", "llama")] = [api_error(413, "Request too large")]
    agent = create_deep_agent(model=router.chat_model("main"), tools=[], system_prompt="be brief")
    out = await agent.ainvoke({"messages": [HumanMessage("hi")]})
    assert out["messages"][-1].content == "answer from gemini:flash"
    assert script.kwargs[-1]["tools"]  # deepagents' built-in tools were bound and forwarded


async def test_failover_events_reach_the_langgraph_custom_stream() -> None:
    """Events raised deep inside a node (even in child tasks) surface as TraceEvents for the UI."""
    import asyncio
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    from agent.nodes import with_router_events

    script = Script()
    router = make_router(script)
    script.outcomes[("groq", "g1", "llama")] = [api_error(429)]

    class S(TypedDict, total=False):
        out: str

    async def node(state: S) -> dict:
        # Run the LLM call in a child task, like the inner agent / sub-agents do.
        result = await asyncio.create_task(router.agenerate("main", MSGS))
        return {"out": result.generations[0].message.content}

    g = StateGraph(S)
    g.add_node("n", with_router_events(node))
    g.add_edge(START, "n")
    g.add_edge("n", END)
    traces = [chunk["trace"] async for mode, chunk in g.compile().astream({}, stream_mode=["custom"])]
    assert traces and traces[0]["kind"] == "failover" and traces[0]["node"] == "llm-router"
    assert "rate-limited" in traces[0]["message"]


# ---------------------------------------------------------------------------
# Tavily key rotation
# ---------------------------------------------------------------------------
async def test_tavily_rotates_keys_on_rate_limit() -> None:
    from tavily.errors import UsageLimitExceededError

    calls: list[str] = []

    class FakeTavily:
        def __init__(self, key: str) -> None:
            self.key = key

        async def search(self, **kwargs: Any) -> dict:
            calls.append(self.key)
            if self.key == "t1":
                raise UsageLimitExceededError("rate limit")
            return {"results": [{"title": "T", "url": "https://a.com", "content": "c", "score": 0.9}]}

    searcher = TavilySearcher(pool=build_tavily_pool(["t1", "t2"], rpm=None), client_factory=FakeTavily)
    outcome = await searcher.search("q")
    assert outcome.error is None and outcome.results[0].url == "https://a.com"
    assert calls == ["t1", "t2"]
    assert searcher.pool.slots[0].status().startswith("cooling")
