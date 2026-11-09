"""Per-provider payload adapters (#4) and sticky routing."""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.resilience.adapters import (
    ADAPTERS,
    GEMINI_PLACEHOLDER_SIGNATURE,
    AdaptedChatOpenAI,
    gemini_adapter,
    mistral_adapter,
    mistral_tool_call_id,
)
from agent.resilience.providers import PROVIDERS, load_provider_configs
from agent.resilience.router import ModelRouter, routing_session
from tests.test_router import MSGS, Script, api_error, make_router

FOREIGN_HISTORY = [
    HumanMessage("research"),
    AIMessage(content="", tool_calls=[{"name": "search_web", "args": {"queries": ["x"]}, "id": "call_8f3kd92jf83kd9fj3k2l"}]),
    ToolMessage(content="results", tool_call_id="call_8f3kd92jf83kd9fj3k2l"),
]


def _payload(adapter, **kw) -> dict:
    llm = AdaptedChatOpenAI(model="m", api_key="k", base_url="http://h/v1", payload_adapter=adapter)
    return llm._get_request_payload(FOREIGN_HISTORY, **kw)


def test_mistral_ids_are_9_alnum_deterministic_and_paired() -> None:
    payload = _payload(mistral_adapter, max_tokens=50)
    call_id = payload["messages"][1]["tool_calls"][0]["id"]
    assert re.fullmatch(r"[A-Za-z0-9]{9}", call_id)
    assert payload["messages"][2]["tool_call_id"] == call_id  # assistant call and tool result stay paired
    assert call_id == mistral_tool_call_id("call_8f3kd92jf83kd9fj3k2l")  # same id on every turn, no state
    assert payload["max_tokens"] == 50 and "max_completion_tokens" not in payload


def test_gemini_adds_placeholder_signature_but_keeps_real_ones() -> None:
    payload = _payload(gemini_adapter, max_tokens=50)
    call = payload["messages"][1]["tool_calls"][0]
    assert call["extra_content"]["google"]["thought_signature"] == GEMINI_PLACEHOLDER_SIGNATURE
    assert payload["max_tokens"] == 50

    real = {"messages": [{"role": "assistant", "tool_calls": [
        {"id": "a", "extra_content": {"google": {"thought_signature": "REAL"}}}]}]}
    assert gemini_adapter(real)["messages"][0]["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "REAL"


def test_unadapted_providers_payload_unchanged() -> None:
    payload = _payload(None, max_tokens=50)
    assert payload["messages"][1]["tool_calls"][0]["id"] == "call_8f3kd92jf83kd9fj3k2l"
    assert "max_completion_tokens" in payload  # OpenAI-native spelling kept for providers that use it


def test_router_default_client_uses_provider_adapter() -> None:
    configs = load_provider_configs({"MISTRAL_API_KEY": "m1", "GROQ_API_KEY": "g1"})
    router = ModelRouter(configs)
    mistral = router._default_client(PROVIDERS["mistral"], "m1", "mistral-small-latest")
    groq = router._default_client(PROVIDERS["groq"], "g1", "llama")
    assert isinstance(mistral, AdaptedChatOpenAI) and mistral.payload_adapter is ADAPTERS["mistral"]
    assert groq.payload_adapter is None


async def test_sticky_routing_keeps_a_session_on_the_last_successful_target() -> None:
    # Groq key g1 has a transient 500 -> groq is skipped for that call -> gemini serves.
    # Next call: without a session, groq (key g2 is healthy) is tried first again;
    # inside a routing session, the call sticks to gemini.
    for sticky in (False, True):
        script = Script()
        router = make_router(script)
        script.outcomes[("groq", "g1", "llama")] = [api_error(500, "upstream error")]

        async def two_calls() -> None:
            await router.agenerate("main", MSGS)
            script.calls.clear()
            await router.agenerate("main", MSGS)

        if sticky:
            with routing_session() as state:
                await two_calls()
            assert state["main"].provider == "gemini"
            assert [c[0] for c in script.calls] == ["gemini"]
        else:
            await two_calls()
            assert [c[0] for c in script.calls] == ["groq"]


async def test_sticky_target_not_ready_falls_back_to_tier_order() -> None:
    script = Script()
    router = make_router(script)
    with routing_session() as state:
        await router.agenerate("main", MSGS)  # served by groq -> sticky = groq
        assert state["main"].provider == "groq"
        script.outcomes[("groq", "llama")] = [api_error(429), api_error(429)]
        script.calls.clear()
        result = await router.agenerate("main", MSGS)
    assert result.generations[0].message.content == "answer from gemini:flash"
