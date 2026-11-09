"""Context-size controls of the inner research agent (offline, scripted fake model)."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

from agent.researcher import ALLOWED_TOOLS, CLEARED_PLACEHOLDER, KEEP_RECENT_TOOL_RESULTS, build_research_agent


class ScriptedModel(BaseChatModel):
    """Returns scripted replies and records every request it receives."""

    replies: list[AIMessage]
    requests: list[dict[str, Any]] = []
    bound_tools: list[dict[str, Any]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        self.bound_tools = [convert_to_openai_tool(t) for t in tools]
        return self

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any) -> ChatResult:
        self.requests.append({"messages": list(messages), "tools": list(self.bound_tools)})
        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])


BIG = "result text " * 700  # ~2k tokens per tool result


@tool
def search_web(queries: list[str]) -> str:
    """Search the web."""
    return BIG


def _calls(n: int) -> list[AIMessage]:
    return [AIMessage(content="", tool_calls=[{"name": "search_web", "args": {"queries": [f"q{i}"]}, "id": f"c{i}"}])
            for i in range(n)] + [AIMessage(content="final notes")]


@pytest.mark.parametrize("engine", ["react", "deep"])
async def test_old_tool_results_are_cleared_recent_kept(engine: str) -> None:
    model = ScriptedModel(replies=_calls(4))
    agent = build_research_agent(engine=engine, model=model, tools=[search_web], today="today")
    out = await agent.ainvoke({"messages": [HumanMessage("research x")]})
    assert out["messages"][-1].content == "final notes"

    last_request = model.requests[-1]["messages"]
    tool_msgs = [m for m in last_request if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 4
    cleared = [m for m in tool_msgs if m.content == CLEARED_PLACEHOLDER]
    kept = [m for m in tool_msgs if m.content == BIG]
    assert len(kept) == KEEP_RECENT_TOOL_RESULTS and tool_msgs[-1].content == BIG  # most recent kept verbatim
    assert len(cleared) == 4 - KEEP_RECENT_TOOL_RESULTS


async def test_deep_engine_exposes_only_allowlisted_tools() -> None:
    model = ScriptedModel(replies=[AIMessage(content="done")])
    agent = build_research_agent(engine="deep", model=model, tools=[search_web], today="today")
    await agent.ainvoke({"messages": [HumanMessage("research x")]})
    names = {t["function"]["name"] for t in model.requests[0]["tools"]}
    assert "search_web" in names and "task" in names
    assert names <= ALLOWED_TOOLS
    assert not names & {"ls", "read_file", "write_file", "edit_file", "glob", "grep"}


async def test_observations_are_trimmed_but_full_text_is_stored() -> None:
    from schemas.evidence import ScrapedPage, SearchResult
    from tools.langchain_tools import OBSERVATION_PREVIEW_CHARS, ToolBudget, make_research_tools
    from tools.url_registry import URLRegistry

    page = "word " * 2_000

    class Scraper:
        async def scrape(self, url: str) -> ScrapedPage:
            return ScrapedPage(url=url, title="T", markdown=page, chars=len(page))

    class Searcher:
        async def search_many(self, queries: list[str]) -> list:
            from tools.search import QueryOutcome

            return [QueryOutcome(query=queries[0], results=[
                SearchResult(title="T", url="https://a.com/x", snippet="s" * 1_000, score=0.9, query=queries[0])])]

    registry = URLRegistry()
    search, scrape = make_research_tools(registry=registry, searcher=Searcher(), scraper=Scraper(),  # type: ignore[arg-type]
                                         budget=ToolBudget(), emit=lambda e: None)
    obs = await search.ainvoke({"queries": ["q"]})
    assert "s" * 200 in obs and "s" * 201 not in obs  # snippet trimmed to 200 chars
    preview = await scrape.ainvoke({"url": "https://a.com/x", "reason": "need detail"})
    assert OBSERVATION_PREVIEW_CHARS == 1_200 and page[:1_200] in preview and page[:1_300] not in preview
    assert registry.pages["https://a.com/x"].markdown == page  # full text kept for synthesis
