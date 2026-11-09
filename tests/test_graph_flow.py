"""End-to-end graph tests with fake LLM / search / scraper / inner agent (no network, no keys)."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

from agent.config import Settings
from agent.graph import build_graph
from agent.nodes import AgentDeps
from agent.runner import ResearchSession, RunResult
from schemas import (
    CitedClaim,
    ClarificationRequest,
    ClarityAssessment,
    EvidenceEvaluation,
    ResearchPlan,
    ScrapedPage,
    SearchResult,
    SubQuery,
    SynthesisDraft,
    TraceEvent,
)
from tools.search import QueryOutcome

QUOTE = "Monoliths remain simpler to operate for small teams with limited platform engineering capacity"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeLLM:
    """Returns canned structured objects keyed by schema; records calls."""

    def __init__(self) -> None:
        self.calls: dict[str, int] = {}

    async def __call__(self, schema: type, messages: list[BaseMessage], fast: bool) -> Any:
        n = self.calls[schema.__name__] = self.calls.get(schema.__name__, 0) + 1
        prompt = str(messages[-1].content)
        if schema is ClarityAssessment:
            request = prompt.split('"""')[1]
            clarified = "User's answer" in request or len(request.split()) > 8
            return ClarityAssessment(is_ambiguous=not clarified, reasoning="too vague",
                                     questions=[] if clarified else ["Which architecture?", "What team size?"])
        if schema is ResearchPlan:
            queries = ["micro vs mono 2026", "monolith ops cost"] if n == 1 else ["micro vs mono 2026", "team size impact"]
            return ResearchPlan(objective="compare", sub_queries=[SubQuery(query=q, rationale="r") for q in queries])
        if schema is EvidenceEvaluation:
            return EvidenceEvaluation(is_sufficient=n >= 2, coverage_summary="ok", gaps=[] if n >= 2 else ["team size"],
                                      conflicting_claims=["A says X, B says Y"], confidence=0.7)
        if schema is SynthesisDraft:
            return SynthesisDraft(
                answer="**TL;DR** Monoliths win for small teams [2]. Microservices scale orgs [1, 2]. Bogus [99].\n\n## Sources\n- junk",
                confidence_score=0.8,
                # Evidence #1 is the scraped shared.com page (scraped pages sort first).
                citations=[CitedClaim(source_id=1, supporting_quote=QUOTE)],
            )
        raise AssertionError(schema)


class FakeSearcher:
    async def search_many(self, queries: list[str]) -> list[QueryOutcome]:
        out = []
        for q in queries:
            slug = re.sub(r"\W+", "-", q)
            out.append(QueryOutcome(query=q, results=[
                SearchResult(title=f"T {q}", url=f"https://site.com/{slug}?utm_source=x", snippet=f"About {q}", score=0.9, query=q),
                # Same article under a tracking URL variant -> must be deduplicated.
                SearchResult(title="dup", url="https://www.shared.com/post#frag", snippet="Shared post", score=0.8, query=q),
                SearchResult(title="weak", url=f"https://weak.com/{slug}", snippet=f"weak {q}", score=0.1, query=q),
            ]))
        return out


class FakeScraper:
    async def __aenter__(self) -> "FakeScraper":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def scrape(self, url: str) -> ScrapedPage:
        body = f"# Page\n\nIntro text about architecture. {QUOTE}. More details follow here."
        return ScrapedPage(url=url, title="Full " + url, markdown=body, chars=len(body))


class FakeAgent:
    """Mimics a compiled ReAct agent: calls the real guarded tools, then yields notes."""

    def __init__(self, tools: list[Any], fail: bool) -> None:
        self.tools, self.fail = tools, fail

    async def astream(self, inputs: dict, config: dict, stream_mode: str):
        if self.fail:
            raise RuntimeError("model exploded")
        task = inputs["messages"][0].content
        queries = re.findall(r"^- (.+?)  \(", task, flags=re.M)
        search, scrape = self.tools
        await search.ainvoke({"queries": queries})
        await scrape.ainvoke({"url": "https://shared.com/post", "reason": "need detail"})
        await scrape.ainvoke({"url": "https://shared.com/post", "reason": "again"})       # -> already scraped
        await scrape.ainvoke({"url": "https://weak.com/x", "reason": "x"})                # -> not discovered
        yield {"model": {"messages": [AIMessage(content="Notes: monoliths simpler (https://shared.com/post)")]}}


def _graph(fail_agent: bool = False, max_iterations: int = 4):
    settings = Settings(max_iterations=max_iterations,
                        max_scrapes_per_iteration=3, _env_file=None)
    deps = AgentDeps(
        settings=settings,
        llm_factory=lambda fast: None,  # type: ignore[arg-type,return-value]
        structured=FakeLLM(),
        searcher_factory=FakeSearcher,  # type: ignore[arg-type]
        scraper_factory=FakeScraper,  # type: ignore[arg-type]
        agent_factory=lambda **kw: FakeAgent(kw["tools"], fail_agent),
        today=lambda: "November 09, 2026",
    )
    return build_graph(deps)


async def _collect(session: ResearchSession, msg: str) -> list[Any]:
    return [item async for item in session.run(msg)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
async def test_full_flow_with_clarification_iteration_and_citations() -> None:
    session = ResearchSession(_graph())

    first = await _collect(session, "microservices?")
    assert isinstance(first[-1], ClarificationRequest)
    assert first[-1].questions == ["Which architecture?", "What team size?"]
    assert await session.is_waiting_for_user()

    second = await _collect(session, "Backend web apps, team of 8 engineers")
    result = second[-1]
    assert isinstance(result, RunResult)
    events = [e for e in second if isinstance(e, TraceEvent)]
    kinds = {e.kind for e in events}
    assert {"dedup", "skip", "action", "observation", "thought"} <= kinds

    # Two iterations: evaluator said insufficient once; repeated query was filtered on iteration 2.
    assert result.iterations == 2
    assert result.queries.count("micro vs mono 2026") == 1
    assert "team size impact" in result.queries

    report = result.report
    assert [s.id for s in report.sources] == [1, 2]  # renumbered by first appearance
    assert "[99]" not in report.answer and "## Sources" not in report.answer
    assert re.search(r"\[\[1\]\]\(https://", report.answer)
    # [2] is cited first -> becomes source 1 (snippet only, no supporting quote -> unverified, plain url).
    unverified, verified = report.sources
    assert unverified.domain == "site.com" and "#:~:text=" not in unverified.deep_link_url
    assert unverified.supporting_quote == "About micro vs mono 2026"  # genuine snippet, not our own claim
    # Old [1] -> source 2: scraped page whose quote is verified -> text-fragment deep link.
    assert verified.domain == "shared.com" and verified.quote_verified
    assert verified.deep_link_url.startswith("https://shared.com/post#:~:text=Monoliths%20remain%20simpler%20to,")
    assert f"[[2]]({verified.deep_link_url})" in report.answer
    assert report.contradictions_noted == ["A says X, B says Y"]
    assert 0.0 <= report.confidence_score <= 1.0


async def test_agent_failure_falls_back_to_deterministic_retrieval() -> None:
    session = ResearchSession(_graph(fail_agent=True, max_iterations=1))
    items = await _collect(session, "Pros and cons of microservices vs monoliths for startups in 2026")
    result = items[-1]
    assert isinstance(result, RunResult)
    assert result.iterations == 1  # max_iterations cap respected
    assert result.stats["pages_ok"] >= 1  # fallback scraped top-scoring pages
    assert any(isinstance(e, TraceEvent) and e.kind == "error" for e in items)


def test_real_deepagents_and_react_engines_compile() -> None:
    """The factory must match the installed deepagents / langchain APIs."""
    from langchain_openai import ChatOpenAI

    from agent.researcher import build_research_agent
    from tools.langchain_tools import ToolBudget, make_research_tools
    from tools.url_registry import URLRegistry

    tools = make_research_tools(registry=URLRegistry(), searcher=FakeSearcher(), scraper=FakeScraper(),  # type: ignore[arg-type]
                                budget=ToolBudget(), emit=lambda e: None)
    model = ChatOpenAI(model="m", api_key="x", base_url="http://localhost:1/v1")
    for engine in ("deep", "react"):
        agent = build_research_agent(engine=engine, model=model, tools=tools, today="today")
        assert hasattr(agent, "astream")
