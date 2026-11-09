"""Graph nodes for the Deep Research workflow.

``assess_clarity -> [ask_user] -> plan -> research -> evaluate -> (plan ...) -> synthesize``

Each node is an ``async`` method that reads the :class:`ResearchState` and returns a partial
state update. External dependencies (LLMs, search API, scraper, inner agent) are injected through
:class:`AgentDeps`, which keeps nodes unit-testable with fakes and no network access.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.types import interrupt
from pydantic import BaseModel

from agent.citations import build_report
from agent.config import Settings
from agent.llm import build_chat_model, build_router, structured_call
from agent.prompts import CLARIFIER_SYSTEM, EVALUATOR_SYSTEM, PLANNER_SYSTEM, SYNTHESIZER_SYSTEM
from agent.researcher import build_research_agent
from agent.resilience.key_pool import KeyPool, parse_keys
from agent.resilience.providers import load_env
from agent.resilience.router import AllProvidersExhaustedError, ModelRouter, RouterEvent, router_events
from agent.state import ResearchState
from schemas.clarification import ClarificationRequest, ClarityAssessment
from schemas.events import TraceEvent
from schemas.evidence import Evidence, EvidenceEvaluation
from schemas.planning import ResearchPlan, SubQuery
from schemas.report import BARE_CITATION_RE, ResearchReport, SynthesisDraft
from tools.langchain_tools import ToolBudget, make_research_tools
from tools.scraper import PageScraper
from tools.search import TavilySearcher, build_tavily_pool
from tools.url_registry import URLRegistry

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

StructuredFn = Callable[[type[T], list[BaseMessage], bool], Awaitable[T]]

#: Errors after which a control node degrades gracefully instead of failing the whole run.
RECOVERABLE_LLM_ERRORS = (ValueError, AllProvidersExhaustedError)


# ---------------------------------------------------------------------------
# Dependency container
# ---------------------------------------------------------------------------
@dataclass
class AgentDeps:
    """Injectable dependencies. Defaults build real clients from :class:`Settings`.

    ``router`` and ``search_pool`` should be *process-wide* singletons (the Streamlit app caches
    them) so key cooldowns and rate budgets persist across requests and graph rebuilds.
    """

    settings: Settings
    router: ModelRouter | None = None
    search_pool: KeyPool | None = None
    llm_factory: Callable[[bool], BaseChatModel] | None = None
    structured: StructuredFn | None = None
    searcher_factory: Callable[[], TavilySearcher] | None = None
    scraper_factory: Callable[[], PageScraper] | None = None
    agent_factory: Callable[..., Any] = field(default=build_research_agent)
    today: Callable[[], str] = field(default=lambda: datetime.now().strftime("%B %d, %Y"))

    def __post_init__(self) -> None:
        s = self.settings
        if self.llm_factory is None:
            router = self.router = self.router or build_router(s)
            cache: dict[bool, BaseChatModel] = {}

            def _llm(fast: bool) -> BaseChatModel:
                if fast not in cache:
                    cache[fast] = build_chat_model(router, fast=fast)
                return cache[fast]

            self.llm_factory = _llm
        if self.structured is None:
            llm_factory = self.llm_factory

            async def _structured(schema: type[T], messages: list[BaseMessage], fast: bool) -> T:
                return await structured_call(llm_factory(fast), schema, messages, method=s.llm_structured_method)

            self.structured = _structured
        if self.searcher_factory is None:
            pool = self.search_pool = self.search_pool or build_tavily_pool(
                parse_keys(load_env(), "TAVILY"), rpm=s.tavily_rpm)
            self.searcher_factory = lambda: TavilySearcher(
                pool=pool, max_results=s.search_results_per_query, search_depth=s.tavily_search_depth
            )
        if self.scraper_factory is None:
            self.scraper_factory = lambda: PageScraper(
                timeout=s.scrape_timeout_seconds, max_chars=s.scrape_max_chars, concurrency=s.scrape_concurrency
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def emit(kind: str, node: str, message: str, **data: Any) -> None:
    """Send a :class:`TraceEvent` to the ``custom`` stream (no-op outside a graph run)."""
    event = TraceEvent(kind=kind, node=node, message=message, data=data)  # type: ignore[arg-type]
    try:
        get_stream_writer()({"trace": event.model_dump()})
    except Exception:  # noqa: BLE001 - not running inside a LangGraph context (e.g. unit test)
        logger.debug("trace: %s", event.message)


def with_router_events(fn: Callable[[ResearchState], Awaitable[dict[str, Any]]]) -> Callable[..., Any]:
    """Wrap a node so LLM-router failovers raised anywhere inside it appear in the UI trace.

    The node captures *its* LangGraph stream writer and installs it as a context-local listener.
    Context variables are inherited by child asyncio tasks, so events from the inner deepagents
    graph, its sub-agents and every tool call reach the outer ``custom`` stream too.
    """

    @functools.wraps(fn)
    async def wrapper(state: ResearchState) -> dict[str, Any]:
        try:
            writer = get_stream_writer()
        except Exception:  # noqa: BLE001 - not inside a graph run (unit tests)
            writer = None

        def listener(ev: RouterEvent) -> None:
            if writer is not None:
                kind = "failover" if ev.kind == "failover" else "error"
                writer({"trace": TraceEvent(kind=kind, node="llm-router", message=ev.message, data=ev.data).model_dump()})

        with router_events(listener):
            return await fn(state)

    return wrapper


def _text(msg: BaseMessage) -> str:
    """Plain text of a message whose content may be a string or a list of content blocks."""
    content = msg.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


def _evidence_digest(evidence: list[Evidence], *, per_item: int, budget: int) -> str:
    """Numbered, size-bounded evidence listing shared by the evaluator and synthesizer."""
    blocks: list[str] = []
    used = 0
    for i, ev in enumerate(evidence, start=1):
        body = (ev.content or ev.snippet)[:per_item]
        kind = "full page" if ev.content else "snippet"
        block = f"[{i}] {ev.title} — {ev.domain} ({kind}, relevance {ev.score:.2f})\n{body}"
        if used + len(block) > budget and blocks:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n---\n\n".join(blocks)


DEFAULT_CLARIFYING_QUESTIONS = [
    "Which specific aspect or question should the research focus on?",
    "Is there a timeframe, region, industry or use-case that matters?",
    "Who is the report for, and how technical should it be?",
]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
class ResearchNodes:
    """Container for all graph nodes and routing functions."""

    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps
        self.s = deps.settings

    async def _structured(self, schema: type[T], messages: list[BaseMessage], *, fast: bool) -> T:
        assert self.deps.structured is not None
        return await self.deps.structured(schema, messages, fast)

    # ------------------------------------------------------------------ 1. Ambiguity gate
    async def assess_clarity(self, state: ResearchState) -> dict[str, Any]:
        """Decide whether the prompt needs clarification before any search is run.

        Combines a cheap length heuristic with an LLM :class:`ClarityAssessment`. The LLM call
        lives here, *not* in :meth:`ask_user`, because LangGraph re-executes the interrupted node
        from the top on resume — keeping ``interrupt()`` in its own node avoids paying for (and
        getting a different answer from) the assessment twice.
        """
        query = state.get("clarified_query") or state["query"]
        rounds = state.get("clarification_rounds", 0)
        if rounds >= self.s.max_clarification_rounds:
            if rounds:
                emit("status", "clarify", "Clarification limit reached — proceeding with stated assumptions")
            return {"pending_clarification": None}

        emit("status", "clarify", "Checking the request for ambiguity…")
        too_short = len(query.split()) < self.s.min_query_words
        try:
            assessment = await self._structured(
                ClarityAssessment,
                [SystemMessage(CLARIFIER_SYSTEM.format(today=self.deps.today())),
                 HumanMessage(f"Research request:\n\"\"\"{query}\"\"\"\n"
                              + ("(Note: the request is very short.)" if too_short else ""))],
                fast=True,
            )
        except RECOVERABLE_LLM_ERRORS as exc:
            emit("error", "clarify", f"Clarity check unavailable ({exc}); using heuristics only")
            assessment = ClarityAssessment(is_ambiguous=too_short, reasoning="Request is very short.")

        if not (assessment.is_ambiguous or too_short):
            emit("thought", "clarify", f"Request is clear: {assessment.reasoning}")
            return {"pending_clarification": None}

        questions = assessment.questions or DEFAULT_CLARIFYING_QUESTIONS
        request = ClarificationRequest(
            original_query=state["query"],
            reasoning=assessment.reasoning or "The request is too short to research precisely.",
            questions=questions[:3],
            round=rounds + 1,
        )
        emit("thought", "clarify", f"Needs clarification: {request.reasoning}", questions=request.questions)
        return {"pending_clarification": request.model_dump()}

    async def ask_user(self, state: ResearchState) -> dict[str, Any]:
        """Pause the graph (human-in-the-loop) and merge the user's reply into the query."""
        request = state["pending_clarification"] or {}
        reply = interrupt(request)  # <- execution suspends here until Command(resume=...)
        answer = (reply.get("answer") if isinstance(reply, dict) else str(reply or "")).strip()
        if not answer:
            answer = "(no additional details — use sensible, clearly stated assumptions)"

        query = state.get("clarified_query") or state["query"]
        qa = "\n".join(f"- {q}" for q in request.get("questions", []))
        clarified = f"{query}\n\nClarifying questions asked:\n{qa}\nUser's answer: {answer}"
        emit("thought", "clarify", f"User clarified: {answer[:200]}")
        return {
            "clarified_query": clarified,
            "clarification_rounds": state.get("clarification_rounds", 0) + 1,
            "pending_clarification": None,
            "clarification_log": [{"questions": request.get("questions", []), "answer": answer}],
        }

    @staticmethod
    def route_after_assess(state: ResearchState) -> Literal["ask_user", "plan"]:
        return "ask_user" if state.get("pending_clarification") else "plan"

    # ------------------------------------------------------------------ 2. Planner
    async def plan(self, state: ResearchState) -> dict[str, Any]:
        """Decompose the question into 2-4 queries; rewrite queries on later iterations."""
        iteration = state.get("iteration", 0) + 1
        query = state.get("clarified_query") or state["query"]
        registry = URLRegistry.from_snapshot(state.get("registry"))
        evaluation = state.get("evaluation") or {}
        emit("status", "plan", f"Iteration {iteration}/{self.s.max_iterations}: planning search queries…")

        prompt = f"Research question:\n\"\"\"{query}\"\"\""
        if iteration > 1:
            prompt += (
                "\n\nThis is a FOLLOW-UP iteration. Earlier queries (do not repeat):\n"
                + "\n".join(f"- {q}" for q in registry.queries_run)
                + "\n\nCoverage so far: " + evaluation.get("coverage_summary", "")
                + "\nGaps to close:\n" + "\n".join(f"- {g}" for g in evaluation.get("gaps", []))
                + "\nConflicts to resolve:\n" + "\n".join(f"- {c}" for c in evaluation.get("conflicting_claims", []))
                + "\nEvaluator suggestions:\n" + "\n".join(f"- {q}" for q in evaluation.get("follow_up_queries", []))
            )
        try:
            plan = await self._structured(
                ResearchPlan,
                [SystemMessage(PLANNER_SYSTEM.format(today=self.deps.today())), HumanMessage(prompt)],
                fast=True,
            )
        except RECOVERABLE_LLM_ERRORS as exc:
            emit("error", "plan", f"Planner failed ({exc}); using fallback queries")
            fallback = evaluation.get("follow_up_queries") or [state["query"]]
            plan = ResearchPlan(objective=state["query"],
                                sub_queries=[SubQuery(query=q, rationale="fallback") for q in fallback[:4]])

        # Query rewriting guard: never re-run a query that already ran in this session.
        already = {" ".join(q.lower().split()) for q in registry.queries_run}
        fresh = [sq for sq in plan.sub_queries if " ".join(sq.query.lower().split()) not in already]
        if not fresh:
            extra = [q for q in evaluation.get("follow_up_queries", []) if " ".join(q.lower().split()) not in already]
            fresh = [SubQuery(query=q, rationale="evaluator follow-up") for q in extra[:3]] or [
                SubQuery(query=f"{plan.objective[:80]} expert analysis {datetime.now().year}", rationale="rewrite")
            ]
            emit("thought", "plan", "Planner repeated earlier queries — rewrote them")
        plan = plan.model_copy(update={"sub_queries": fresh})

        for sq in plan.sub_queries:
            emit("thought", "plan", f"Query: “{sq.query}” — {sq.rationale}")
        return {"plan": plan.model_dump(), "iteration": iteration}

    # ------------------------------------------------------------------ 3. Search + selective scrape
    async def research(self, state: ResearchState) -> dict[str, Any]:
        """Run the inner ReAct agent (deepagents / create_agent) with the guarded tools.

        Falls back to deterministic retrieval (search all planned queries, scrape the top-scoring
        unread URLs) if the agent errors out or never calls a tool — so one flaky model response
        can't stall the whole research run.
        """
        plan = ResearchPlan.model_validate(state["plan"])
        registry = URLRegistry.from_snapshot(state.get("registry"))
        budget = ToolBudget(max_scrapes=self.s.max_scrapes_per_iteration,
                            relevance_threshold=self.s.scrape_relevance_threshold)
        evaluation = state.get("evaluation") or {}
        queries_before = len(registry.queries_run)
        notes = ""

        def _emit(ev: TraceEvent) -> None:
            emit(ev.kind, ev.node, ev.message, **ev.data)

        assert self.deps.searcher_factory and self.deps.scraper_factory
        searcher = self.deps.searcher_factory()
        async with self.deps.scraper_factory() as scraper:
            tools = make_research_tools(registry=registry, searcher=searcher, scraper=scraper,
                                        budget=budget, emit=_emit)
            task = (
                f"Research question:\n\"\"\"{state.get('clarified_query') or state['query']}\"\"\"\n\n"
                f"Objective: {plan.objective}\n\nPlanned search queries:\n"
                + "\n".join(f"- {sq.query}  ({sq.rationale})" for sq in plan.sub_queries)
                + (("\n\nKnown gaps to focus on:\n" + "\n".join(f"- {g}" for g in evaluation.get("gaps", [])))
                   if evaluation.get("gaps") else "")
                + f"\n\nScrape budget this round: {budget.max_scrapes} pages."
            )
            emit("status", "research", f"Researching with the '{self.s.research_engine}' ReAct engine…")
            try:
                assert self.deps.llm_factory is not None
                agent = self.deps.agent_factory(engine=self.s.research_engine, model=self.deps.llm_factory(False),
                                                tools=tools, today=self.deps.today())
                notes = await self._stream_agent(agent, task)
            except Exception as exc:  # noqa: BLE001 - degrade gracefully, never crash the run
                logger.exception("Research agent failed")
                emit("error", "research", f"Research agent error: {type(exc).__name__}: {str(exc)[:300]}")

            if len(registry.queries_run) == queries_before:
                emit("thought", "research", "Agent made no searches — running deterministic fallback retrieval")
                await self._fallback_retrieval(plan, registry, budget, tools)

        stats = registry.stats()
        emit("observation", "research",
             f"Evidence pool: {stats['results']} unique sources, {stats['pages_ok']} full pages read, "
             f"{len(budget.skipped)} scrapes skipped", **stats)
        return {"registry": registry.snapshot(), "research_notes": [notes] if notes else [],
                "skipped_scrapes": budget.skipped}

    async def _stream_agent(self, agent: Any, task: str) -> str:
        """Stream the inner agent, surfacing its Thoughts; return its final notes."""
        final_notes = ""
        config = {"recursion_limit": self.s.research_recursion_limit}
        async for update in agent.astream({"messages": [HumanMessage(task)]}, config=config, stream_mode="updates"):
            if not isinstance(update, dict):
                continue
            for value in update.values():
                messages = value.get("messages") if isinstance(value, dict) else None
                if not isinstance(messages, list):
                    continue
                for msg in messages:
                    if not isinstance(msg, AIMessage):
                        continue
                    text = _text(msg).strip()
                    if text:
                        emit("thought", "researcher", text[:600])
                    for call in msg.tool_calls or []:
                        if call["name"] == "task":
                            emit("thought", "researcher",
                                 f"Delegating to sub-agent `{call['args'].get('subagent_type', '?')}`: "
                                 f"{str(call['args'].get('description', ''))[:200]}")
                        elif call["name"] == "write_todos":
                            todos = call["args"].get("todos", [])
                            emit("thought", "researcher", "Plan: " + "; ".join(
                                str(t.get("content", t)) if isinstance(t, dict) else str(t) for t in todos)[:400])
                    if text and not msg.tool_calls:
                        final_notes = text
        return final_notes

    async def _fallback_retrieval(self, plan: ResearchPlan, registry: URLRegistry,
                                  budget: ToolBudget, tools: list[Any]) -> None:
        """Deterministic search-then-scrape-top-K, reusing the same guarded tools."""
        search_tool, scrape_tool = tools
        await search_tool.ainvoke({"queries": [sq.query for sq in plan.sub_queries]})
        candidates = sorted(
            (r for u, r in registry.results.items() if u not in registry.scraped_urls),
            key=lambda r: r.score, reverse=True,
        )[: budget.max_scrapes - budget.scrapes_used]
        await asyncio.gather(*(
            scrape_tool.ainvoke({"url": r.url, "reason": "fallback: top relevance, snippet too short"})
            for r in candidates
        ))

    # ------------------------------------------------------------------ 4. Reflection
    async def evaluate(self, state: ResearchState) -> dict[str, Any]:
        """Judge sufficiency; identify gaps and conflicts that drive the next iteration."""
        registry = URLRegistry.from_snapshot(state.get("registry"))
        evidence = registry.evidence()
        iteration = state.get("iteration", 1)
        emit("status", "evaluate", "Evaluating evidence coverage…")

        if not evidence:
            evaluation = EvidenceEvaluation(is_sufficient=False, coverage_summary="No evidence gathered yet.",
                                            gaps=["No sources found"], confidence=0.0,
                                            follow_up_queries=[f"{state['query']} overview"])
        else:
            digest = _evidence_digest(evidence, per_item=900, budget=14_000)
            notes = "\n\n".join(state.get("research_notes", [])[-2:])
            try:
                evaluation = await self._structured(
                    EvidenceEvaluation,
                    [SystemMessage(EVALUATOR_SYSTEM),
                     HumanMessage(f"Question:\n\"\"\"{state.get('clarified_query') or state['query']}\"\"\"\n\n"
                                  f"Queries run so far: {registry.queries_run}\n\n"
                                  f"Researcher notes:\n{notes or '(none)'}\n\nEvidence:\n{digest}")],
                    fast=True,
                )
            except RECOVERABLE_LLM_ERRORS as exc:
                emit("error", "evaluate", f"Evaluator failed ({exc}); assuming evidence is sufficient")
                evaluation = EvidenceEvaluation(is_sufficient=True, coverage_summary="Evaluator unavailable.",
                                                confidence=0.4)

        verdict = "sufficient ✅" if evaluation.is_sufficient else "insufficient — will iterate"
        if not evaluation.is_sufficient and iteration >= self.s.max_iterations:
            verdict = f"gaps remain but max_iterations={self.s.max_iterations} reached — synthesizing"
        emit("thought", "evaluate", f"Evidence {verdict}. {evaluation.coverage_summary}",
             gaps=evaluation.gaps, conflicts=evaluation.conflicting_claims, confidence=evaluation.confidence)
        for g in evaluation.gaps[:4]:
            emit("thought", "evaluate", f"Gap: {g}")
        for c in evaluation.conflicting_claims[:4]:
            emit("thought", "evaluate", f"Conflict: {c}")
        dumped = evaluation.model_dump()
        return {"evaluation": dumped, "evaluation_history": [dumped]}

    def route_after_evaluate(self, state: ResearchState) -> Literal["plan", "synthesize"]:
        evaluation = state.get("evaluation") or {}
        if state.get("iteration", 0) >= self.s.max_iterations or evaluation.get("is_sufficient"):
            return "synthesize"
        return "plan"

    # ------------------------------------------------------------------ 5. Synthesis
    async def synthesize(self, state: ResearchState) -> dict[str, Any]:
        """Write the grounded report, then deterministically link and validate citations."""
        registry = URLRegistry.from_snapshot(state.get("registry"))
        evidence = registry.evidence()[: self.s.max_report_sources]
        evaluation = state.get("evaluation") or {}
        question = state.get("clarified_query") or state["query"]
        emit("status", "synthesize", f"Synthesizing report from {len(evidence)} sources…")

        if not evidence:
            report = ResearchReport(
                answer="I couldn't find any usable sources for this request. Try rephrasing it or "
                       "checking the search API key / network connectivity.",
                confidence_score=0.0,
            )
            return {"report": report.model_dump()}

        digest = _evidence_digest(evidence, per_item=2_500, budget=30_000)
        conflicts = evaluation.get("conflicting_claims", [])
        try:
            draft = await self._structured(
                SynthesisDraft,
                [SystemMessage(SYNTHESIZER_SYSTEM.format(today=self.deps.today())),
                 HumanMessage(f"Question:\n\"\"\"{question}\"\"\"\n\n"
                              + (f"Known conflicting claims:\n" + "\n".join(f"- {c}" for c in conflicts) + "\n\n"
                                 if conflicts else "")
                              + f"Numbered evidence:\n\n{digest}")],
                fast=False,
            )
            report = build_report(draft, evidence, evaluator_confidence=evaluation.get("confidence"),
                                  extra_contradictions=conflicts)
        except Exception as exc:  # noqa: BLE001 - last-resort degradation keeps the UI useful
            logger.exception("Synthesis failed")
            emit("error", "synthesize", f"Synthesis failed ({type(exc).__name__}); returning research notes")
            notes = "\n\n".join(state.get("research_notes", [])) or "No notes available."
            report = ResearchReport(
                answer="> ⚠️ The final synthesis step failed; below are the raw research notes.\n\n"
                       + BARE_CITATION_RE.sub("", notes),
                confidence_score=0.1,
                contradictions_noted=conflicts,
            )

        verified = sum(s.quote_verified for s in report.sources)
        emit("observation", "synthesize",
             f"Report ready: {len(report.sources)} cited sources ({verified} quotes verified), "
             f"confidence {report.confidence_score:.2f}")
        return {"report": report.model_dump()}
