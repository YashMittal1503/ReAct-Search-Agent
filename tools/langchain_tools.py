"""LangChain tool adapters exposing the two-tier retriever to the ReAct agent.

Tools are created by a factory that closes over the session's :class:`URLRegistry`, the shared
:class:`TavilySearcher` / :class:`PageScraper` instances and a trace-event emitter. This gives the
LLM a minimal, safe surface (two tools) while all deduplication and guard logic stays in
deterministic Python.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated

from langchain_core.tools import BaseTool, tool

from schemas.events import TraceEvent
from schemas.evidence import ScrapedPage
from tools.scraper import PageScraper
from tools.search import TavilySearcher
from tools.url_registry import URLRegistry, normalize_url

Emitter = Callable[[TraceEvent], None]

#: Max characters of a scraped page returned to the LLM as the observation. The full text is
#: kept in the registry for the evaluator and synthesizer; this keeps the ReAct context small.
OBSERVATION_PREVIEW_CHARS = 2_500


@dataclass
class ToolBudget:
    """Per-iteration limits enforced by the tools themselves (not by the prompt)."""

    max_scrapes: int = 4
    relevance_threshold: float = 0.5
    scrapes_used: int = 0
    skipped: list[dict[str, str]] = field(default_factory=list)


def make_research_tools(
    *,
    registry: URLRegistry,
    searcher: TavilySearcher,
    scraper: PageScraper,
    budget: ToolBudget,
    emit: Emitter,
) -> list[BaseTool]:
    """Build the ``search_web`` and ``scrape_page`` tools bound to one research iteration."""

    @tool("search_web")
    async def search_web(
        queries: Annotated[list[str], "1-4 concise web search queries to run in parallel."],
    ) -> str:
        """Discover sources on the web. Returns titles, URLs, relevance scores and SHORT snippets.

        Duplicate URLs and duplicate snippets already seen in this session are filtered out
        automatically. Use the snippets to decide which pages deserve a full read.
        """
        fresh = [q for q in queries[:4] if q.strip() and registry.record_query(q)]
        repeated = [q for q in queries[:4] if q not in fresh]
        for q in repeated:
            emit(TraceEvent(kind="dedup", node="search_web", message=f"Query already run: “{q}”"))
        if not fresh:
            return "All of these queries were already run. Use the existing results or rephrase."

        emit(TraceEvent(kind="action", node="search_web", message="search_web(" + " | ".join(fresh) + ")",
                        data={"queries": fresh}))
        outcomes = await searcher.search_many(fresh)

        lines: list[str] = []
        new_count = dup_count = 0
        for outcome in outcomes:
            lines.append(f"\n### Query: {outcome.query}")
            if outcome.error:
                emit(TraceEvent(kind="error", node="search_web", message=f"Search failed: {outcome.error}"))
                lines.append(f"(search error: {outcome.error})")
                continue
            for r in outcome.results:
                accepted, reason = registry.register_result(r)
                if not accepted:
                    dup_count += 1
                    emit(TraceEvent(kind="dedup", node="search_web",
                                    message=f"Filtered {reason}: {normalize_url(r.url)}",
                                    data={"url": normalize_url(r.url), "reason": reason}))
                    continue
                new_count += 1
                stored = registry.results[normalize_url(r.url)]
                lines.append(
                    f"- [{stored.score:.2f}] {stored.title}\n  URL: {stored.url}\n  Snippet: {stored.snippet[:350]}"
                )
            if lines[-1].startswith("\n### Query"):
                lines.append("(no new results — all duplicates)")

        emit(TraceEvent(kind="observation", node="search_web",
                        message=f"{new_count} new results, {dup_count} duplicates filtered",
                        data={"new": new_count, "duplicates": dup_count}))
        header = (f"{new_count} new results ({dup_count} duplicates filtered). Scores are relevance "
                  f"in [0,1]; pages below {budget.relevance_threshold:.2f} cannot be scraped. "
                  f"Scrape budget left this round: {budget.max_scrapes - budget.scrapes_used}.")
        return header + "\n" + "\n".join(lines)

    @tool("scrape_page")
    async def scrape_page(
        url: Annotated[str, "A URL previously returned by search_web."],
        reason: Annotated[str, "Why the snippet is relevant but lacks the detail you need."],
    ) -> str:
        """Read the FULL text of a page as clean Markdown.

        Only use when a search snippet is highly relevant but too short to answer the question.
        The tool refuses URLs that were never discovered, were already read, have a low relevance
        score, or when the per-round scrape budget is exhausted.
        """
        norm = normalize_url(url)
        if budget.scrapes_used >= budget.max_scrapes:
            return _skip(budget, emit, norm, "scrape budget for this round exhausted")
        allowed, why = registry.claim_for_scrape(norm, budget.relevance_threshold)
        if not allowed:
            return _skip(budget, emit, norm, why)

        budget.scrapes_used += 1
        emit(TraceEvent(kind="action", node="scrape_page", message=f"scrape_page({norm})",
                        data={"url": norm, "reason": reason}))
        try:
            page = await scraper.scrape(norm)
        except Exception as exc:  # noqa: BLE001 - defensive: never leave a URL claimed
            page =ScrapedPage(url=norm, ok=False, error=f"{type(exc).__name__}: {exc}")
        registry.mark_scraped(norm, page)

        if not page.ok:
            emit(TraceEvent(kind="error", node="scrape_page", message=f"Could not read {norm}: {page.error}",
                            data={"url": norm}))
            return f"FAILED to read {norm}: {page.error}. Rely on its snippet or pick another source."

        emit(TraceEvent(kind="observation", node="scrape_page",
                        message=f"Extracted {page.chars:,} chars from {norm}",
                        data={"url": norm, "chars": page.chars, "title": page.title}))
        preview = page.markdown[:OBSERVATION_PREVIEW_CHARS]
        more = "" if len(page.markdown) <= OBSERVATION_PREVIEW_CHARS else (
            f"\n…[{len(page.markdown) - OBSERVATION_PREVIEW_CHARS:,} more chars stored for synthesis]")
        return f"# {page.title or norm}\nURL: {norm}\n\n{preview}{more}"

    return [search_web, scrape_page]


def _skip(budget: ToolBudget, emit: Emitter, url: str, reason: str) -> str:
    """Record and report a guarded (skipped) scrape."""
    budget.skipped.append({"url": url, "reason": reason})
    emit(TraceEvent(kind="skip", node="scrape_page", message=f"Skipped {url}: {reason}",
                    data={"url": url, "reason": reason}))
    return f"SKIPPED {url}: {reason}."
