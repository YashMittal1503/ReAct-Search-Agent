"""Discovery tier: asynchronous Tavily web search over a multi-key pool."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from tavily import AsyncTavilyClient

from agent.resilience.errors import ErrorKind, classify_error
from agent.resilience.key_pool import KeyPool
from schemas.evidence import SearchResult

logger = logging.getLogger(__name__)

SEARCH_SCOPE = "search"


@dataclass
class QueryOutcome:
    """Raw results (pre-deduplication) for one sub-query."""

    query: str
    results: list[SearchResult] = field(default_factory=list)
    error: str | None = None


def build_tavily_pool(keys: list[str], *, rpm: float | None = 100) -> KeyPool:
    """Key pool for Tavily (``TAVILY_API_KEY``, ``TAVILY_API_KEY_2`` … / ``TAVILY_API_KEYS``)."""
    return KeyPool("tavily", keys, rpm=rpm, cooldown_base=10, cooldown_max=120)


class TavilySearcher:
    """Async Tavily wrapper with multi-key rotation.

    Returns titles, URLs, short snippets and relevance scores only (no raw page content):
    full-page reading is a separate, *selective* step handled by the scraper.

    Key handling mirrors the LLM router: round-robin across keys, proactive RPM budgets, 429 →
    cool the key and rotate, plan-limit (432/433) → park the key for an hour, invalid key →
    disable it, transient error → short cooldown and retry on another key.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        pool: KeyPool | None = None,
        max_results: int = 5,
        search_depth: str = "basic",
        retries: int = 2,
        max_wait: float = 20.0,
        client_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.pool = pool or build_tavily_pool([api_key] if api_key else [])
        if not len(self.pool):
            raise ValueError("No Tavily API key configured (TAVILY_API_KEY / TAVILY_API_KEYS)")
        self.client_factory = client_factory or (lambda key: AsyncTavilyClient(api_key=key))
        self.max_results = max_results
        self.search_depth = search_depth
        self.retries = retries
        self.max_wait = max_wait

    def _client(self, slot: Any) -> Any:
        if "tavily" not in slot.clients:
            slot.clients["tavily"] = self.client_factory(slot.key)
        return slot.clients["tavily"]

    async def search(self, query: str) -> QueryOutcome:
        """Run one query, rotating keys on failure; never raises."""
        last_error = "no Tavily key available (all cooling down or disabled)"
        for _ in range(len(self.pool) + self.retries):
            slot, wait = self.pool.pick(SEARCH_SCOPE, max_wait=self.max_wait)
            if slot is None:
                break
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                resp = await self._client(slot).search(
                    query=query,
                    max_results=self.max_results,
                    search_depth=self.search_depth,
                    include_answer=False,
                    include_raw_content=False,
                )
            except Exception as exc:  # noqa: BLE001 - classified below
                err = classify_error(exc)
                last_error = f"{type(exc).__name__}: {exc}"
                cooldown = self.pool.apply_failure(slot, SEARCH_SCOPE, err)
                logger.warning("Tavily key %s failed (%s, cooldown %.0fs): %s", slot.masked, err.kind.value, cooldown, exc)
                if err.kind in (ErrorKind.FATAL, ErrorKind.INCOMPATIBLE, ErrorKind.CONTEXT_TOO_LARGE):
                    break  # a bad request won't succeed on another key
                continue
            slot.record_success()
            results = [
                SearchResult(
                    title=(r.get("title") or r.get("url", "")).strip(),
                    url=r.get("url", ""),
                    snippet=(r.get("content") or "").strip(),
                    score=min(max(float(r.get("score") or 0.0), 0.0), 1.0),
                    query=query,
                )
                for r in resp.get("results", [])
                if r.get("url")
            ]
            return QueryOutcome(query=query, results=results)
        return QueryOutcome(query=query, error=last_error)

    async def search_many(self, queries: list[str]) -> list[QueryOutcome]:
        """Run several sub-queries concurrently (spread across keys by round-robin)."""
        return list(await asyncio.gather(*(self.search(q) for q in queries)))
