"""Session-level URL registry: URL normalization, deduplication and scrape bookkeeping.

The registry is the single source of truth for *what the agent has already seen*. It is:

* **Serializable**: :meth:`URLRegistry.snapshot` / :meth:`URLRegistry.from_snapshot` round-trip
  through plain dicts, so the registry lives inside the LangGraph state and gets checkpointed
  with it. That means it survives human-in-the-loop interrupts and is scoped per ``thread_id``
  (i.e. per research session) without any global mutable store.
* **Race-free under asyncio**: all mutations are synchronous (no ``await`` between check and
  update). A URL is *claimed* (``in_flight``) before the scrape coroutine awaits the network, so
  two concurrent ``scrape_page`` calls for the same URL can never both fetch it.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from schemas.evidence import Evidence, ScrapedPage, SearchResult

# Exact query-parameter names that only carry tracking / attribution information.
TRACKING_PARAMS: frozenset[str] = frozenset(
    {
        "gclid", "dclid", "fbclid", "msclkid", "yclid", "twclid", "igshid", "mc_cid", "mc_eid",
        "ref", "ref_src", "ref_url", "referrer", "source", "src", "cmpid", "spm", "_ga", "_gl",
        "si", "share", "oly_anon_id", "oly_enc_id", "vero_id", "wickedid", "trk", "trkcampaign",
    }
)
# Parameter prefixes that indicate tracking (``utm_source``, ``hsa_cam``, ...).
TRACKING_PREFIXES: tuple[str, ...] = ("utm_", "hsa_", "pk_", "mtm_", "itm_", "_hs")

_DEFAULT_PORTS = {"http": "80", "https": "443"}
_WS_RE = re.compile(r"\s+")


def normalize_url(url: str) -> str:
    """Return a canonical form of ``url`` suitable for deduplication.

    Normalization steps:

    1. Trim whitespace; default the scheme to ``https`` for scheme-less URLs.
    2. Lowercase the scheme and host; drop ``www.``, default ports and user-info.
    3. Strip the fragment (``#section`` and ``#:~:text=`` directives).
    4. Remove tracking parameters (``utm_*``, ``gclid``, ``fbclid``, ...) and sort the rest.
    5. Remove a trailing slash from non-root paths.

    >>> normalize_url("HTTPS://www.Example.com/a/?utm_source=x&b=2&a=1#top")
    'https://example.com/a?a=1&b=2'
    """
    url = url.strip()
    if not url:
        return url
    if "://" not in url:
        url = "https://" + url.lstrip("/")

    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    port = parts.port
    netloc = host if port is None or str(port) == _DEFAULT_PORTS.get(scheme) else f"{host}:{port}"

    path = re.sub(r"/{2,}", "/", parts.path or "/")
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    query_pairs = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if k.lower() not in TRACKING_PARAMS and not k.lower().startswith(TRACKING_PREFIXES)
    ]
    query = urlencode(sorted(query_pairs), doseq=True)

    return urlunsplit((scheme, netloc, path, query, ""))


def snippet_fingerprint(snippet: str) -> str:
    """Stable hash of a snippet, insensitive to case, whitespace and punctuation.

    Used to catch syndicated/mirrored content that appears under different URLs.
    """
    text = _WS_RE.sub(" ", re.sub(r"[^\w\s]", "", snippet.lower())).strip()
    return hashlib.sha1(text[:300].encode("utf-8")).hexdigest()


@dataclass
class URLRegistry:
    """Tracks discovered results, seen snippets and scraped pages for one research session.

    Attributes:
        results: Normalized URL -> first :class:`SearchResult` that surfaced it.
        seen_snippets: Fingerprints of every accepted snippet.
        scraped_urls: URLs whose full page has been fetched (successfully or not).
        pages: Normalized URL -> :class:`ScrapedPage` for successful scrapes.
        queries_run: Every search query executed so far (for query-rewrite avoidance).
        in_flight: URLs currently being scraped (transient, not persisted).
    """

    results: dict[str, SearchResult] = field(default_factory=dict)
    seen_snippets: set[str] = field(default_factory=set)
    scraped_urls: set[str] = field(default_factory=set)
    pages: dict[str, ScrapedPage] = field(default_factory=dict)
    queries_run: list[str] = field(default_factory=list)
    in_flight: set[str] = field(default_factory=set)

    # ------------------------------------------------------------------ search tier
    def register_result(self, result: SearchResult) -> tuple[bool, str]:
        """Try to add a search result; return ``(accepted, reason)``.

        A result is rejected if its normalized URL or its snippet fingerprint was already seen.
        The stored result always carries the *normalized* URL.
        """
        url = normalize_url(result.url)
        if not url.startswith(("http://", "https://")):
            return False, "non-http url"
        if url in self.results:
            return False, "duplicate url"
        fp = snippet_fingerprint(result.snippet)
        if result.snippet.strip() and fp in self.seen_snippets:
            return False, "duplicate snippet"

        self.results[url] = result.model_copy(update={"url": url})
        if result.snippet.strip():
            self.seen_snippets.add(fp)
        return True, "new"

    def record_query(self, query: str) -> bool:
        """Record a query; return ``False`` if an equivalent query already ran."""
        key = " ".join(query.lower().split())
        if key in {" ".join(q.lower().split()) for q in self.queries_run}:
            return False
        self.queries_run.append(query)
        return True

    # ------------------------------------------------------------------ scrape tier
    def claim_for_scrape(self, url: str, threshold: float) -> tuple[bool, str]:
        """Decide whether ``url`` may be scraped, and claim it atomically if so.

        Guard rules (the "selective" in selective scraping):

        * the URL must have been discovered by ``search_web`` (blocks hallucinated URLs);
        * it must not already be scraped or currently in flight;
        * its search relevance score must be at least ``threshold``.
        """
        norm = normalize_url(url)
        result = self.results.get(norm)
        if result is None:
            return False, "URL was not returned by search_web; only scrape discovered URLs"
        if norm in self.scraped_urls or norm in self.in_flight:
            return False, "already scraped"
        if result.score < threshold:
            return False, f"low relevance score {result.score:.2f} < {threshold:.2f}"
        self.in_flight.add(norm)
        return True, "ok"

    def mark_scraped(self, url: str, page: ScrapedPage) -> None:
        """Record the outcome of a scrape and release the in-flight claim."""
        norm = normalize_url(url)
        self.in_flight.discard(norm)
        self.scraped_urls.add(norm)
        if page.ok and page.markdown:
            self.pages[norm] = page

    # ------------------------------------------------------------------ views
    def evidence(self) -> list[Evidence]:
        """All evidence collected so far, highest relevance first, scraped pages enriched."""
        items = [
            Evidence(
                url=url,
                title=(self.pages[url].title if url in self.pages and self.pages[url].title else r.title),
                snippet=r.snippet,
                score=r.score,
                query=r.query,
                content=self.pages[url].markdown if url in self.pages else None,
            )
            for url, r in self.results.items()
        ]
        return sorted(items, key=lambda e: (e.content is not None, e.score), reverse=True)

    def stats(self) -> dict[str, int]:
        """Counters for the UI."""
        return {
            "results": len(self.results),
            "scraped": len(self.scraped_urls),
            "pages_ok": len(self.pages),
            "queries": len(self.queries_run),
        }

    # ------------------------------------------------------------------ (de)serialization
    def snapshot(self) -> dict[str, Any]:
        """Serialize to plain JSON-compatible data (for LangGraph state/checkpoints)."""
        return {
            "results": {u: r.model_dump(exclude={"domain"}) for u, r in self.results.items()},
            "seen_snippets": sorted(self.seen_snippets),
            "scraped_urls": sorted(self.scraped_urls),
            "pages": {u: p.model_dump() for u, p in self.pages.items()},
            "queries_run": list(self.queries_run),
        }

    @classmethod
    def from_snapshot(cls, data: dict[str, Any] | None) -> "URLRegistry":
        """Rebuild a registry from :meth:`snapshot` output (``None`` -> empty registry)."""
        if not data:
            return cls()
        return cls(
            results={u: SearchResult.model_validate(r) for u, r in data.get("results", {}).items()},
            seen_snippets=set(data.get("seen_snippets", [])),
            scraped_urls=set(data.get("scraped_urls", [])),
            pages={u: ScrapedPage.model_validate(p) for u, p in data.get("pages", {}).items()},
            queries_run=list(data.get("queries_run", [])),
        )
