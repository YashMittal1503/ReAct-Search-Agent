"""Reading tier: asynchronous full-page fetch (httpx) + clean Markdown extraction (trafilatura)."""

from __future__ import annotations

import asyncio
import logging
from types import TracebackType

import httpx
import trafilatura

from schemas.evidence import ScrapedPage

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36 DeepResearchAgent/1.0"
    ),
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5",
    "Accept-Language": "en-US,en;q=0.8",
}
HTML_TYPES = ("text/html", "application/xhtml+xml", "text/plain")
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class PageScraper:
    """Concurrency-limited async page scraper.

    Use as an async context manager so the underlying ``httpx.AsyncClient`` (and its connection
    pool) is shared across all scrapes of a research iteration and closed afterwards::

        async with PageScraper() as scraper:
            page = await scraper.scrape("https://example.com/article")

    Every failure mode (timeout, HTTP error, non-HTML content, oversize body, empty extraction)
    is returned as a ``ScrapedPage(ok=False, error=...)`` instead of raising, so a single bad URL
    can never crash the ReAct loop.
    """

    def __init__(
        self,
        *,
        timeout: float = 15.0,
        max_chars: int = 12_000,
        max_bytes: int = 3_000_000,
        concurrency: int = 4,
        retries: int = 1,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.timeout = timeout
        self.max_chars = max_chars
        self.max_bytes = max_bytes
        self.retries = retries
        self._sem = asyncio.Semaphore(concurrency)
        self._client = client
        self._owns_client = client is None

    async def __aenter__(self) -> "PageScraper":
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers=DEFAULT_HEADERS,
                timeout=httpx.Timeout(self.timeout, connect=min(5.0, self.timeout)),
                follow_redirects=True,
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------ public API
    async def scrape(self, url: str) -> ScrapedPage:
        """Fetch ``url`` and return clean Markdown (truncated to ``max_chars``)."""
        if self._client is None:
            raise RuntimeError("PageScraper must be used as an async context manager")
        async with self._sem:
            html_or_error = await self._fetch(url)
        if isinstance(html_or_error, ScrapedPage):
            return html_or_error
        return await asyncio.to_thread(self._extract, url, html_or_error)

    # ------------------------------------------------------------------ internals
    async def _fetch(self, url: str) -> str | ScrapedPage:
        """Download the body with retries and a streaming size cap."""
        assert self._client is not None
        delay = 0.8
        last_error = "unknown error"
        for attempt in range(self.retries + 1):
            try:
                async with self._client.stream("GET", url) as resp:
                    if resp.status_code in RETRYABLE_STATUS and attempt < self.retries:
                        last_error = f"HTTP {resp.status_code}"
                        await asyncio.sleep(delay)
                        delay *= 2
                        continue
                    if resp.status_code >= 400:
                        return _fail(url, f"HTTP {resp.status_code}")
                    ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                    if ctype and not ctype.startswith(HTML_TYPES):
                        return _fail(url, f"unsupported content-type '{ctype}'")

                    chunks: list[bytes] = []
                    size = 0
                    async for chunk in resp.aiter_bytes():
                        size += len(chunk)
                        if size > self.max_bytes:
                            return _fail(url, f"page larger than {self.max_bytes // 1_000_000} MB")
                        chunks.append(chunk)
                    encoding = resp.encoding or "utf-8"
                    return b"".join(chunks).decode(encoding, errors="replace")
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.info("Fetch failed for %s (attempt %d): %s", url, attempt + 1, last_error)
                if attempt < self.retries:
                    await asyncio.sleep(delay)
                    delay *= 2
        return _fail(url, last_error)

    def _extract(self, url: str, html: str) -> ScrapedPage:
        """CPU-bound boilerplate removal; runs in a worker thread via ``asyncio.to_thread``."""
        try:
            markdown = trafilatura.extract(
                html,
                url=url,
                output_format="markdown",
                include_comments=False,
                include_tables=True,
                include_links=False,
                include_images=False,
                favor_precision=True,
            )
            if not markdown or len(markdown.strip()) < 200:
                # Fall back to recall-oriented extraction before giving up.
                markdown = trafilatura.extract(html, url=url, output_format="markdown", favor_recall=True)
            if not markdown or not markdown.strip():
                return _fail(url, "no extractable main content")

            meta = trafilatura.extract_metadata(html, default_url=url)
            title = (meta.title if meta and meta.title else "").strip()
            text = markdown.strip()
            if len(text) > self.max_chars:
                text = text[: self.max_chars].rsplit(" ", 1)[0] + " …"
            return ScrapedPage(url=url, title=title, markdown=text, ok=True, chars=len(text))
        except Exception as exc:  # noqa: BLE001 - trafilatura can raise lxml errors on bad markup
            return _fail(url, f"extraction error: {exc}")


def _fail(url: str, error: str) -> ScrapedPage:
    """Build a failed :class:`ScrapedPage`."""
    return ScrapedPage(url=url, ok=False, error=error)
