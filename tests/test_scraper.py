"""Scraper tests using ``httpx.MockTransport`` (no network)."""

import httpx

from tools.scraper import PageScraper

ARTICLE = "<p>" + " ".join(["Microservices allow independent scaling of components."] * 30) + "</p>"
HTML = f"""<html><head><title>Micro vs Mono</title></head>
<body><nav>Home | About</nav><article><h1>Micro vs Mono</h1>{ARTICLE}</article>
<footer>Copyright</footer></body></html>"""


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/ok":
        return httpx.Response(200, html=HTML)
    if path == "/pdf":
        return httpx.Response(200, content=b"%PDF", headers={"content-type": "application/pdf"})
    if path == "/missing":
        return httpx.Response(404)
    if path == "/flaky":
        _handler.calls = getattr(_handler, "calls", 0) + 1  # type: ignore[attr-defined]
        return httpx.Response(503) if _handler.calls == 1 else httpx.Response(200, html=HTML)  # type: ignore[attr-defined]
    raise httpx.ConnectError("boom", request=request)


def _scraper(**kw) -> PageScraper:
    client = httpx.AsyncClient(transport=httpx.MockTransport(_handler), follow_redirects=True)
    return PageScraper(client=client, **kw)


async def test_extracts_markdown_and_title() -> None:
    async with _scraper() as s:
        page = await s.scrape("https://site.test/ok")
    assert page.ok, page.error
    assert "independent scaling" in page.markdown
    assert "Copyright" not in page.markdown
    assert page.title == "Micro vs Mono"


async def test_failures_are_returned_not_raised() -> None:
    async with _scraper(retries=0) as s:
        pdf = await s.scrape("https://site.test/pdf")
        missing = await s.scrape("https://site.test/missing")
        down = await s.scrape("https://site.test/down")
    assert not pdf.ok and "content-type" in (pdf.error or "")
    assert not missing.ok and missing.error == "HTTP 404"
    assert not down.ok and "ConnectError" in (down.error or "")


async def test_retries_transient_status_and_truncates() -> None:
    async with _scraper(retries=1, max_chars=300) as s:
        page = await s.scrape("https://site.test/flaky")
    assert page.ok
    assert page.chars <= 302 and page.markdown.endswith("…")
