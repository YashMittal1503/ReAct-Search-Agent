"""Tests for URL normalization, deduplication and scrape guards."""

import pytest

from schemas.evidence import ScrapedPage, SearchResult
from tools.url_registry import URLRegistry, normalize_url


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://www.Example.com/a/?utm_source=x&utm_medium=y#top", "https://example.com/a"),
        ("http://example.com:80/path/", "http://example.com/path"),
        ("https://example.com/p?b=2&a=1&gclid=abc&fbclid=z", "https://example.com/p?a=1&b=2"),
        ("example.com/article", "https://example.com/article"),
        ("https://example.com/x#:~:text=hello", "https://example.com/x"),
        ("https://example.com", "https://example.com/"),
        ("https://example.com//a//b", "https://example.com/a/b"),
    ],
)
def test_normalize_url(raw: str, expected: str) -> None:
    assert normalize_url(raw) == expected


def _r(url: str, snippet: str = "s", score: float = 0.9) -> SearchResult:
    return SearchResult(title="t", url=url, snippet=snippet, score=score, query="q")


def test_duplicate_urls_and_snippets_are_filtered() -> None:
    reg = URLRegistry()
    assert reg.register_result(_r("https://a.com/x?utm_campaign=1", "alpha beta")) == (True, "new")
    assert reg.register_result(_r("https://www.a.com/x", "different")) == (False, "duplicate url")
    # Same snippet text (modulo case / punctuation) under another URL -> mirrored content.
    assert reg.register_result(_r("https://mirror.com/y", "Alpha, BETA!")) == (False, "duplicate snippet")
    assert list(reg.results) == ["https://a.com/x"]


def test_scrape_guard_rules() -> None:
    reg = URLRegistry()
    reg.register_result(_r("https://good.com/a", "g", 0.9))
    reg.register_result(_r("https://weak.com/a", "w", 0.2))

    assert reg.claim_for_scrape("https://unknown.com", 0.5)[0] is False
    assert reg.claim_for_scrape("https://weak.com/a", 0.5)[0] is False
    assert reg.claim_for_scrape("https://good.com/a?utm_x=1", 0.5) == (True, "ok")
    # Second concurrent claim is refused while in flight, and after completion.
    assert reg.claim_for_scrape("https://good.com/a", 0.5) == (False, "already scraped")
    reg.mark_scraped("https://good.com/a", ScrapedPage(url="https://good.com/a", markdown="body", chars=4))
    assert reg.claim_for_scrape("https://good.com/a", 0.5) == (False, "already scraped")


def test_snapshot_roundtrip_and_evidence() -> None:
    reg = URLRegistry()
    reg.register_result(_r("https://a.com/1", "one", 0.4))
    reg.register_result(_r("https://b.com/2", "two", 0.8))
    reg.record_query("q1")
    reg.claim_for_scrape("https://b.com/2", 0.1)
    reg.mark_scraped("https://b.com/2", ScrapedPage(url="https://b.com/2", title="B", markdown="full", chars=4))

    clone = URLRegistry.from_snapshot(reg.snapshot())
    assert clone.snapshot() == reg.snapshot()
    ev = clone.evidence()
    assert ev[0].url == "https://b.com/2" and ev[0].content == "full" and ev[0].title == "B"
    assert clone.record_query("Q1 ") is False
