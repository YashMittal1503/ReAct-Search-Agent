"""Tests for quote verification and text-fragment deep links."""

from tools.deep_links import build_text_fragment, clean_markdown, verify_quote

SOURCE = (
    "# Microservices in 2026\n\n"
    "Teams adopting **microservices** report faster independent deployments. "
    "However, operational overhead grows with every new service boundary. "
    "Monoliths remain simpler to debug for small teams."
)


def test_short_quote_is_encoded_whole() -> None:
    link = build_text_fragment("https://ex.com/a#old", "faster independent deployments")
    assert link == "https://ex.com/a#:~:text=faster%20independent%20deployments"


def test_long_quote_uses_start_end_range_and_escapes_delimiters() -> None:
    quote = "However, operational overhead grows with every new service-boundary, says the report."
    link = build_text_fragment("https://ex.com/a", quote)
    assert link.startswith("https://ex.com/a#:~:text=However%2C%20operational%20overhead%20grows,")
    directive = link.split("#:~:text=", 1)[1]
    start, end = directive.split(",")
    assert "%2D" in end  # hyphen is a spec delimiter and must be escaped
    assert not end.endswith(".")


def test_verify_exact_match_ignores_markdown_and_case() -> None:
    passage, ok = verify_quote("teams adopting microservices report faster", SOURCE)
    assert ok and passage == "teams adopting microservices report faster"


def test_verify_paraphrase_snaps_to_best_sentence() -> None:
    passage, ok = verify_quote("Operational overhead grows with each new service boundary.", SOURCE)
    assert ok
    assert passage.startswith("However, operational overhead grows")


def test_verify_unrelated_quote_is_flagged() -> None:
    _, ok = verify_quote("Quantum computers will replace all databases by 2027.", SOURCE)
    assert ok is False


def test_clean_markdown() -> None:
    assert clean_markdown("## **Bold** [link](http://x) text") == "Bold link text"
