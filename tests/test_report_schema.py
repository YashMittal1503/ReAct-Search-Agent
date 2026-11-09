"""Tests for the strict citation contract enforced by ``ResearchReport``."""

import pytest
from pydantic import ValidationError

from schemas.report import ResearchReport, Source


def _src(i: int = 1) -> Source:
    return Source(
        id=i,
        title="T",
        url=f"https://ex.com/{i}",
        deep_link_url=f"https://ex.com/{i}#:~:text=hello",
        supporting_quote="hello",
    )


def test_valid_report() -> None:
    r = ResearchReport(
        answer="Claim [[1]](https://ex.com/1#:~:text=hello) and [[2]](https://ex.com/2).",
        confidence_score=0.8,
        sources=[_src(1), _src(2)],
    )
    assert r.cited_ids == [1, 2]
    assert r.sources[0].domain == "ex.com"


@pytest.mark.parametrize(
    "answer",
    [
        "Bare citation [1].",                          # not hyperlinked
        "Unknown [[9]](https://ex.com/9).",            # no such source
        "Wrong url [[1]](https://evil.com/1).",        # url mismatch
    ],
)
def test_invalid_citations_rejected(answer: str) -> None:
    with pytest.raises(ValidationError):
        ResearchReport(answer=answer, confidence_score=0.5, sources=[_src(1)])


def test_confidence_bounds_and_duplicate_ids() -> None:
    with pytest.raises(ValidationError):
        ResearchReport(answer="x", confidence_score=1.5)
    with pytest.raises(ValidationError):
        ResearchReport(answer="x", confidence_score=0.5, sources=[_src(1), _src(1)])


def test_deep_link_must_extend_url() -> None:
    with pytest.raises(ValidationError):
        Source(id=1, title="t", url="https://a.com/x", deep_link_url="https://b.com/x", supporting_quote="q")
