"""Deterministic post-processing that turns an LLM draft into a validated, grounded report.

The LLM only ever writes ``[n]`` markers and quotes. This module:

1. normalizes citation syntax (``[1, 2]`` -> ``[1][2]``; strips any URLs the model added anyway);
2. drops markers that reference non-existent evidence (hallucinated ids);
3. renumbers sources by order of first appearance (1..k);
4. verifies each supporting quote against the retrieved text, snapping paraphrases to the real
   sentence, and builds a ``#:~:text=`` deep link only for verified passages;
5. rewrites every marker to a clickable ``[[n]](deep_link_url)`` hyperlink;
6. constructs :class:`ResearchReport`, whose validators re-check the whole contract.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from schemas.evidence import Evidence
from schemas.report import BARE_CITATION_RE, ResearchReport, Source, SynthesisDraft
from tools.deep_links import build_text_fragment, clean_markdown, verify_quote

_GROUP_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)+)\]")
_LINKED_ANY_RE = re.compile(r"\[\[?(\d+)\]?\]\([^)\s]*\)")
_SOURCES_SECTION_RE = re.compile(
    r"\n#{1,6}\s*\**\s*(sources|references|citations|bibliography|works cited)\b.*\Z", re.I | re.S
)
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+|\n")


def _normalize_markers(answer: str) -> str:
    """Canonicalize every citation into the bare ``[n]`` form."""
    answer = _SOURCES_SECTION_RE.sub("", answer.strip())
    answer = _LINKED_ANY_RE.sub(r"[\1]", answer)
    return _GROUP_RE.sub(lambda m: "".join(f"[{n.strip()}]" for n in re.split(r"[,;]", m.group(1))), answer)


def count_valid_citations(answer: str, n_evidence: int) -> int:
    """Distinct evidence ids (1..n_evidence) cited in ``answer``, after the same normalization as
    :func:`build_report` (grouped ``[1, 2]``, linked forms, trailing sources section stripped).

    >>> count_valid_citations("A [1]. B [1, 3]. C [[2]](http://x). Bogus [9].", 3)
    3
    """
    ids = {int(m.group(1)) for m in BARE_CITATION_RE.finditer(_normalize_markers(answer))}
    return sum(1 for i in ids if 1 <= i <= n_evidence)


def _claim_context(answer: str, source_id: int) -> str:
    """The sentence in ``answer`` that cites ``source_id`` (used when the LLM omitted a quote)."""
    marker = f"[{source_id}]"
    idx = answer.find(marker)
    if idx < 0:
        return ""
    start = max((m.end() for m in _SENTENCE_BOUNDARY_RE.finditer(answer, 0, idx)), default=0)
    return clean_markdown(BARE_CITATION_RE.sub("", answer[start:idx]))


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = " ".join(item.lower().split())
        if item.strip() and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def build_report(
    draft: SynthesisDraft,
    evidence: list[Evidence],
    *,
    evaluator_confidence: float | None = None,
    extra_contradictions: Iterable[str] = (),
) -> ResearchReport:
    """Ground ``draft`` in ``evidence`` (1-indexed as shown to the LLM) and validate it."""
    answer = _normalize_markers(draft.answer)

    # Valid ids in order of first appearance.
    order: list[int] = []
    for m in BARE_CITATION_RE.finditer(answer):
        n = int(m.group(1))
        if 1 <= n <= len(evidence) and n not in order:
            order.append(n)
    remap = {old: new for new, old in enumerate(order, start=1)}
    quotes = {c.source_id: c.supporting_quote for c in draft.citations}

    sources: list[Source] = []
    for old in order:
        ev = evidence[old - 1]
        candidate = quotes.get(old) or _claim_context(answer, old) or ev.snippet
        passage, verified = verify_quote(candidate, ev.text)
        if not verified:
            # Never display text that isn't from the source (e.g. a hallucinated quote or our own
            # claim sentence). Fall back to the genuine search snippet, clearly flagged as unverified.
            passage = clean_markdown(ev.snippet)[:300] or passage
        deep_link = build_text_fragment(ev.url, passage) if verified else ev.url.replace("(", "%28").replace(")", "%29")
        sources.append(
            Source(
                id=remap[old],
                title=ev.title or ev.domain,
                url=ev.url,
                deep_link_url=deep_link,
                supporting_quote=passage,
                quote_verified=verified,
            )
        )

    def _link(m: re.Match[str]) -> str:
        n = int(m.group(1))
        if n not in remap:
            return ""  # hallucinated / out-of-range id: drop the marker
        src = sources[remap[n] - 1]
        return f"[[{src.id}]]({src.deep_link_url})"

    linked_answer = BARE_CITATION_RE.sub(_link, answer)

    # Calibrated confidence: blend the writer's self-assessment, the evaluator's verdict and the
    # fraction of citations whose quotes were verified in the source text.
    verified_ratio = (sum(s.quote_verified for s in sources) / len(sources)) if sources else 0.0
    parts = [(draft.confidence_score, 0.5), (verified_ratio, 0.2)]
    parts.append((evaluator_confidence if evaluator_confidence is not None else draft.confidence_score, 0.3))
    confidence = round(min(max(sum(v * w for v, w in parts), 0.0), 1.0), 2)
    if not sources:
        confidence = min(confidence, 0.2)

    return ResearchReport(
        answer=linked_answer,
        confidence_score=confidence,
        contradictions_noted=_dedupe([*draft.contradictions_noted, *extra_contradictions]),
        sources=sources,
    )
