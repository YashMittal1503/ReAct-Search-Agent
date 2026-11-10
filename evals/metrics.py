"""Scoring and aggregation for evaluation runs."""

from __future__ import annotations

import random
import re
from collections import defaultdict
from statistics import mean
from typing import Any

from evals.dataset import GoldenQuestion, Nugget
from evals.judge import CitationPair, CitationVerdict, NuggetVerdict
from schemas.report import LINKED_CITATION_RE, ResearchReport
from tools.deep_links import clean_markdown

VERDICT_SCORE = {"supported": 1.0, "partial": 0.5, "missing": 0.0}
CITATION_SCORE = {"yes": 1.0, "partial": 0.5, "no": 0.0}
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


# ---------------------------------------------------------------------------
# Nuggets
# ---------------------------------------------------------------------------
def nugget_scores(nuggets: list[Nugget], verdicts: dict[str, NuggetVerdict]) -> dict[str, float | None]:
    """Weighted nugget recall (vital = 1.0, non-vital = 0.5), vital-only recall, contradiction coverage."""
    if not nuggets:
        return {"nugget_recall": None, "vital_recall": None, "contradiction_coverage": None}

    def score(n: Nugget) -> float:
        return VERDICT_SCORE[verdicts[n.id].verdict] if n.id in verdicts else 0.0

    weights = {n.id: 1.0 if n.vital else 0.5 for n in nuggets}
    recall = sum(score(n) * weights[n.id] for n in nuggets) / sum(weights.values())
    vital = [n for n in nuggets if n.vital]
    contra = [n for n in nuggets if n.kind == "contradiction"]
    return {
        "nugget_recall": round(recall, 4),
        "vital_recall": round(mean(score(n) for n in vital), 4) if vital else None,
        "contradiction_coverage": round(mean(score(n) for n in contra), 4) if contra else None,
    }


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------
def _window(text: str, quote: str, size: int = 1_500) -> str:
    """A passage of ``text`` centred on ``quote`` (or its start), for the citation judge."""
    if not text:
        return quote
    idx = text.lower().find(quote[:60].lower()) if quote else -1
    if idx < 0:
        return text[: 2 * size]
    start = max(0, idx - size)
    return text[start: idx + len(quote) + size]


def citation_pairs(report: ResearchReport, source_texts: dict[str, str] | None = None,
                   max_pairs: int = 12) -> list[CitationPair]:
    """(claim sentence, cited passage) pairs, deduplicated and evenly sampled across the report.

    The passage is the source's verified supporting quote plus surrounding text when the full source
    is available (replay mode), otherwise the supporting quote alone.
    """
    by_id = {s.id: s for s in report.sources}
    pairs: list[CitationPair] = []
    seen: set[tuple[str, int]] = set()
    for sentence in _SENTENCE_RE.split(report.answer):
        ids = [int(i) for i, _ in LINKED_CITATION_RE.findall(sentence)]
        claim = re.sub(r"\s+([.,;:!?])", r"\1", clean_markdown(LINKED_CITATION_RE.sub("", sentence))).strip(" -*#>")
        if not ids or len(claim) < 15:
            continue
        for sid in dict.fromkeys(ids):
            src = by_id.get(sid)
            if src is None or (claim, sid) in seen:
                continue
            seen.add((claim, sid))
            text = (source_texts or {}).get(src.url, "")
            passage = _window(text, src.supporting_quote) if text else src.supporting_quote
            pairs.append(CitationPair(source_id=sid, claim=claim[:600], source_title=src.title, passage=passage[:3_500]))
    if len(pairs) <= max_pairs:
        return pairs
    step = len(pairs) / max_pairs
    return [pairs[int(i * step)] for i in range(max_pairs)]


def citation_precision(verdicts: list[CitationVerdict]) -> float | None:
    return round(mean(CITATION_SCORE[v.supported] for v in verdicts), 4) if verdicts else None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def bootstrap_ci(values: list[float], *, iterations: int = 2_000, alpha: float = 0.05,
                 seed: int = 7) -> tuple[float, float] | None:
    """Percentile bootstrap confidence interval for the mean (deterministic seed)."""
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(values, k=len(values))) for _ in range(iterations))
    lo = means[int(alpha / 2 * iterations)]
    hi = means[int((1 - alpha / 2) * iterations) - 1]
    return round(lo, 4), round(hi, 4)


def _mean_ci(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return {"mean": round(mean(vals), 4) if vals else None, "ci95": bootstrap_ci(vals), "n": len(vals)}


def summarize(rows: list[dict[str, Any]], questions: list[GoldenQuestion]) -> dict[str, Any]:
    """Aggregate per-question results into the run summary."""
    qmap = {q.id: q for q in questions}
    answerable = [r for r in rows if not qmap[r["id"]].expected_clarification and r.get("status") == "ok"]
    ambiguous = [r for r in rows if qmap[r["id"]].expected_clarification]
    clear = [r for r in rows if not qmap[r["id"]].expected_clarification]

    by_domain: dict[str, list[float]] = defaultdict(list)
    for r in answerable:
        if r.get("nugget_recall") is not None:
            by_domain[qmap[r["id"]].domain].append(r["nugget_recall"])

    def total(key: str) -> int:
        return int(sum((r.get("llm") or {}).get(key, 0) for r in rows))

    summary: dict[str, Any] = {
        "questions": len(rows),
        "answered": len(answerable),
        "errors": sum(r.get("status") == "error" for r in rows),
        "nugget_recall": _mean_ci(answerable, "nugget_recall"),
        "vital_recall": _mean_ci(answerable, "vital_recall"),
        "citation_precision": _mean_ci(answerable, "citation_precision"),
        "contradiction_coverage": _mean_ci(answerable, "contradiction_coverage"),
        "verified_ratio": _mean_ci(answerable, "verified_ratio"),
        "confidence": _mean_ci(answerable, "confidence"),
        "cited_sources": _mean_ci(answerable, "cited_sources"),
        "recall_by_domain": {d: round(mean(v), 4) for d, v in sorted(by_domain.items())},
        "input_tokens": total("input_tokens"),
        "output_tokens": total("output_tokens"),
        "llm_calls": total("llm_calls"),
        "failed_attempts": total("failed_attempts"),
        "mean_duration_s": round(mean(r.get("duration_s", 0.0) for r in rows), 2) if rows else 0.0,
    }
    if ambiguous:
        summary["gate_recall"] = round(mean(1.0 if r.get("clarification_requested") else 0.0 for r in ambiguous), 4)
    if clear and any("clarification_requested" in r for r in clear):
        summary["gate_false_positive_rate"] = round(
            mean(1.0 if r.get("clarification_requested") else 0.0 for r in clear), 4)
    return summary


def summary_markdown(summary: dict[str, Any], meta: dict[str, Any]) -> str:
    """Human-readable summary table."""
    def fmt(m: Any) -> str:
        if isinstance(m, dict):
            if m.get("mean") is None:
                return "—"
            ci = f" (95% CI {m['ci95'][0]:.2f}–{m['ci95'][1]:.2f})" if m.get("ci95") else ""
            return f"{m['mean']:.3f}{ci} · n={m['n']}"
        return f"{m:,}" if isinstance(m, int) else str(m)

    lines = [f"# Eval run `{meta.get('run_name', '')}`", "",
             f"- mode: **{meta.get('mode')}** · variant: **{meta.get('variant')}** · commit `{meta.get('git_commit', '?')}`",
             f"- judge: {meta.get('judge', '?')} · dataset: {meta.get('dataset_size', '?')} questions", "",
             "| Metric | Value |", "|---|---|"]
    for key in ("nugget_recall", "vital_recall", "citation_precision", "contradiction_coverage", "verified_ratio",
                "confidence", "cited_sources", "gate_recall", "gate_false_positive_rate", "answered", "errors",
                "input_tokens", "output_tokens", "llm_calls", "failed_attempts", "mean_duration_s"):
        if key in summary:
            lines.append(f"| {key} | {fmt(summary[key])} |")
    if summary.get("recall_by_domain"):
        lines += ["", "**Nugget recall by domain:** " + ", ".join(f"{d} {v:.2f}" for d, v in summary["recall_by_domain"].items())]
    lines += ["", "_LLM-judged metrics are noisy with a small dataset; compare variants on the same snapshots and "
              "read the confidence intervals before concluding._"]
    return "\n".join(lines) + "\n"
