"""Compare two evaluation runs and apply the ship rule.

    python -m evals.compare evals/results/<baseline-run> evals/results/<candidate-run>

Ship rule (from #19): the candidate may not lose more than ``--max-drop`` (default 0.02 = 2 points) of
**nugget recall** or **citation precision**, and must cut LLM input tokens by at least
``--min-token-saving`` (default 20%). Deltas are computed on the questions both runs answered
(paired), with a bootstrap confidence interval on the paired difference, so a "pass" or "fail" that
is within noise is reported as such.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

from evals.metrics import bootstrap_ci

QUALITY_METRICS = ("nugget_recall", "citation_precision")
EXTRA_METRICS = ("vital_recall", "contradiction_coverage", "verified_ratio", "cited_sources", "confidence")


def load_run(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(path)
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (path / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return summary, rows


def paired(a_rows: list[dict[str, Any]], b_rows: list[dict[str, Any]], key: str) -> list[float]:
    """Per-question differences (b - a) for questions both runs scored."""
    a = {r["id"]: r.get(key) for r in a_rows if r.get("status") == "ok"}
    b = {r["id"]: r.get(key) for r in b_rows if r.get("status") == "ok"}
    return [b[q] - a[q] for q in sorted(a.keys() & b.keys())
            if isinstance(a[q], (int, float)) and isinstance(b[q], (int, float))]


def compare(a_dir: str | Path, b_dir: str | Path, *, max_drop: float = 0.02,
            min_token_saving: float = 0.20) -> dict[str, Any]:
    a_sum, a_rows = load_run(a_dir)
    b_sum, b_rows = load_run(b_dir)
    metrics: dict[str, Any] = {}
    for key in QUALITY_METRICS + EXTRA_METRICS:
        diffs = paired(a_rows, b_rows, key)
        metrics[key] = {
            "a": a_sum["summary"].get(key, {}).get("mean"),
            "b": b_sum["summary"].get(key, {}).get("mean"),
            "paired_delta": round(mean(diffs), 4) if diffs else None,
            "delta_ci95": bootstrap_ci(diffs),
            "n_paired": len(diffs),
        }

    a_tok, b_tok = a_sum["summary"].get("input_tokens", 0), b_sum["summary"].get("input_tokens", 0)
    token_saving = round(1 - b_tok / a_tok, 4) if a_tok else None

    checks: dict[str, str] = {}
    for key in QUALITY_METRICS:
        m = metrics[key]
        if m["paired_delta"] is None:
            checks[key] = "INCONCLUSIVE (no paired data)"
        elif m["paired_delta"] >= -max_drop:
            ci = m["delta_ci95"]
            checks[key] = "PASS" if not ci or ci[0] >= -max_drop else "PASS (within noise: CI crosses the limit)"
        else:
            ci = m["delta_ci95"]
            checks[key] = "FAIL" if not ci or ci[1] < -max_drop else "FAIL (within noise: CI crosses the limit)"
    checks["token_saving"] = ("INCONCLUSIVE (no token data)" if token_saving is None
                              else "PASS" if token_saving >= min_token_saving else "FAIL")

    if any(v.startswith("FAIL") for v in checks.values()):
        verdict = "DO NOT SHIP"
    elif any("INCONCLUSIVE" in v or "noise" in v for v in checks.values()):
        verdict = "INCONCLUSIVE: collect more data (more questions or repeated runs)"
    else:
        verdict = "SHIP"
    return {"a": a_sum["meta"], "b": b_sum["meta"], "metrics": metrics,
            "tokens": {"a": a_tok, "b": b_tok, "saving": token_saving}, "checks": checks, "verdict": verdict,
            "rule": {"max_drop": max_drop, "min_token_saving": min_token_saving}}


def to_markdown(result: dict[str, Any]) -> str:
    a, b = result["a"], result["b"]
    lines = [f"# {b.get('variant')} vs {a.get('variant')}", "",
             f"- A: `{a.get('run_name')}` (commit `{a.get('git_commit')}`)",
             f"- B: `{b.get('run_name')}` (commit `{b.get('git_commit')}`)", "",
             "| Metric | A | B | Paired Δ (B−A) | 95% CI of Δ | n |", "|---|---|---|---|---|---|"]
    for key, m in result["metrics"].items():
        fa = "—" if m["a"] is None else f"{m['a']:.3f}"
        fb = "—" if m["b"] is None else f"{m['b']:.3f}"
        d = "—" if m["paired_delta"] is None else f"{m['paired_delta']:+.3f}"
        ci = "—" if not m["delta_ci95"] else f"{m['delta_ci95'][0]:+.3f} … {m['delta_ci95'][1]:+.3f}"
        lines.append(f"| {key} | {fa} | {fb} | {d} | {ci} | {m['n_paired']} |")
    t = result["tokens"]
    saving = "—" if t["saving"] is None else f"{t['saving']:.1%}"
    lines += ["", f"**Input tokens:** A {t['a']:,} → B {t['b']:,} (saving {saving})", "",
              "**Checks:** " + ", ".join(f"{k}: {v}" for k, v in result["checks"].items()), "",
              f"## Verdict: {result['verdict']}"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI wiring
    parser = argparse.ArgumentParser(prog="python -m evals.compare")
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--max-drop", type=float, default=0.02)
    parser.add_argument("--min-token-saving", type=float, default=0.20)
    args = parser.parse_args(argv)
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    result = compare(args.baseline, args.candidate, max_drop=args.max_drop, min_token_saving=args.min_token_saving)
    md = to_markdown(result)
    (Path(args.candidate) / f"compare-vs-{Path(args.baseline).name}.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
