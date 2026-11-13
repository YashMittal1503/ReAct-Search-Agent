# Evaluation harness

Measures answer quality and cost so pipeline changes (prompts, synthesis strategy, RAG, models) are
judged on evidence rather than impressions. Tracks issue
[#7](https://github.com/YashMittal1503/ReAct-Search-Agent/issues/7); used by the RAG experiment
([#19](https://github.com/YashMittal1503/ReAct-Search-Agent/issues/19)).

## Golden dataset — `golden/questions.jsonl`
25 questions across software, AI, health, energy, economics, a simple factual question, and 2
deliberately ambiguous prompts (`"AI"`, `"best database"`) that should trigger the clarification gate.

Each answerable question has **nuggets**: atomic claims a good answer must contain (TREC-style
nugget evaluation). There are 115 in total.
- `kind`: `fact` · `caveat` · `contradiction` (known disagreements between sources)
- `vital`: vital nuggets weigh 1.0, nice-to-have ones 0.5
- `time_sensitive`: facts that may drift (4 questions); re-check periodically
- `reviewed`: **all `false` initially.** The nuggets were drafted from well-documented sources and
  kept deliberately conservative, but a human should verify each one and flip the flag. Scores on
  unreviewed items are indicative, not authoritative.

Validate after editing: `python -m evals.run stats` (the schema rejects duplicate ids, nugget ids
without the question prefix, and answerable questions without nuggets).

## Modes
| Command | What it does | Cost |
|---|---|---|
| `python -m evals.run snapshot` | Runs the full pipeline once per answerable question and **freezes the evidence** (URL registry with scraped pages, evaluator verdict, notes, baseline report) to `snapshots/<qid>.json`. Existing snapshots are reused (`--force` to redo). | Highest: search, scraping and all LLM steps, ~23 runs |
| `python -m evals.run replay --variant <name>` | Re-runs **only synthesis** on the frozen evidence, then judges it. Same input for every variant → fair A/B. | Synthesis + judge calls only |
| `python -m evals.run e2e --variant <name>` | Full pipeline per question including the clarification gate. Realistic but noisy (live search results change). | Highest |
| `python -m evals.compare <runA> <runB>` | Paired comparison + ship rule (below). | Free |

Filters: `--ids q01 q07`, `--domains health energy`, `--limit 5`. Variants live in
[`variants.py`](variants.py): `baseline` (production defaults), `single_step` (legacy synthesis from
raw excerpts), `react_engine`, and on the `experiment/rag` branch `rag` (hybrid BM25 + bge-small passages), `rag_bm25` (lexical-only ablation) and `rag_tight` (8k-char passage budget). Add one entry per experiment.

Outputs: `results/<UTC>-<mode>-<variant>/` with `results.jsonl` (per question: scores, judge
verdicts with justifications, LLM usage, the answer), `summary.json` and `summary.md`. Snapshots and
results contain scraped web content and are git-ignored. When LangSmith is enabled, eval runs are
traced with `eval` / `question_id` metadata.

## Metrics
| Metric | How |
|---|---|
| **Nugget recall** | LLM judge marks each nugget *supported* (1) / *partial* (0.5) / *missing* (0) in the report; weighted by vital (1.0) vs non-vital (0.5) |
| Vital recall | Same, vital nuggets only |
| Contradiction coverage | Recall over `contradiction` nuggets |
| **Citation precision** | For up to 12 (claim sentence, cited passage) pairs, evenly sampled: does the passage support the claim? yes 1 / partial 0.5 / no 0 |
| Verified-quote ratio, confidence, cited sources | From the report (deterministic) |
| Gate recall / false-positive rate | e2e only: clarification requested on ambiguous / clear questions |
| Tokens, LLM calls, failed attempts, duration | From router telemetry (in `replay`: synthesis cost only) |

Means come with **95% bootstrap confidence intervals.** With ~23 questions they're wide, so the
summary says so.

## Ship rule (`evals.compare`)
On the questions both runs answered (paired), the candidate must:
- lose **≤ 2 points** of nugget recall and of citation precision (`--max-drop 0.02`), and
- cut LLM input tokens by **≥ 20%** (`--min-token-saving 0.20`).

If a paired-delta confidence interval crosses a limit, the check reports "within noise", and the
verdict becomes **INCONCLUSIVE** rather than a false pass or fail.

## Typical workflow
```bash
python -m evals.run snapshot                      # once (expensive)
python -m evals.run replay --variant baseline     # reference numbers
python -m evals.run replay --variant single_step  # e.g. re-check the two-step synthesis decision (#3)
python -m evals.compare evals/results/<baseline-run> evals/results/<single_step-run>
```

## Caveats
- The judge is an LLM from the same router chain; it can share biases with the writer model.
  Judge verdicts include short evidence or reasons so disagreements can be audited.
- Snapshots freeze *retrieval*; `replay` measures synthesis-side changes only. Retrieval changes
  need fresh snapshots or `e2e` runs.
