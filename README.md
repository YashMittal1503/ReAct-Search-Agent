# 🔎 Deep Research ReAct Agent

An **asynchronous, production-grade research agent** that turns a broad question such as
*"Pros and cons of microservices vs. monoliths in 2026"* into a fully formatted, **source-grounded
research report**. Every claim carries a clickable citation, and each citation **deep-links to the
exact supporting passage** on the source page.

Built with **LangGraph** (workflow orchestration and human-in-the-loop), **deepagents** (ReAct research
engine with planning and sub-agents), **Tavily** (search), **httpx + trafilatura** (async full-page
extraction), **Pydantic v2** (strict contracts) and **Streamlit** (chat UI with a live reasoning trace).
A **multi-tier, multi-key, cross-provider LLM router** (Groq → Cerebras → Mistral → Gemini → OpenRouter →
OpenAI) keeps research fast and absorbs rate limits.

```
Clarify ──► Plan ──► Search ──► Selective Scrape ──► Evaluate ──► Synthesize
   ▲  (HITL)   ▲                                       │
   └─ user ─┘  └────────── gaps / conflicts (≤ 4 iterations) ┘
```

---

## Table of contents
1. [What it does](#1-what-it-does)
2. [Architecture](#2-architecture)
3. [The pipeline, stage by stage](#3-the-pipeline-stage-by-stage)
4. [Resilient multi-provider orchestration & multi-key pools](#4-resilient-multi-provider-orchestration--multi-key-pools)
5. [Key engineering decisions & trade-offs](#5-key-engineering-decisions--trade-offs)
6. [Project structure](#6-project-structure)
7. [Data contracts (Pydantic v2)](#7-data-contracts-pydantic-v2)
8. [Getting started](#8-getting-started)
9. [Configuration reference](#9-configuration-reference)
10. [Testing](#10-testing)
11. [Walkthrough of a single request](#11-walkthrough-of-a-single-request)
12. [Interview guide: talking points & likely questions](#12-interview-guide-talking-points--likely-questions)
13. [Limitations & roadmap](#13-limitations--roadmap)

---

## 1. What it does

| Capability | How |
|---|---|
| **Ambiguity detection (human-in-the-loop)** | A cheap length heuristic plus an LLM `ClarityAssessment`. If the prompt is vague, the graph **pauses** with LangGraph `interrupt()` and asks 2-3 targeted questions. Your chat reply **resumes the same thread**. |
| **Query decomposition** | Turns the question into 2-4 non-overlapping search queries. On later iterations it **rewrites** queries to close identified gaps, and code filters out any query that already ran. |
| **Deduplicated two-tier retrieval** | Tier 1, `search_web`: Tavily discovery with titles, URLs, scores and snippets. Tier 2, `scrape_page`: full-page Markdown via httpx + trafilatura, used **only** when a snippet is relevant but too thin. A session-level `URLRegistry` normalizes URLs (strips `utm_*`, `gclid`, fragments, `www.`, trailing slashes and so on) and drops duplicate URLs *and* duplicate snippets (mirrored content). |
| **ReAct research engine** | A `deepagents` deep agent runs the Thought → Action → Observation loop. It gets a todo planner, a virtual scratch filesystem, auto-summarization, and a `source-reader` **sub-agent** it can delegate sub-questions to. |
| **Reflection / self-correction** | An evaluator judges coverage, lists gaps and conflicting claims, and proposes follow-up queries. It loops back to the planner up to `MAX_ITERATIONS=4`. |
| **Grounded synthesis** | The LLM writes Markdown with `[n]` markers and verbatim quotes. **Code** checks each quote against the retrieved text, builds `#:~:text=` deep links, rewrites markers to `[[n]](url)`, and Pydantic validates the citation contract. |
| **Streamlit UI** | Chat interface, live `st.status("Researching...")` trace (including deduplicated URLs and skipped scrapes), a metrics row, the report, a contradictions panel, and **Verified Sources** cards with deep links, domain badges and exact quotes. You can also download the report as `.md`. |
| **Rate-limit-proof LLM layer** | Multi-tier cross-provider failover with **multi-key pools** (unlimited keys per provider, round-robin), proactive RPM/TPM budgets, Retry-After-aware cooldowns and error-aware blame (key vs model vs provider). Exposed as one LangChain chat model, so deepagents and structured output use it unchanged. See [§4](#4-resilient-multi-provider-orchestration--multi-key-pools). |
| **Provider-agnostic** | Any OpenAI-compatible endpoint (Groq, Cerebras, Mistral, Gemini, OpenRouter, OpenAI, vLLM, Ollama). Adding a tier means adding a key to `.env`. |

---

## 2. Architecture

### 2.1 Layered view

```mermaid
flowchart TB
    subgraph UI["Presentation — app.py (Streamlit)"]
        chat[Chat + clarification turns]
        trace[Live trace in st.status]
        report[Report + Verified Sources]
    end

    subgraph Bridge["agent/async_bridge.py"]
        loop[Persistent asyncio loop in a daemon thread]
    end

    subgraph Session["agent/runner.py — ResearchSession"]
        run["run(message) → TraceEvent | ClarificationRequest | RunResult"]
    end

    subgraph Graph["agent/graph.py — LangGraph StateGraph + InMemorySaver"]
        A[assess_clarity] -->|ambiguous| B[ask_user ⏸ interrupt]
        B --> A
        A -->|clear| C[plan]
        C --> D[research]
        D --> E[evaluate]
        E -->|gaps & iter < max| C
        E -->|sufficient / max| F[synthesize]
    end

    subgraph Engine["agent/researcher.py — inner ReAct engine"]
        DA["deepagents.create_deep_agent<br/>(todos · virtual FS · summarization · source-reader sub-agent)"]
        RA["langchain create_agent<br/>(lightweight alternative)"]
    end

    subgraph Tools["tools/ — deterministic Python"]
        T1[search_web → TavilySearcher]
        T2[scrape_page → PageScraper httpx+trafilatura]
        REG[(URLRegistry<br/>normalize · dedup · guards)]
        DL[deep_links: verify_quote · text fragments]
    end

    UI --> Bridge --> Session --> Graph
    D --> Engine --> T1 & T2
    T1 & T2 <--> REG
    F --> DL
```

### 2.2 The two-level design in one sentence
**LangGraph owns *control flow*** (explicit, inspectable, checkpointed, interruptible), while
**deepagents owns *exploration*** (the open-ended ReAct tool loop inside the `research` node). The
tools enforce safety and efficiency in plain Python, so the LLM can't bypass them.

### 2.3 Request lifecycle (sequence)

```mermaid
sequenceDiagram
    actor U as User
    participant S as Streamlit
    participant G as LangGraph
    participant R as deepagents researcher
    participant T as Tools + URLRegistry
    participant L as LLM

    U->>S: "AI"
    S->>G: astream(initial_state)
    G->>L: ClarityAssessment
    G-->>S: interrupt(ClarificationRequest)
    S-->>U: 2-3 questions
    U->>S: "AI coding assistants, for a 10-person startup"
    S->>G: Command(resume=answer)  (same thread_id)
    G->>L: ResearchPlan (2-4 queries)
    G->>R: task + planned queries
    loop ReAct
        R->>L: Thought
        R->>T: search_web(queries) → dedup
        R->>T: scrape_page(url, reason) → guard → httpx+trafilatura
    end
    G->>L: EvidenceEvaluation
    alt gaps and iteration < 4
        G->>L: rewritten ResearchPlan
    else sufficient
        G->>L: SynthesisDraft ([n] markers + quotes)
        G->>G: verify quotes · deep links · [[n]](url) · Pydantic validate
        G-->>S: RunResult(ResearchReport)
    end
```

---

## 3. The pipeline, stage by stage

### ① Ambiguity Gate (`assess_clarity` + `ask_user`)
- **Heuristic:** fewer than `MIN_QUERY_WORDS` (4) words always counts as ambiguous.
- **LLM:** `ClarityAssessment {is_ambiguous, reasoning, missing_constraints, questions[≤3]}`. The prompt
  explicitly tells the model that *broad but well-formed* topics are clear, to avoid over-asking.
- **Pause and resume:** `ask_user` calls `interrupt(ClarificationRequest)`. The Streamlit app sees the
  `__interrupt__` event and shows the questions. The user's next message is sent as
  `Command(resume={"answer": ...})` on the **same `thread_id`**, and the checkpointer restores the
  full state.
- **Multi-turn:** after a reply, control goes back to `assess_clarity`. Up to
  `MAX_CLARIFICATION_ROUNDS` (2) rounds run, then the agent proceeds with stated assumptions.

> **Why two nodes?** LangGraph re-executes an interrupted node from the top on resume. Putting the
> LLM call in `assess_clarity` and *only* `interrupt()` in `ask_user` means resuming never re-pays for,
> or changes, the assessment.

### ② Query Decomposer (`plan`)
- Produces a `ResearchPlan {objective, sub_queries[2-4]}`. A Pydantic validator dedupes the queries and caps them at 4.
- **Query rewriting:** on iteration ≥ 2 the planner receives earlier queries, the coverage summary,
  gaps, conflicts and the evaluator's suggestions, and is told to write *new* angles. Any query the
  registry has already executed is then **filtered out in code**. If nothing new remains, the
  evaluator's follow-ups or a synthesized rewrite are used instead.

### ③ + ④ Search → Selective Scrape (`research`)
The node rebuilds the `URLRegistry` from state, creates the tools bound to this iteration's budget,
and runs the inner ReAct agent:

| Tool | Behaviour enforced *in code* |
|---|---|
| `search_web(queries: list[str])` | Runs queries **concurrently** (`asyncio.gather`). Skips queries already run. Normalizes every URL, drops duplicate URLs and duplicate snippets, and emits `dedup` trace events. Returns compact snippets with relevance scores. |
| `scrape_page(url, reason)` | **Guard rules:** (a) the URL must have been returned by `search_web` (blocks hallucinated URLs); (b) it must not be already scraped or in flight; (c) its relevance score must be ≥ `SCRAPE_RELEVANCE_THRESHOLD`; (d) the per-iteration budget must not be exhausted. The required `reason` argument makes the model justify *why the snippet lacks detail*. A refusal returns `SKIPPED: <reason>` and emits a `skip` event. A successful scrape returns only a 2.5k-char preview to keep the ReAct context small, while the full text (≤ 12k chars) is stored for the evaluator and synthesizer. |

**Scraper internals** (`tools/scraper.py`): a shared `httpx.AsyncClient` with timeouts, redirects,
connection limits and a semaphore. It streams the body with a size cap, rejects non-HTML content
types, and retries transient errors (429/5xx/transport) with backoff. `trafilatura.extract(...,
output_format="markdown")` runs in `asyncio.to_thread` because it's CPU-bound, with a
precision → recall fallback. **All failures come back as `ScrapedPage(ok=False, error=...)`, never as exceptions.**

**Resilience:** if the agent raises (rate limit, malformed tool call) or finishes without searching,
the node runs a **deterministic fallback**: it searches every planned query, then scrapes the top-K
unread URLs, using the *same guarded tools*.

### ⑤ Evidence Evaluator (`evaluate`)
`EvidenceEvaluation {is_sufficient, coverage_summary, gaps, conflicting_claims, follow_up_queries,
confidence}` is computed over a size-bounded, numbered evidence digest. The router loops back to
`plan` while evidence is insufficient **and** `iteration < MAX_ITERATIONS`. Conflicts are carried
forward into the final report.

### ⑥ Grounded Synthesizer (`synthesize` + `agent/citations.py`)
1. The LLM receives *numbered* evidence and returns a `SynthesisDraft`: a Markdown answer with `[n]`
   markers, a list of `{source_id, verbatim supporting_quote}`, confidence and contradictions. **It
   never writes URLs.**
2. Deterministic post-processing:
   - normalizes `[1, 2]` → `[1][2]` and strips any links or "Sources" section the model added anyway;
   - **drops hallucinated ids** (e.g. `[99]`) and renumbers sources 1..k by first appearance;
   - **verifies each quote** against the retrieved text (normalized substring match, otherwise a
     `difflib` best-sentence match ≥ 0.72). A verified quote gets a **Scroll-to-Text-Fragment** deep
     link, `url#:~:text=<first 4 words>,<last 4 words>`, percent-encoded with `-`, `,` and `&` escaped
     per the spec. An unverified source falls back to its real search snippet and a plain URL, and the
     UI flags it, so text that isn't from the source is never shown as a quote;
   - rewrites markers to `[[n]](deep_link_url)`.
3. **`ResearchReport` validators** re-check everything: unique ids, no bare `[n]` left, every link
   resolves to a known source and its URL, `confidence_score ∈ [0, 1]`, and each deep link extends its
   source URL.
4. **Calibrated confidence** = `0.5 × writer self-assessment + 0.3 × evaluator confidence + 0.2 ×
   fraction of verified quotes`, capped at 0.2 when there are no sources.

---

## 4. Resilient multi-provider orchestration & multi-key pools

A deep-research run makes **dozens of LLM calls**: control steps, the ReAct loop and parallel
sub-agents. On free and low tiers that hits rate limits within minutes. Groq's free TPM cap per
model, for example, is smaller than one deepagents prompt. The `agent/resilience/` package makes
rate limits a routing problem rather than a failure. It extends the pattern from my
[RAG-PROJECT](https://github.com/YashMittal1503/RAG-PROJECT) (`ProviderKeyPool` + tiered failover)
for a tool-calling agent.

### 4.1 Components

```mermaid
flowchart LR
    subgraph Nodes["Graph nodes / deepagents / create_agent / with_structured_output"]
        RCM["ResilientChatModel(role)<br/>LangChain BaseChatModel facade"]
    end
    RCM --> R["ModelRouter<br/>attempt planner + failure policy"]
    R --> C1["main chain<br/>groq:gpt-oss-120b → groq:llama-3.3-70b → cerebras → mistral → gemini → openrouter → openai"]
    R --> C2["fast chain<br/>groq:llama-3.1-8b → … → main models (last resort)"]
    R --> P1["KeyPool groq<br/>key1 key2 key3"]
    R --> P2["KeyPool gemini<br/>key1 …"]
    R --> P3["KeyPool openrouter …"]
    P1 --> S["KeySlot (per key)<br/>RPM + TPM token buckets per model<br/>cooldowns per model · disable on auth"]
    T["TavilySearcher"] --> PT["KeyPool tavily"]
```

| Module | Responsibility |
|---|---|
| `providers.py` | Registry of OpenAI-compatible providers (base URL, default fast/main models, default RPM), tier order, `provider:model` chain parsing, key discovery from `.env` |
| `key_pool.py` | `KeyPool` / `KeySlot`: round-robin rotation, **proactive RPM/TPM token buckets** per key and model, Retry-After-aware cooldowns with exponential backoff, auth disable, health snapshot, key masking |
| `errors.py` | Maps any SDK exception (openai, httpx, Tavily) to an `ErrorKind` and extracts Retry-After from headers (`retry-after`, `retry-after-ms`, Groq `x-ratelimit-reset-*`) or messages ("try again in 7.66s") |
| `router.py` | `ModelRouter` (attempt planning, failure policy, stats, events) and `ResilientChatModel`, which implements `bind_tools`, so deepagents, `create_agent` and `with_structured_output` work unchanged |

### 4.2 How a call is routed

1. **Estimate the request:** prompt tokens (approx.) + tool schemas + a completion allowance.
2. **Ready pass (never waits):** walk the chain in tier order; for each model, try its keys
   round-robin and take the first whose budgets and cooldowns allow sending now. Requests that
   exceed a model's TPM capacity skip it instantly (no wasted 413 round trip).
3. **Send** with SDK retries disabled (`max_retries=0`), so no hidden 1-2-4 s backoff sleeps happen
   inside the SDK. The router fails over in milliseconds instead.
4. **On failure, blame the right thing:**

   | Error | Action |
   |---|---|
   | 429 rate limit | cool **this key for this model** (Retry-After or 20 s → 40 s → … ≤ 300 s), next key |
   | 402 / quota / Tavily 432-433 | park the key for all models (1 h) |
   | 401/403 invalid key | disable the key |
   | 413 / context too large | skip the **model** (key untouched), try a higher-limit one |
   | 404 / decommissioned | remember the model as dead for 15 min |
   | invalid tool call | retry on a different model |
   | 5xx / timeout | short cooldown, fail over to the next provider |
   | our own bug | re-raise immediately |
5. **Patience pass:** only if nothing anywhere is ready, wait for the soonest key/model (bounded by
   `LLM_MAX_WAIT_SECONDS`).
6. **Observe:** every rotation is streamed into the UI trace as a 🔀 event (via a `ContextVar`
   listener inherited by sub-agent tasks), counted in the report's *LLM failovers* metric, and shown
   per key in the sidebar's **Provider health** panel.

### 4.3 Why this keeps research fast
- **Avoid instead of absorb:** proactive budgets mean most 429s are never triggered. Traffic moves to
  the next key or tier *before* the limit.
- **Round-robin across keys** spreads parallel sub-agent calls, so N keys ≈ N× throughput.
- **Per-model cooldown scope:** Groq and Gemini limit per model, so a 429 on one model leaves the
  key's other models usable.
- **Fast chain for control steps:** clarify, plan and evaluate run on small, fast models and keep the big
  models' budgets for research and synthesis.
- **Process-wide state:** the router and pools are Streamlit `cache_resource` singletons, so cooldowns
  learned in one request protect the next.
- **Stale models degrade gracefully:** a decommissioned default model costs one failed call, then
  it's skipped.

---

## 5. Key engineering decisions & trade-offs

| Decision | Why | Trade-off |
|---|---|---|
| **LangGraph outer graph + deepagents inner engine** | The outer workflow needs guarantees (iteration cap, HITL, checkpointing, explicit routing) that a free-running agent can't give. Searching and reading benefit from an autonomous loop with planning and sub-agents. | Two layers to understand, but each does what it's best at. |
| **Engine is a config flag (`deep` / `react`)** | deepagents adds a large system prompt and extra tools. Small or rate-limited models (Groq free tier) run better on `create_agent`. Both expose the same `{"messages"}` interface. | `react` gives up sub-agent delegation and todo planning. |
| **The LLM never writes URLs** | Removes the biggest hallucination vector in cited reports. Links come from the registry, so every citation points at a page that was actually retrieved. | Needs a post-processing layer (`citations.py`). |
| **Guards live in tools, not prompts** | Prompts are suggestions and code is law. Dedup, scrape budget, relevance threshold and "discovered URLs only" can't be talked around. | Thresholds need tuning per search provider. |
| **`URLRegistry` stored in graph state** (snapshot/restore) | It's checkpointed with the thread, survives interrupts, is isolated per session, and needs no global mutable store. | Page text is serialized into the checkpoint; fine in memory, but use a blob store at scale. |
| **Race-free dedup without locks** | All registry mutations are synchronous. A URL is *claimed* (`in_flight`) before the scrape `await`s, so concurrent tool calls can't double-fetch. | Relies on single-threaded asyncio semantics, which is documented in the code. |
| **Two-tier retrieval** | Snippets are cheap and often enough, while full pages are expensive in latency and tokens. Scraping only when needed cuts cost and context bloat. | The model can occasionally skip a page that would have helped; the evaluator loop mitigates this. |
| **`interrupt()` isolated in its own node** | Resume re-runs the node, so keeping it LLM-free avoids duplicate cost and non-determinism. | One extra node. |
| **Persistent background event loop for Streamlit** | `asyncio.run()` per rerun breaks loop-bound clients ("Event loop is closed"). One daemon-thread loop plus a queue streams events to the script thread live. | Slightly more plumbing (`async_bridge.py`). |
| **Dependency injection (`AgentDeps`)** | Every external dependency (LLM, structured-output call, searcher, scraper, agent factory, clock) can be swapped, so the full graph is tested with fakes and no network. | A bit of indirection. |
| **Structured output with self-correction** | Open-weight models sometimes emit invalid tool arguments. `structured_call` feeds the validation error back once before failing, and each node has a graceful fallback. | One extra LLM call in the failure case. |
| **Router as a `BaseChatModel`, not `with_fallbacks()`** | LangChain's `RunnableWithFallbacks` has no `bind_tools` (so deepagents can't use it), no key pools and no cooldown memory. A custom chat model makes failover invisible to every caller. | Owns its own retry/failover logic, which is fully unit-tested. |
| **Proactive budgets + reactive cooldowns** | Budgets prevent most 429s, and cooldowns handle the rest using the provider's own Retry-After. | Budgets are estimates (chars/4 token counting); defaults are conservative free-tier values. |
| **Error classification decides blame** | Punishing a key for a 413 or a dead model wastes healthy capacity. Classification picks key-, model- or provider-level action. | Message heuristics must track provider wording; all mapped cases are covered by tests. |

---

## 6. Project structure

```
.
├── app.py                     # Streamlit chat UI (HITL, live trace, report, Verified Sources)
├── agent/
│   ├── config.py              # Settings (pydantic-settings, .env)
│   ├── llm.py                 # Router-backed chat models + structured_call (self-correcting)
│   ├── resilience/            # Multi-tier, multi-key, cross-provider orchestration
│   │   ├── providers.py       #   provider registry, tier order, model chains, key discovery
│   │   ├── key_pool.py        #   KeyPool/KeySlot: round-robin, RPM/TPM buckets, cooldowns, health
│   │   ├── errors.py          #   ErrorKind classification + Retry-After parsing
│   │   └── router.py          #   ModelRouter + ResilientChatModel (LangChain facade)
│   ├── prompts.py             # All system prompts
│   ├── state.py               # ResearchState (TypedDict + reducers), checkpoint-friendly
│   ├── nodes.py               # assess_clarity · ask_user · plan · research · evaluate · synthesize + routers
│   ├── researcher.py          # deepagents / create_agent factory (inner ReAct engine)
│   ├── citations.py           # Draft → grounded ResearchReport (quote verification, deep links, linking)
│   ├── graph.py               # StateGraph wiring + checkpointer
│   ├── runner.py              # ResearchSession async API + CLI
│   └── async_bridge.py        # Persistent event loop for Streamlit
├── tools/
│   ├── url_registry.py        # normalize_url · URLRegistry (dedup, scrape guards, snapshot)
│   ├── search.py              # TavilySearcher (async, concurrent, multi-key rotation)
│   ├── scraper.py             # PageScraper (httpx streaming + trafilatura Markdown)
│   ├── deep_links.py          # verify_quote · build_text_fragment (#:~:text=)
│   └── langchain_tools.py     # search_web / scrape_page LangChain tools with guards + trace events
├── schemas/
│   ├── clarification.py       # ClarityAssessment, ClarificationRequest
│   ├── planning.py            # ResearchPlan, SubQuery
│   ├── evidence.py            # SearchResult, ScrapedPage, Evidence, EvidenceEvaluation
│   ├── report.py              # SynthesisDraft, Source, ResearchReport (strict validators)
│   └── events.py              # TraceEvent (thought/action/observation/dedup/skip/error/status)
├── tests/                     # 66 tests: registry, deep links, scraper, schema, graph flow, key pools, router
├── .env.example · requirements.txt · pyproject.toml · .streamlit/config.toml
```

---

## 7. Data contracts (Pydantic v2)

```python
class Source(BaseModel):
    id: int
    title: str
    url: str                      # normalized, http(s) only
    deep_link_url: str            # url + "#:~:text=start,end" when the quote is verified
    supporting_quote: str         # verbatim passage from the source
    quote_verified: bool
    domain: str                   # computed field, for the UI badge

class ResearchReport(BaseModel):
    answer: str                   # Markdown; every citation is "[[n]](url)"
    confidence_score: float       # 0.0 – 1.0
    contradictions_noted: list[str]
    sources: list[Source]
    # model_validator: unique ids · no bare [n] · each [[n]](url) resolves to source n and its url
```

The trace stream uses `TraceEvent {kind, node, message, data, ts}`, where `kind ∈ {status, thought,
action, observation, dedup, skip, error}`.

---

## 8. Getting started

### Prerequisites
- Python **3.11+**
- A **Tavily** API key (free tier available at tavily.com)
- At least one **LLM provider key**: Groq, Cerebras, Mistral, Gemini, OpenRouter, OpenAI, or a local
  OpenAI-compatible server. More keys and more providers give more throughput and resilience.

### Install
```bash
python -m venv .venv
```
```bash
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
```
```bash
pip install -r requirements.txt
```
```bash
cp .env.example .env            # then add provider keys (e.g. GROQ_API_KEY…) and TAVILY_API_KEY
```

### Run the web app
```bash
streamlit run app.py
```
Try the built-in examples. **"AI"** deliberately triggers the Ambiguity Gate.

### Run from the command line
```bash
python -m agent.runner "Pros and cons of microservices vs. monoliths in 2026"
```

### Use as a library
```python
from agent import ResearchSession, RunResult, default_graph

session = ResearchSession(default_graph())
async for item in session.run("State of solid-state EV batteries in 2026"):
    if isinstance(item, RunResult):
        print(item.report.answer)
```

### Model & rate-limit tips
- **Add keys before anything else.** Each extra key adds throughput, and each extra provider adds a
  failover tier. A practical free setup: 2-3 Groq keys + 1 Cerebras + 1 Gemini + 1 OpenRouter.
- Keep `GROQ_TPM` set on the free tier. Oversized deep-agent prompts then go straight to a
  higher-limit provider instead of failing with 413.
- Default model ids live in `agent/resilience/providers.py`. Override them per role with
  `LLM_CHAIN_MAIN` / `LLM_CHAIN_FAST`. Unknown or decommissioned models are skipped automatically.
- Tool calling matters most for the `main` chain (Llama 3.3 70B, gpt-oss-120b, Gemini Flash and
  GPT-4o-mini all work). Small models (≤ 8B) belong in the `fast` chain.
- If you only have one small-quota key, `RESEARCH_ENGINE=react` sends much smaller prompts.

---

## 9. Configuration reference

| Variable | Default | Purpose |
|---|---|---|
| `<P>_API_KEY`, `<P>_API_KEY_2…`, `<P>_API_KEYS` | — | Keys per provider `P` ∈ `GROQ, CEREBRAS, MISTRAL, GEMINI, OPENROUTER, OPENAI` (unlimited) |
| `<P>_RPM` / `<P>_TPM` | free-tier RPM; TPM off | Proactive per-key, per-model budgets (`0`/`off` disables) |
| `LLM_CHAIN_MAIN` / `LLM_CHAIN_FAST` | auto (tier order) | Explicit `provider:model,…` fallback chains |
| `LLM_MAX_WAIT_SECONDS` | `45` | Longest wait when every key is saturated |
| `LLM_COOLDOWN_BASE_SECONDS` / `_MAX_SECONDS` | `20` / `300` | 429 backoff when there's no Retry-After |
| `LLM_BASE_URL`, `LLM_MODEL`, `LLM_FAST_MODEL`, `LLM_API_KEY(S)` | — | Optional extra "custom" tier (vLLM, Ollama, LM Studio…) |
| `LLM_STRUCTURED_METHOD` | `function_calling` | Structured-output method |
| `TAVILY_API_KEY`, `TAVILY_API_KEY_2…`, `TAVILY_API_KEYS` | — | Search keys (rotated like LLM keys) |
| `TAVILY_RPM` | `100` | Proactive per-key Tavily budget |
| `TAVILY_SEARCH_DEPTH` | `basic` | `basic` / `advanced` / `fast` |
| `SEARCH_RESULTS_PER_QUERY` | `5` | Results per sub-query |
| `RESEARCH_ENGINE` | `deep` | `deep` (deepagents) or `react` (create_agent) |
| `MAX_ITERATIONS` | `4` | Reflection loop cap |
| `MAX_CLARIFICATION_ROUNDS` | `2` | HITL rounds before proceeding with assumptions |
| `MIN_QUERY_WORDS` | `4` | Shorter prompts always trigger clarification |
| `MAX_SCRAPES_PER_ITERATION` | `4` | Scrape budget per iteration |
| `SCRAPE_RELEVANCE_THRESHOLD` | `0.5` | Minimum Tavily score to allow a full scrape |
| `SCRAPE_TIMEOUT_SECONDS` / `SCRAPE_MAX_CHARS` | `15` / `12000` | Fetch timeout and stored text cap |
| `MAX_REPORT_SOURCES` | `12` | Evidence items shown to the synthesizer |

---

## 10. Testing

```bash
python -m pytest -q
```
```bash
python -m pytest -q --doctest-modules tools
```

The suite needs **no API keys and no network**:

| File | Covers |
|---|---|
| `test_url_registry.py` | URL normalization table, duplicate URL and snippet filtering, scrape guards (unknown, low relevance, in flight, already scraped), snapshot round-trip |
| `test_deep_links.py` | Whole-quote vs. range fragments, delimiter escaping, Markdown-insensitive verification, paraphrase snapping, unrelated-quote rejection |
| `test_scraper.py` | trafilatura extraction via `httpx.MockTransport`; PDF, 404 and connection errors returned (not raised); 503 retry; truncation |
| `test_report_schema.py` | Citation contract: bare `[n]`, unknown id, URL mismatch, confidence bounds, duplicate ids, deep-link base check |
| `test_resilience_primitives.py` | Key parsing and masking, token buckets, round-robin, per-model cooldown scope, exponential backoff, Retry-After precedence, auth disable / quota parking, error classification for real `openai` SDK exceptions and Tavily errors, provider discovery and chain building |
| `test_router.py` | Key rotation on 429, cross-provider failover, round-robin distribution, 413 skips the model and not the key, **proactive TPM/RPM routing**, dead-model memory, bounded waiting, exhaustion and fatal re-raise, sync path, `with_structured_output`, a real `create_agent` tool loop and a real **deepagents** run on the router, failover events reaching the LangGraph stream, Tavily key rotation |
| `test_graph_flow.py` | **Full graph:** clarify → `interrupt` → resume → 2 iterations with query rewriting → synthesis with renumbering, hallucinated-id removal, verified deep links and contradictions. Also covers agent-failure fallback with the `max_iterations` cap, and that the real `deepagents` and `create_agent` engines compile against the installed libraries. |

---

## 11. Walkthrough of a single request

*Illustrative example. Exact queries, counts and sources vary by model and by what the web returns.*

**Input:** `"microservices?"`
1. `assess_clarity`: 1 word, below the minimum, so the LLM proposes *"Which architecture are you comparing
   against?"* and *"What team size and domain?"*. The graph pauses at `ask_user`.
2. The user replies *"vs monoliths, backend web apps, team of 8"*. `Command(resume=...)` merges the
   reply into `clarified_query`, and the re-assessment says it's clear.
3. `plan` (iteration 1): `["microservices vs monolith trade-offs 2026", "monolith operational cost small teams", "microservices team size threshold"]`.
4. `research`: the deep agent calls `search_web` with all three queries (15 results, 3 duplicates
   filtered, e.g. the same InfoQ article with `?utm_source=`). It scrapes 3 high-scoring pages whose
   snippets lack numbers. One scrape returns `SKIPPED: low relevance score 0.31`, and one PDF fails
   gracefully.
5. `evaluate`: *insufficient*. The gap is "no data on migration cost", and there's a conflict over
   whether "microservices improve deployment frequency" depends on team size.
6. `plan` (iteration 2): rewritten queries such as `"monolith to microservices migration cost case study"`.
7. `research` → `evaluate`: *sufficient*.
8. `synthesize`: draft with `[1]…[7]`. Quotes are verified and deep links built, then Pydantic validates
   and the report renders, with Verified Sources cards whose titles jump straight to the highlighted passage.

---

## 12. Interview guide: talking points & likely questions

**Elevator pitch:** *"A research agent that treats the LLM as an untrusted reasoning component. The
LLM decides what to search and read and writes the prose. Deterministic code owns deduplication,
budgets, URL integrity and citation linking, and Pydantic validates the final contract. LangGraph gives
me explicit, checkpointed control flow with human-in-the-loop, and deepagents gives me an autonomous
ReAct researcher with planning and sub-agents."*

**Q: Why not just one big agent?**
Because you can't reliably guarantee "ask before searching", "at most 4 iterations" or "always produce
a validated report" with prompts alone. Those are workflow invariants, so they live in a graph. The
open-ended part (which pages to read) is where an agent adds value, so that's the only place it runs.

**Q: How do you prevent hallucinated citations?**
Three layers. (1) The LLM only emits `[n]`, never URLs. (2) Ids that don't exist are dropped, and URLs
come from the registry of pages actually retrieved. (3) Quotes are checked against the source text, and
only verified passages get a highlight deep link. Unverified sources show their real snippet and are
flagged in the UI. Pydantic then enforces the contract.

**Q: How does human-in-the-loop work technically?**
`interrupt()` saves the graph state through the checkpointer and surfaces a payload. The UI renders the
questions. The next user message is sent as `Command(resume=...)` with the same `thread_id`, and LangGraph
re-enters the paused node with the answer as `interrupt()`'s return value.

**Q: How is deduplication made robust?**
Canonical URLs (scheme and host case, `www.`, default ports, tracking params, fragments, trailing
slashes, sorted query), plus snippet fingerprints that catch syndicated copies under different URLs,
plus query dedup so rewrites don't re-run earlier searches. The scrape claim happens before any
`await`, which prevents double-fetches under concurrency.

**Q: How would you scale this to production?**
Swap `InMemorySaver` for `PostgresSaver`/`SqliteSaver`, move page bodies to object storage with
references kept in state, serve the graph behind LangGraph Platform or FastAPI with SSE, add a shared
Redis cache for pages and searches keyed by normalized URL and query, add per-tenant rate limiting, and
add LangSmith tracing and eval datasets for evaluator quality and citation precision.

**Q: How do you evaluate quality?**
Measurable proxies: citation precision (the verified-quote ratio is already computed), source diversity
(unique domains), iterations-to-sufficiency, and scrape efficiency (useful pages ÷ scrapes). Next would
be an LLM-as-judge rubric on a fixed question set, with regression runs in CI.

**Q: What breaks most often?**
Weak tool calling on small models (mitigated by structured-output self-correction, the `react` engine
and the deterministic fallback), paywalled or JS-only pages (trafilatura returns nothing, so the agent
falls back to the snippet), and provider rate limits (handled by the router: see the next question).

**Q: How do you deal with rate limits?**
With a router that treats them as a scheduling problem. Every provider has a pool of keys with
per-model RPM/TPM token buckets, so requests go to a key that has budget *before* a 429 happens.
When a 429 does happen, the key is cooled for exactly the provider's Retry-After, scoped to that model,
and the call moves to the next key, then the next model, then the next provider, in milliseconds,
because SDK retries are off. Errors are classified so a 413 skips the model without punishing the
key, a bad key is disabled, and a dead model is remembered. It's a LangChain `BaseChatModel`, so
deepagents and structured output didn't change at all.

**Q: Why not LangChain's `with_fallbacks()`?**
It returns a `RunnableWithFallbacks`, which has no `bind_tools`, so deepagents and `create_agent`
can't use it. It also has no notion of keys, cooldowns, budgets or shared state across requests.

---

## 13. Limitations & roadmap

**Known limitations**
- Text fragments highlight in Chromium browsers and Safari 16.1+. Firefox opens the page without the highlight.
- JS-rendered pages and PDFs aren't extracted. The agent relies on their snippets.
- Checkpoints are in-memory, so restarting the app clears conversations.
- Confidence is a heuristic blend, not a calibrated probability.
- Router and key-pool state (cooldowns, budgets) lives in process memory. Multiple app replicas would
  each keep their own view; a shared Redis-backed pool is the fix.
- Token budgets use an approximate count (chars/4), so TPM limits are respected approximately, not exactly.

**Roadmap**
- [ ] Redis-backed key pools so cooldowns and budgets are shared across replicas
- [ ] Learn real limits from `x-ratelimit-remaining-*` response headers to auto-tune RPM/TPM budgets
- [ ] PDF extraction (`pypdf`) and an optional headless-browser fetcher for JS-heavy sites
- [ ] Persistent checkpointer (SQLite/Postgres) and a conversation history sidebar
- [ ] Source credibility scoring (domain allow and deny lists, publication date freshness)
- [ ] Token-level streaming of the final report
- [ ] LangSmith tracing and an offline evaluation harness
- [ ] FastAPI + SSE backend so the agent can serve non-Streamlit clients
