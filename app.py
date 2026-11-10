"""Streamlit chat frontend for the Deep Research ReAct Agent.

Run with:  ``streamlit run app.py``

Features
--------
* Chat UI (``st.chat_message`` / ``st.chat_input``) with multi-turn clarification: when the
  Ambiguity Gate pauses the graph, the assistant asks its questions and the *next* chat message resumes the same LangGraph thread.
* Live Thought / Action / Observation trace inside ``st.status("Researching...")``, including deduplicated URLs and skipped scrapes.
* Final Markdown report with clickable inline citations, plus a "Verified Sources" panel with text-fragment deep links (``target="_blank"``), domain badges and the exact extracted quotes.
"""

from __future__ import annotations

import html
import pathlib
import uuid
from typing import Any

import streamlit as st
from langgraph.checkpoint.memory import InMemorySaver

from agent.async_bridge import BackgroundLoop
from agent.config import Settings, get_settings
from agent.graph import build_graph
from agent.llm import build_router
from agent.nodes import AgentDeps
from agent.observability import RunRecorder, configure_logging
from agent.observability.tracing import TracingConfig, configure_tracing
from agent.resilience import KeyPool, ModelRouter, load_env, parse_keys
from agent.runner import ResearchSession, RunResult
from schemas.clarification import ClarificationRequest
from schemas.events import TraceEvent
from schemas.report import ResearchReport
from tools.search import build_tavily_pool

st.set_page_config(page_title="Deep Research Agent", page_icon="🔎", layout="wide")

EXAMPLES = [
    "Pros and cons of microservices vs. monoliths in 2026",
    "How effective are GLP-1 drugs for long-term weight loss, and what are the main risks?",
    "State of solid-state batteries for EVs: who is closest to mass production?",
    "AI",  # deliberately vague -> triggers the Ambiguity Gate
]

#: Number of most-recent steps shown in the live trace window (older ones scroll off).
TRACE_WINDOW = 8
#: Pixel height of the scrollable boxes for the full trace and the end-of-run summary.
TRACE_BOX_HEIGHT = 320

CSS = """
<style>
.src-card{border:1px solid rgba(128,128,128,.28);border-radius:12px;padding:12px 14px;margin:0 0 10px 0;
          background:rgba(128,128,128,.06)}
.src-head{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:6px}
.src-id{font-weight:700;font-variant-numeric:tabular-nums;opacity:.85}
.src-title{font-weight:600;text-decoration:none}
.src-title:hover{text-decoration:underline}
.badge{font-size:.72rem;padding:2px 8px;border-radius:999px;background:rgba(124,156,255,.18);
       border:1px solid rgba(124,156,255,.45)}
.badge.ok{background:rgba(46,160,67,.16);border-color:rgba(46,160,67,.5)}
.badge.warn{background:rgba(210,153,34,.16);border-color:rgba(210,153,34,.5)}
.src-quote{margin:6px 0 4px 0;padding:6px 10px;border-left:3px solid rgba(124,156,255,.7);
           font-style:italic;opacity:.92}
.src-raw{font-size:.78rem;opacity:.7;text-decoration:none}
.trace-line{font-size:.88rem;line-height:1.45;margin:1px 0;white-space:nowrap;overflow:hidden;
            text-overflow:ellipsis}
.trace-node{font-family:ui-monospace,monospace;font-size:.78rem;opacity:.7}
.trace-more{font-size:.78rem;opacity:.55;margin-bottom:4px}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached resources (one per server process)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_loop() -> BackgroundLoop:
    """Single persistent event loop shared by all reruns."""
    return BackgroundLoop()


@st.cache_resource
def get_checkpointer() -> InMemorySaver:
    """Shared checkpointer so paused threads survive graph rebuilds (e.g. settings changes)."""
    return InMemorySaver()


@st.cache_resource
def get_recorder() -> RunRecorder:
    """Configure logging once per process and return the persistent run recorder."""
    settings = get_settings()
    configure_logging(settings.log_dir, settings.log_level)
    return RunRecorder(pathlib.Path(settings.log_dir) / "runs", enabled=settings.run_logs)


@st.cache_resource
def get_tracing() -> TracingConfig:
    """Enable LangSmith tracing once per process if LANGSMITH_API_KEY is configured."""
    return configure_tracing()


@st.cache_resource
def get_router() -> ModelRouter:
    """Process-wide LLM router: key cooldowns and rate budgets persist across reruns and sessions."""
    return build_router(get_settings())


@st.cache_resource
def get_search_pool() -> KeyPool:
    """Process-wide Tavily key pool."""
    return build_tavily_pool(parse_keys(load_env(), "TAVILY"), rpm=get_settings().tavily_rpm)


@st.cache_resource
def get_graph(engine: str, max_iterations: int, threshold: float) -> Any:
    """Compiled graph for a given configuration (shares the router, key pools and checkpointer)."""
    settings = get_settings().model_copy(update={
        "research_engine": engine, "max_iterations": max_iterations, "scrape_relevance_threshold": threshold,
    })
    deps = AgentDeps(settings=settings, router=get_router(), search_pool=get_search_pool())
    return build_graph(deps, checkpointer=get_checkpointer())


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_state() -> None:
    st.session_state.setdefault("thread_id", uuid.uuid4().hex)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("queued_prompt", None)


def reset_conversation() -> None:
    st.session_state.thread_id = uuid.uuid4().hex
    st.session_state.messages = []


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def trace_line_html(ev: TraceEvent) -> str:
    """One trace step as a single-line HTML row (long messages are ellipsized; full text on hover)."""
    msg = html.escape(ev.message)
    return (f"<div class='trace-line' title='{msg}'>{ev.icon} "
            f"<span class='trace-node'>{html.escape(ev.node)}</span> {msg}</div>")


def render_live_trace(placeholder: Any, trace: list[TraceEvent], window: int = TRACE_WINDOW) -> None:
    """Rolling live view: re-render only the latest ``window`` steps so the box never grows.

    Older lines are dropped from the live view (they stay in ``trace`` and appear in the
    scrollable "Research trace" expander once the run ends).
    """
    hidden = max(len(trace) - window, 0)
    more = f"<div class='trace-more'>… {hidden} earlier step{'s' if hidden != 1 else ''}</div>" if hidden else ""
    placeholder.markdown(more + "".join(trace_line_html(ev) for ev in trace[-window:]), unsafe_allow_html=True)


def has_trace_summary(trace: list[TraceEvent]) -> bool:
    return any(e.kind in ("dedup", "skip", "failover") for e in trace)


def render_trace_summary(trace: list[TraceEvent], container: Any) -> None:
    """Deduplicated URLs, skipped scrapes and provider failovers, grouped for quick scanning."""
    dedup = [e for e in trace if e.kind == "dedup"]
    skips = [e for e in trace if e.kind == "skip"]
    failovers = [e for e in trace if e.kind == "failover"]
    if failovers:
        container.markdown(f"**🔀 Provider failovers ({len(failovers)})**")
        container.markdown("\n".join(f"- {html.escape(e.message)}" for e in failovers[:40]))
    if dedup:
        container.markdown(f"**♻️ Deduplicated ({len(dedup)})**")
        container.markdown("\n".join(f"- `{html.escape(e.data.get('url', e.message))}` — {e.data.get('reason', '')}"
                                     for e in dedup[:40]))
    if skips:
        container.markdown(f"**⏭️ Skipped scrapes ({len(skips)})**")
        container.markdown("\n".join(f"- `{html.escape(e.data.get('url', ''))}` — {html.escape(e.data.get('reason', ''))}"
                                     for e in skips[:40]))


def render_clarification(req: ClarificationRequest) -> None:
    st.markdown(f"🤔 **Before I start researching, a few quick questions** _(round {req.round})_")
    st.caption(req.reasoning)
    st.markdown("\n".join(f"{i}. {q}" for i, q in enumerate(req.questions, 1)))
    st.caption("Reply in the chat box below — answer any or all, or say “just go ahead”.")


def render_sources(report: ResearchReport) -> None:
    st.markdown("### 🔗 Verified Sources")
    st.caption("Titles open a deep link that scrolls to and highlights the supporting passage "
               "(Chrome, Edge, Safari 16.1+).")
    for s in report.sources:
        verified = ("<span class='badge ok'>✓ quote verified</span>" if s.quote_verified
                    else "<span class='badge warn'>snippet · quote not verified</span>")
        st.markdown(
            f"""<div class="src-card">
  <div class="src-head">
    <span class="src-id">[{s.id}]</span>
    <a class="src-title" href="{html.escape(s.deep_link_url, quote=True)}" target="_blank" rel="noopener noreferrer">{html.escape(s.title)}</a>
    <span class="badge">{html.escape(s.domain)}</span>{verified}
  </div>
  <div class="src-quote">“{html.escape(s.supporting_quote)}”</div>
  <a class="src-raw" href="{html.escape(s.url, quote=True)}" target="_blank" rel="noopener noreferrer">Open page without highlight ↗</a>
</div>""",
            unsafe_allow_html=True,
        )


def report_to_markdown(report: ResearchReport) -> str:
    lines = [report.answer, "", "## Sources", ""]
    lines += [f"{s.id}. [{s.title}]({s.deep_link_url}) — {s.domain}\n   > {s.supporting_quote}" for s in report.sources]
    return "\n".join(lines)


def render_run_metrics(result: RunResult) -> None:
    """LLM usage for this run: calls, tokens, latency and a per-model breakdown."""
    llm = result.llm or {}
    if not llm:
        return
    with st.expander(f"📊 Run metrics — {llm.get('llm_calls', 0)} LLM calls, "
                     f"{llm.get('input_tokens', 0) + llm.get('output_tokens', 0):,} tokens, {result.duration_s:.0f}s"):
        cols = st.columns(5)
        cols[0].metric("LLM calls", llm.get("llm_calls", 0), help="Successful calls (attempts incl. failures below)")
        cols[1].metric("Failed attempts", llm.get("failed_attempts", 0))
        cols[2].metric("Input tokens", f"{llm.get('input_tokens', 0):,}")
        cols[3].metric("Output tokens", f"{llm.get('output_tokens', 0):,}")
        cols[4].metric("LLM time", f"{llm.get('llm_time_s', 0):.1f}s")
        rows = [{"model": m, **v} for m, v in llm.get("by_model", {}).items()]
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        if llm.get("failures_by_kind"):
            st.caption("Failures by kind: " + ", ".join(f"{k} × {v}" for k, v in llm["failures_by_kind"].items()))
        if result.log_file:
            st.caption(f"Run log: `{result.log_file}` · run id `{result.run_id}`")


def render_report(result: RunResult, key: str, failovers: int = 0) -> None:
    report = result.report
    cols = st.columns(6)
    cols[0].metric("Confidence", f"{report.confidence_score:.0%}")
    cols[1].metric("Iterations", result.iterations)
    cols[2].metric("Sources found", result.stats.get("results", 0))
    cols[3].metric("Pages read", result.stats.get("pages_ok", 0))
    cols[4].metric("Scrapes skipped", len(result.skipped_scrapes))
    cols[5].metric("LLM failovers", failovers, help="Key rotations / provider switches absorbed by the router")
    st.progress(report.confidence_score)

    st.markdown(report.answer)

    if report.contradictions_noted:
        with st.expander(f"⚖️ Contradictions noted ({len(report.contradictions_noted)})", expanded=True):
            st.markdown("\n".join(f"- {c}" for c in report.contradictions_noted))

    if report.sources:
        render_sources(report)

    with st.expander(f"🔍 Search queries run ({len(result.queries)})"):
        st.markdown("\n".join(f"- {q}" for q in result.queries) or "_none_")

    render_run_metrics(result)
    if result.trace_url:
        st.link_button("🔎 View full trace in LangSmith", result.trace_url)

    st.download_button("⬇️ Download report (.md)", report_to_markdown(report), file_name="research_report.md",
                       mime="text/markdown", key=f"dl-{key}")


def render_history() -> None:
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🔎"):
            kind = msg["kind"]
            if kind == "text":
                st.markdown(msg["content"])
            elif kind == "clarification":
                render_clarification(msg["content"])
            elif kind == "report":
                render_report(msg["content"], key=str(i),
                              failovers=sum(e.kind == "failover" for e in msg.get("trace", [])))
            elif kind == "error":
                st.error(msg["content"])
            if msg.get("trace"):
                with st.expander(f"🧠 Research trace ({len(msg['trace'])} steps)"):
                    box = st.container(height=TRACE_BOX_HEIGHT)  # fixed height, scrolls internally
                    box.markdown("".join(trace_line_html(ev) for ev in msg["trace"]), unsafe_allow_html=True)
                    if has_trace_summary(msg["trace"]):
                        render_trace_summary(msg["trace"], st.container(height=TRACE_BOX_HEIGHT))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def sidebar(settings: Settings) -> tuple[str, int, float]:
    with st.sidebar:
        st.header("🔎 Deep Research Agent")
        st.caption("LangGraph workflow · deepagents ReAct engine · Tavily · trafilatura")
        if st.button("➕ New research", use_container_width=True):
            reset_conversation()
            st.rerun()

        st.subheader("Settings")
        engine = st.selectbox("Research engine", ["deep", "react"],
                              index=0 if settings.research_engine == "deep" else 1, key="engine",
                              help="deep = deepagents harness (planning + sub-agents); "
                                   "react = lightweight langchain create_agent loop.")
        max_iter = st.slider("Max iterations", 1, 6, settings.max_iterations)
        threshold = st.slider("Scrape relevance threshold", 0.0, 1.0, settings.scrape_relevance_threshold, 0.05)

        render_provider_health(get_router(), get_search_pool())
        tracing = get_tracing()
        st.subheader("Observability")
        if tracing.enabled:
            st.markdown(f"🟢 LangSmith tracing → project `{tracing.project}`")
        else:
            st.caption("LangSmith tracing off: add `LANGSMITH_API_KEY` to `.env` to enable.")
        st.caption(f"Logs: `{get_settings().log_dir}/app.jsonl` · runs: `{get_settings().log_dir}/runs/`")
        st.caption(f"Thread `{st.session_state.thread_id[:8]}`")
    return engine, max_iter, threshold


def render_provider_health(router: ModelRouter, search_pool: KeyPool) -> None:
    """Live view of the multi-tier, multi-key orchestration layer."""
    st.subheader("Provider health")
    health = router.health()
    if not router.configured:
        st.warning("No LLM provider keys found. Add e.g. `GROQ_API_KEY` (and more tiers) to `.env`.")
    if not len(search_pool):
        st.warning("No `TAVILY_API_KEY` found in `.env`.")
    st.caption(f"Failovers absorbed this session: **{health['failovers']}**")
    for role in ("main", "fast"):
        chain = health["chains"].get(role, [])
        if chain:
            st.markdown(f"**{role} chain**")
            st.markdown("\n".join(f"{i}. `{c['target']}`{' 💀' if c['dead'] else ''}" for i, c in enumerate(chain, 1)))
    with st.expander("Keys & limits", expanded=False):
        dot = {"ready": "🟢", "disabled": "🔴"}
        for prov in health["providers"]:
            limits = ", ".join(f"{n.upper()} {int(v):,}" for n, v in (("rpm", prov["rpm"]), ("tpm", prov["tpm"]),
                                                                      ("ctx", prov.get("context"))) if v)
            per_model = ", ".join(f"`{m}` {int(v):,}" if v else f"`{m}` ∞" for m, v in prov.get("model_tpm", {}).items())
            st.markdown(f"**{prov['label']}** · {len(prov['keys'])} key(s){' · ' + limits if limits else ''}")
            if per_model:
                st.caption(f"TPM per model: {per_model}")
            for k in prov["keys"]:
                icon = dot.get(k["status"], "🟡")
                st.markdown(f"{icon} `{k['key']}` {k['status']} · {k['ok']}/{k['calls']} ok · {k['rate_limited']} × 429")
        if len(search_pool):
            st.markdown(f"**Tavily** · {len(search_pool)} key(s)")
            for k in search_pool.snapshot():
                st.markdown(f"{dot.get(k['status'], '🟡')} `{k['key']}` {k['status']} · {k['ok']}/{k['calls']} ok")
        if health["dead_models"]:
            st.markdown("**Skipped models**")
            st.markdown("\n".join(f"- `{m}`" for m in health["dead_models"]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_turn(prompt: str, graph: Any) -> None:
    """Execute one user turn (new request or clarification reply) with live streaming."""
    st.session_state.messages.append({"role": "user", "kind": "text", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    settings = get_settings()
    chains = {role: [t.label for t in chain] for role, chain in get_router().chains.items()}
    session = ResearchSession(graph, thread_id=st.session_state.thread_id, recorder=get_recorder(),
                              metadata={"engine": st.session_state.get("engine", settings.research_engine),
                                        "chains": chains})
    trace: list[TraceEvent] = []
    with st.chat_message("assistant", avatar="🔎"):
        status = st.status("Researching...", expanded=True)
        live = status.empty()  # single slot, overwritten on every event -> rolling window
        try:
            for item in get_loop().iterate(lambda: session.run(prompt)):
                if isinstance(item, TraceEvent):
                    trace.append(item)
                    if item.kind == "status":
                        status.update(label=item.message)
                    render_live_trace(live, trace)
                elif isinstance(item, ClarificationRequest):
                    status.update(label="Need a bit more detail before researching", state="complete", expanded=False)
                    render_clarification(item)
                    st.session_state.messages.append(
                        {"role": "assistant", "kind": "clarification", "content": item, "trace": trace})
                elif isinstance(item, RunResult):
                    if has_trace_summary(trace):
                        render_trace_summary(trace, status.container(height=TRACE_BOX_HEIGHT))
                    status.update(label=f"Research complete — {len(item.report.sources)} cited sources",
                                  state="complete", expanded=False)
                    render_report(item, key=f"live-{len(st.session_state.messages)}",
                                  failovers=sum(e.kind == "failover" for e in trace))
                    st.session_state.messages.append(
                        {"role": "assistant", "kind": "report", "content": item, "trace": trace})
        except Exception as exc:  # noqa: BLE001 - show any failure in the chat instead of a stack trace
            status.update(label="Research failed", state="error", expanded=True)
            message = f"**{type(exc).__name__}:** {exc}"
            st.error(message)
            st.session_state.messages.append({"role": "assistant", "kind": "error", "content": message, "trace": trace})


def main() -> None:
    init_state()
    get_recorder()  # configure persistent logging once per process
    get_tracing()   # enable LangSmith before any LLM call (no-op without a key)
    settings = get_settings()
    engine, max_iter, threshold = sidebar(settings)
    graph = get_graph(engine, max_iter, threshold)

    if not st.session_state.messages:
        st.title("Deep Research Agent")
        st.markdown("Ask a broad research question. The agent checks it for ambiguity, plans searches, "
                    "reads the most relevant pages, reflects on gaps, and writes a cited report.")
        cols = st.columns(len(EXAMPLES))
        for col, example in zip(cols, EXAMPLES):
            if col.button(example, use_container_width=True):
                st.session_state.queued_prompt = example

    render_history()

    prompt = st.chat_input("What should I research?")
    if not prompt and st.session_state.queued_prompt:
        prompt, st.session_state.queued_prompt = st.session_state.queued_prompt, None
    if prompt:
        run_turn(prompt, graph)


main()
