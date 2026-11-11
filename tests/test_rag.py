"""Post-retrieval RAG (#19): chunking, BM25, fusion, passage selection and the rag synthesis mode."""

from __future__ import annotations

from pathlib import Path

from tests.test_graph_flow import QUOTE, _collect, _graph
from tools.rag.bm25 import BM25, tokenize
from tools.rag.chunker import chunk_markdown
from tools.rag.embeddings import cosine, get_default_embedder
from tools.rag.retriever import HybridRetriever, rrf, select_passages


class KeywordEmbedder:
    """Deterministic fake: vector = keyword counts over a tiny vocabulary."""

    name = "fake-keywords"
    vocab = ("monolith", "microservice", "cost", "team", "battery")

    def _vec(self, text: str) -> list[float]:
        t = text.lower()
        return [float(t.count(w)) for w in self.vocab]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def test_chunker_respects_size_headings_and_overlap() -> None:
    body = " ".join(f"Sentence {i} talks about deployment cost number {i}." for i in range(40))
    doc = f"# Intro\nOpening line.\n\n## Costs\n{body}\n\n## Outlook\nFinal note."
    chunks = chunk_markdown("u", doc, max_chars=300, overlap=80)
    assert all(len(c.text) <= 300 for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].heading == "Intro" and chunks[-1].heading == "Outlook" and chunks[-1].text == "Final note."
    costs = [c for c in chunks if c.heading == "Costs"]
    assert len(costs) >= 3
    for a, b in zip(costs, costs[1:]):  # consecutive chunks share a trailing sentence
        assert a.text.split(". ")[-1].rstrip(".") in b.text
    assert "Final note." not in costs[-1].text  # no overlap across section boundaries


def test_chunker_hard_splits_giant_sentences_and_handles_small_docs() -> None:
    giant = "word " * 400
    chunks = chunk_markdown("u", giant, max_chars=200, overlap=0)
    assert len(chunks) >= 10 and all(len(c.text) <= 200 for c in chunks)
    assert [c.text for c in chunk_markdown("u", "Just one line.")] == ["Just one line."]
    assert chunk_markdown("u", "") == []


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------
def test_bm25_and_rrf() -> None:
    docs = [tokenize(t) for t in ("monolith deployment is simple", "microservices scale teams independently",
                                  "battery chemistry and solid electrolytes")]
    scores = BM25(docs).scores(tokenize("microservices teams"))
    assert scores[1] > 0 and scores[0] == scores[2] == 0
    assert rrf([[1, 0], [0]]) == {1: 1 / 61, 0: 1 / 62 + 1 / 61}
    assert cosine([1, 0], [1, 0]) == 1.0 and cosine([0, 0], [1, 1]) == 0.0


def test_hybrid_retriever_uses_dense_ranking_when_available() -> None:
    from tools.rag.chunker import Chunk

    chunks = [Chunk("a", 0, "Monolith teams move fast."), Chunk("a", 1, "Cost of microservice ops."),
              Chunk("b", 0, "Unrelated battery text.")]
    lexical = HybridRetriever(chunks)
    assert lexical.mode == "bm25" and lexical.rank("battery")[0][0] == 2
    hybrid = HybridRetriever(chunks, KeywordEmbedder())
    assert hybrid.mode == "hybrid (bm25 + fake-keywords)"
    qv = KeywordEmbedder().embed_queries(["monolith"])[0]
    assert hybrid.rank("monolith", qv)[0][0] == 0


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
PAGE_A = "# Microservices\n" + "\n\n".join(
    f"Paragraph {i}: microservices let teams deploy independently, but operational cost rises {i}%." for i in range(12))
PAGE_B = "# Batteries\n" + "\n\n".join(f"Paragraph {i}: solid electrolytes and dendrites in battery cells." for i in range(6))


def test_select_passages_budget_guarantee_neighbours_and_render() -> None:
    sel = select_passages({"https://a.com": PAGE_A, "https://b.com": PAGE_B},
                          ["microservices operational cost"], chunk_chars=200, overlap=0, top_k=2,
                          neighbors=1, budget_chars=1_200)
    assert sel.mode == "bm25" and sel.total_chunks > sel.selected_chunks > 0
    assert sel.selected_chars <= 1_200
    assert set(sel.passages) == {"https://a.com", "https://b.com"}  # every source keeps >= 1 chunk
    a_idx = [c.index for c in sel.passages["https://a.com"]]
    assert a_idx == sorted(a_idx) and len(a_idx) >= 2  # top hits plus neighbours, in document order
    rendered = sel.render("https://a.com")
    assert rendered.startswith("## Microservices") and "operational cost" in rendered
    if any(b - a > 1 for a, b in zip(a_idx, a_idx[1:])):
        assert "[…]" in rendered


def test_select_passages_hybrid_and_empty() -> None:
    sel = select_passages({"https://a.com": PAGE_A}, ["team deploy"], embedder=KeywordEmbedder(), chunk_chars=300)
    assert sel.mode.startswith("hybrid") and sel.passages["https://a.com"]
    empty = select_passages({}, ["q"])
    assert empty.total_chunks == 0 and empty.passages == {}


def test_unknown_embedding_model_falls_back_to_none() -> None:
    assert get_default_embedder("not-a-real/model-name") is None  # logged, no exception


# ---------------------------------------------------------------------------
# rag synthesis mode end-to-end (fakes) and in the eval harness
# ---------------------------------------------------------------------------
async def test_rag_mode_feeds_retrieved_passages_to_synthesizer() -> None:
    from agent.config import Settings
    from agent.graph import build_graph
    from agent.nodes import AgentDeps
    from agent.runner import ResearchSession, RunResult
    from schemas import TraceEvent
    from tests.test_graph_flow import FakeAgent, FakeLLM, FakeScraper, FakeSearcher

    for embedder, expected_mode in ((None, "bm25"), (KeywordEmbedder(), "hybrid")):
        llm = FakeLLM()
        settings = Settings(max_iterations=1, max_scrapes_per_iteration=3, synthesis_mode="rag", _env_file=None)
        deps = AgentDeps(settings=settings, llm_factory=lambda fast: None, structured=llm,  # type: ignore[arg-type,return-value]
                         searcher_factory=FakeSearcher, scraper_factory=FakeScraper,  # type: ignore[arg-type]
                         agent_factory=lambda **kw: FakeAgent(kw["tools"], False), today=lambda: "today",
                         embedder_factory=lambda e=embedder: e)
        items = await _collect(ResearchSession(build_graph(deps)),
                               "Pros and cons of microservices vs monoliths for startups in 2026")
        assert isinstance(items[-1], RunResult)
        assert "Relevant passages:" in llm.synthesis_prompt and QUOTE in llm.synthesis_prompt
        assert "Key points:" not in llm.synthesis_prompt and "SourceNotes" not in llm.calls  # no extraction calls
        rag_events = [e for e in items if isinstance(e, TraceEvent) and e.message.startswith("RAG: kept")]
        assert rag_events and expected_mode in rag_events[0].data["mode"]
        assert any(s.quote_verified for s in items[-1].report.sources)  # citations still verify on full text


async def test_rag_variant_replays_on_frozen_evidence(tmp_path: Path) -> None:
    from evals.variants import VARIANTS
    from tests.test_evals import GQ, _harness

    assert VARIANTS["rag"] == {"synthesis_mode": "rag"} and VARIANTS["rag_bm25"]["rag_embeddings"] == "off"
    base, _ = _harness(tmp_path)
    assert (await base.snapshot(GQ))["status"] == "ok"
    rag, rag_llm = _harness(tmp_path, "rag")
    rag.deps.embedder_factory = lambda: None  # offline: lexical retrieval
    row = await rag.replay(GQ)
    assert row["status"] == "ok" and row["nugget_recall"] == 0.5
    assert "Relevant passages:" in rag_llm.synthesis_prompt
