"""Hybrid retrieval (BM25 + dense, fused with RRF) and passage selection for synthesis.

:func:`select_passages` turns the scraped pages of one research run into a compact set of
*relevant passages*:

1. **Chunk** every scraped page (:mod:`tools.rag.chunker`).
2. **Rank** chunks for each query: the research question plus every search query the agent ran, so
   each researched facet gets its own retrieval. BM25 and dense rankings are fused with
   Reciprocal Rank Fusion.
3. **Select** under a character budget, in priority order:
   a. the best chunk of *every* source, so no scraped source (and no minority viewpoint) is silently
      dropped;
   b. the top-k chunks per query;
   c. neighbouring chunks (±``neighbors``) of selected chunks, to restore context that chunking
      cut off ("this grew 40%").
4. **Render** each source's selected chunks in document order, marking gaps with "[…]".
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from tools.rag.bm25 import BM25, tokenize
from tools.rag.chunker import Chunk, chunk_markdown
from tools.rag.embeddings import Embedder, cosine

RRF_K = 60


def rrf(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion: ``sum(1 / (k + rank))`` over rankings (rank is 1-based).

    >>> rrf([[2, 0, 1], [0, 2]]) == {2: 1/61 + 1/62, 0: 1/62 + 1/61, 1: 1/63}
    True
    """
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] += 1.0 / (k + rank)
    return dict(fused)


class HybridRetriever:
    """Ranks a fixed set of chunks for arbitrary queries."""

    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None) -> None:
        self.chunks = chunks
        self.bm25 = BM25([tokenize(c.scoring_text) for c in chunks])
        self.embedder = embedder
        self.vectors: list[list[float]] | None = None
        if embedder is not None and chunks:
            self.vectors = embedder.embed_passages([c.scoring_text for c in chunks])

    @property
    def mode(self) -> str:
        return f"hybrid (bm25 + {self.embedder.name})" if self.vectors is not None and self.embedder else "bm25"

    def rank(self, query: str, query_vector: list[float] | None = None) -> list[tuple[int, float]]:
        """All chunk indices with a fused score, best first. Chunks matching nothing are excluded."""
        if not self.chunks:
            return []
        bm = self.bm25.scores(tokenize(query))
        rankings = [[i for i in sorted(range(len(bm)), key=lambda i: -bm[i]) if bm[i] > 0]]
        if self.vectors is not None and query_vector is not None:
            sims = [cosine(query_vector, v) for v in self.vectors]
            rankings.append(sorted(range(len(sims)), key=lambda i: -sims[i]))
        fused = rrf(rankings)
        return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


@dataclass
class PassageSelection:
    """Selected passages per source URL, plus statistics for tracing and evaluation."""

    passages: dict[str, list[Chunk]] = field(default_factory=dict)
    total_chunks: int = 0
    selected_chunks: int = 0
    selected_chars: int = 0
    sources: int = 0
    mode: str = "bm25"

    def render(self, url: str) -> str:
        """Selected chunks of one source in document order; '[…]' marks skipped text."""
        out: list[str] = []
        prev_index, prev_heading = None, None
        for c in self.passages.get(url, []):
            if prev_index is not None and c.index != prev_index + 1:
                out.append("[…]")
            if c.heading and c.heading != prev_heading:
                out.append(f"## {c.heading}")
            out.append(c.text)
            prev_index, prev_heading = c.index, c.heading
        return "\n".join(out)


def select_passages(
    pages: dict[str, str],
    queries: list[str],
    *,
    embedder: Embedder | None = None,
    chunk_chars: int = 900,
    overlap: int = 150,
    top_k: int = 4,
    neighbors: int = 1,
    budget_chars: int = 12_000,
) -> PassageSelection:
    """Pick the most relevant passages of ``pages`` (url -> markdown) for ``queries``."""
    chunks = [c for url, text in pages.items() if text for c in chunk_markdown(url, text, max_chars=chunk_chars,
                                                                               overlap=overlap)]
    selection = PassageSelection(total_chunks=len(chunks), sources=len(pages))
    if not chunks:
        return selection
    queries = list(dict.fromkeys(q.strip() for q in queries if q and q.strip()))
    retriever = HybridRetriever(chunks, embedder)
    selection.mode = retriever.mode
    qvecs = embedder.embed_queries(queries) if retriever.vectors is not None and embedder else [None] * len(queries)

    best_score: dict[int, float] = defaultdict(float)
    top_hits: list[int] = []
    for query, qvec in zip(queries, qvecs):
        ranked = retriever.rank(query, qvec)
        for idx, score in ranked:
            best_score[idx] = max(best_score[idx], score)
        top_hits.extend(idx for idx, _ in ranked[:top_k])

    by_url: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(chunks):
        by_url[c.url].append(i)
    position = {(c.url, c.index): i for i, c in enumerate(chunks)}

    # Priority 1: best chunk per source (falls back to the first chunk if nothing matched).
    guaranteed = [max(idxs, key=lambda i: (best_score.get(i, 0.0), -i)) for idxs in by_url.values()]
    # Priority 2: top-k per query, strongest first.
    top = sorted(dict.fromkeys(top_hits), key=lambda i: -best_score.get(i, 0.0))
    # Priority 3: neighbours of everything chosen so far.
    neighbours = [position[(chunks[i].url, chunks[i].index + d)]
                  for i in dict.fromkeys(guaranteed + top) for d in range(-neighbors, neighbors + 1)
                  if d and (chunks[i].url, chunks[i].index + d) in position]

    chosen: list[int] = []
    used = 0
    for i in dict.fromkeys(guaranteed + top + neighbours):
        size = len(chunks[i].text)
        if chosen and used + size > budget_chars:
            continue  # keep scanning: a smaller chunk may still fit
        chosen.append(i)
        used += size

    for i in sorted(chosen, key=lambda i: (chunks[i].url, chunks[i].index)):
        selection.passages.setdefault(chunks[i].url, []).append(chunks[i])
    selection.selected_chunks, selection.selected_chars = len(chosen), used
    return selection
