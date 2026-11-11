"""Local dense embeddings (fastembed + ONNX), with graceful fallback.

``BAAI/bge-small-en-v1.5`` (384-dim) runs on CPU through ONNX Runtime with no API calls or keys. The model
(~70 MB) is downloaded once on first use to fastembed's cache. If fastembed is missing, the
download fails, or embedding raises, :func:`get_default_embedder` returns ``None`` and retrieval
falls back to BM25 only. A RAG failure never breaks a research run.
"""

from __future__ import annotations

import logging
import threading
from typing import Protocol

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder(Protocol):
    """Anything that turns texts into vectors (queries and passages may be encoded differently)."""

    name: str

    def embed_queries(self, texts: list[str]) -> list[list[float]]: ...

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...


class FastEmbedEmbedder:
    """fastembed ``TextEmbedding`` wrapper (uses query/passage prefixes where the model defines them)."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from fastembed import TextEmbedding  # import lazily: heavy (onnxruntime)

        self.name = model_name
        self._model = TextEmbedding(model_name=model_name)
        self._lock = threading.Lock()  # ONNX sessions are not guaranteed thread-safe

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            return [list(map(float, v)) for v in self._model.query_embed(texts)]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            return [list(map(float, v)) for v in self._model.passage_embed(texts)]


_cache: dict[str, FastEmbedEmbedder | None] = {}
_cache_lock = threading.Lock()


def get_default_embedder(model_name: str = DEFAULT_MODEL) -> FastEmbedEmbedder | None:
    """Process-wide cached embedder, or ``None`` if it can't be loaded (logged once)."""
    with _cache_lock:
        if model_name not in _cache:
            try:
                _cache[model_name] = FastEmbedEmbedder(model_name)
                logger.info("Loaded embedding model %s", model_name)
            except Exception:  # noqa: BLE001 - missing package, offline download, ONNX error, ...
                logger.warning("Embedding model %s unavailable; RAG falls back to BM25 only", model_name,
                               exc_info=True)
                _cache[model_name] = None
        return _cache[model_name]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0
