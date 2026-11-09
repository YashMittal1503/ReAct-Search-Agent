"""Typed trace events streamed from the graph to the UI (Thought / Action / Observation)."""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

EventKind = Literal[
    "status",       # phase transitions ("Planning search queries...")
    "thought",      # model reasoning / decisions
    "action",       # a tool call about to run
    "observation",  # a tool result
    "dedup",        # duplicate URL/snippet filtered by the URLRegistry
    "skip",         # scrape skipped by the relevance / already-scraped guard
    "failover",     # LLM router rotated a key or switched provider/model (rate limit, outage, ...)
    "error",        # recoverable error (failed fetch, retry, ...)
]


class TraceEvent(BaseModel):
    """A single step in the agent's reasoning trace."""

    kind: EventKind
    node: str = Field(description="Graph node or tool that emitted the event.")
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    ts: float = Field(default_factory=time.time)

    @property
    def icon(self) -> str:
        """Emoji used by the Streamlit trace renderer."""
        return {
            "status": "🧭",
            "thought": "💭",
            "action": "🛠️",
            "observation": "👁️",
            "dedup": "♻️",
            "skip": "⏭️",
            "failover": "🔀",
            "error": "⚠️",
        }[self.kind]
