"""Deep Research ReAct agent: LangGraph workflow + deepagents research engine.

Public names are resolved lazily so that low-level subpackages (e.g. ``agent.resilience``) can be
imported by ``tools`` without pulling in the whole graph and creating import cycles.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "AgentDeps": "agent.nodes",
    "ResearchSession": "agent.runner",
    "RunResult": "agent.runner",
    "Settings": "agent.config",
    "get_settings": "agent.config",
    "build_graph": "agent.graph",
    "default_graph": "agent.graph",
    "ModelRouter": "agent.resilience.router",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'agent' has no attribute {name!r}")
