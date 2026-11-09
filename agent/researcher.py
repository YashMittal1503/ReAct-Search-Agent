"""Factory for the inner ReAct research agent.

Two interchangeable engines share the exact same tools and contract (``{"messages": [...]}`` in,
messages out), so swapping them is a config flag (``RESEARCH_ENGINE``):

* ``deep`` — :func:`deepagents.create_deep_agent`: a batteries-included harness on top of LangGraph
  with a planning/todo tool, a virtual file system for notes, automatic context summarization and a
  ``task`` tool that delegates each sub-question to an isolated **source-reader sub-agent**.
* ``react`` — :func:`langchain.agents.create_agent`: a minimal tool-calling ReAct loop. Lower token
  overhead; useful for small-context or heavily rate-limited models (e.g. Groq free tier).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from agent.prompts import RESEARCHER_SYSTEM, SOURCE_READER_DESCRIPTION, SOURCE_READER_SYSTEM


def build_research_agent(
    *,
    engine: Literal["deep", "react"],
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    today: str,
) -> CompiledStateGraph:
    """Compile the research agent for one iteration (tools are bound to that iteration's budget)."""
    system_prompt = RESEARCHER_SYSTEM.format(today=today)

    if engine == "deep":
        from deepagents import create_deep_agent

        return create_deep_agent(
            model=model,
            tools=list(tools),
            system_prompt=system_prompt
            + "\n\nFor 3+ sub-questions you may delegate each one to the `source-reader` sub-agent "
            "via the `task` tool so they run in isolated contexts; then merge their findings.",
            subagents=[
                {
                    "name": "source-reader",
                    "description": SOURCE_READER_DESCRIPTION,
                    "system_prompt": SOURCE_READER_SYSTEM,
                    "tools": list(tools),
                }
            ],
            name="deep-researcher",
        )

    from langchain.agents import create_agent

    return create_agent(model, list(tools), system_prompt=system_prompt, name="react-researcher")
