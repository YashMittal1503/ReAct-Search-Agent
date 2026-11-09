"""Factory for the inner ReAct research agent.

Two interchangeable engines share the exact same tools and contract (``{"messages": [...]}`` in,
messages out), so swapping them is a config flag (``RESEARCH_ENGINE``):

* ``deep`` — :func:`deepagents.create_deep_agent`: a batteries-included harness on top of LangGraph
  with a ``task`` tool that delegates each sub-question to an isolated **source-reader sub-agent**.
* ``react`` — :func:`langchain.agents.create_agent`: a minimal tool-calling ReAct loop.

Context-size control (applied to both engines and to sub-agents): LangChain's
:class:`ContextEditingMiddleware` + :class:`ClearToolUsesEdit` replace older tool results with a short
placeholder once the conversation grows past a token threshold. The full data is already stored in
the :class:`~tools.url_registry.URLRegistry`, so the agent loses nothing it needs, but each ReAct turn
stops resending every previous search result.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from langchain.agents.middleware import AgentMiddleware, ClearToolUsesEdit, ContextEditingMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from agent.prompts import RESEARCHER_SYSTEM, SOURCE_READER_DESCRIPTION, SOURCE_READER_SYSTEM

#: Start clearing old tool results once the conversation exceeds this many (approx.) tokens…
CLEAR_TOOL_RESULTS_TRIGGER = 3_500
#: …keeping this many most-recent tool results verbatim.
KEEP_RECENT_TOOL_RESULTS = 2
CLEARED_PLACEHOLDER = (
    "[earlier tool output cleared to save context — its sources are stored in the session registry "
    "and will be available to the evaluator and the final report]"
)


def context_middleware() -> list[AgentMiddleware]:
    """Middleware stack that keeps every model call small (fresh instances per agent)."""
    return [
        ContextEditingMiddleware(edits=[ClearToolUsesEdit(
            trigger=CLEAR_TOOL_RESULTS_TRIGGER,
            keep=KEEP_RECENT_TOOL_RESULTS,
            placeholder=CLEARED_PLACEHOLDER,
        )]),
    ]


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
            middleware=context_middleware(),
            subagents=[
                {
                    "name": "source-reader",
                    "description": SOURCE_READER_DESCRIPTION,
                    "system_prompt": SOURCE_READER_SYSTEM,
                    "tools": list(tools),
                    "middleware": context_middleware(),
                }
            ],
            name="deep-researcher",
        )

    from langchain.agents import create_agent

    return create_agent(model, list(tools), system_prompt=system_prompt, middleware=context_middleware(),
                        name="react-researcher")
