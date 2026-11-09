"""LangGraph wiring for the Deep Research workflow.

::

    START -> assess_clarity --(ambiguous)--> ask_user --(interrupt / resume)--> assess_clarity
                   |
                   +--(clear)--> plan -> research -> evaluate --(gaps & iter < max)--> plan
                                                          |
                                                          +--(sufficient | max iters)--> synthesize -> END
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.config import Settings, get_settings
from agent.nodes import AgentDeps, ResearchNodes, with_router_events
from agent.state import ResearchState


def build_graph(
    deps: AgentDeps | None = None,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Compile the research graph.

    Args:
        deps: Injected dependencies (defaults to real clients built from settings).
        checkpointer: Persistence for interrupts/resume. Defaults to an in-process
            :class:`InMemorySaver`; swap for ``SqliteSaver``/``PostgresSaver`` in production.
    """
    deps = deps or AgentDeps(settings=get_settings())
    nodes = ResearchNodes(deps)

    g = StateGraph(ResearchState)
    # Every LLM-using node streams router failover events (key rotation, provider switches) to the UI.
    g.add_node("assess_clarity", with_router_events(nodes.assess_clarity))
    g.add_node("ask_user", nodes.ask_user)
    g.add_node("plan", with_router_events(nodes.plan))
    g.add_node("research", with_router_events(nodes.research))
    g.add_node("evaluate", with_router_events(nodes.evaluate))
    g.add_node("synthesize", with_router_events(nodes.synthesize))

    g.add_edge(START, "assess_clarity")
    g.add_conditional_edges("assess_clarity", nodes.route_after_assess, ["ask_user", "plan"])
    g.add_edge("ask_user", "assess_clarity")
    g.add_edge("plan", "research")
    g.add_edge("research", "evaluate")
    g.add_conditional_edges("evaluate", nodes.route_after_evaluate, ["plan", "synthesize"])
    g.add_edge("synthesize", END)

    return g.compile(checkpointer=checkpointer or InMemorySaver(), name="deep-research-agent")


def default_graph(settings: Settings | None = None) -> CompiledStateGraph:
    """Convenience constructor used by the UI and CLI."""
    return build_graph(AgentDeps(settings=settings or get_settings()))
