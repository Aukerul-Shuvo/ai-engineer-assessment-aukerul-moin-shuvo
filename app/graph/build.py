"""Wire the nodes into the graph.

The shape::

    START -> analyze_query
               |-- out_of_scope / chitchat -> respond_directly -> END
               '-- Send per ready sub-query -> retrieve_documents | superhero_agent
                                                   |
                                                collect  (fan-in)
                                                   |-- pending, wave < 2 -> resolve_dependencies
                                                   |      '-- Send per ready -> branches -> collect
                                                   '-- synthesize -> check_grounding
                                                          |-- unsupported, retry left -> synthesize
                                                          '-- END

``Send`` dispatches one branch invocation per sub-query in the same superstep, so branches run in
parallel and ``collect`` runs once when all of them finish. Their ``evidence`` lists merge through
the state reducer.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from app.graph.nodes.analyze_query import make_analyze_query
from app.graph.nodes.check_grounding import make_check_grounding
from app.graph.nodes.resolve_dependencies import make_resolve_dependencies
from app.graph.nodes.respond_directly import respond_directly
from app.graph.nodes.retrieve_documents import make_retrieve_documents
from app.graph.nodes.superhero_agent import make_superhero_agent
from app.graph.nodes.synthesize import make_synthesize
from app.graph.schemas import SubQuery
from app.graph.state import GraphState
from app.llm.providers import ChatModels
from app.retrieval.store import VectorStore

MAX_WAVES = 2


@dataclass
class GraphDependencies:
    """Everything the graph needs, built once at startup."""

    models: ChatModels | None
    store: VectorStore | None
    superhero_tools: Sequence[BaseTool] = field(default_factory=list)
    document_titles: Sequence[str] = field(default_factory=list)
    max_agent_steps: int = 4
    max_query_rewrites: int = 1
    max_regenerations: int = 1
    max_sub_queries: int = 6
    grade_top_k: int = 10
    evidence_per_sub_query: int = 6


def _as_node(fn: Callable[..., Awaitable[dict[str, Any]]]) -> Any:
    """Erase the node callable's type in one place.

    LangGraph's ``add_node`` overloads use a contravariant protocol that mypy cannot solve from a
    typed async callable, so every registration would need an ignore. The nodes themselves stay
    fully typed; only the handoff to LangGraph is untyped, and only here.
    """
    return fn


def build_graph(deps: GraphDependencies) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile the graph."""
    graph = StateGraph(GraphState)

    graph.add_node(
        "analyze_query",
        _as_node(
            make_analyze_query(
                deps.models, deps.document_titles, max_sub_queries=deps.max_sub_queries
            )
        ),
    )
    graph.add_node("respond_directly", _as_node(respond_directly))
    graph.add_node(
        "retrieve_documents",
        _as_node(
            make_retrieve_documents(
                deps.models,
                deps.store,
                grade_top_k=deps.grade_top_k,
                evidence_per_sub_query=deps.evidence_per_sub_query,
                max_query_rewrites=deps.max_query_rewrites,
            )
        ),
    )
    graph.add_node(
        "superhero_agent",
        _as_node(
            make_superhero_agent(
                deps.models, deps.superhero_tools, max_agent_steps=deps.max_agent_steps
            )
        ),
    )
    graph.add_node("collect", _as_node(_collect))
    graph.add_node("resolve_dependencies", _as_node(make_resolve_dependencies(deps.models)))
    graph.add_node("synthesize", _as_node(make_synthesize(deps.models)))
    graph.add_node("check_grounding", _as_node(make_check_grounding(deps.models)))

    graph.add_edge(START, "analyze_query")
    graph.add_conditional_edges(
        "analyze_query",
        _route_after_plan,
        ["respond_directly", "retrieve_documents", "superhero_agent"],
    )
    graph.add_edge("respond_directly", END)
    graph.add_edge("retrieve_documents", "collect")
    graph.add_edge("superhero_agent", "collect")
    graph.add_conditional_edges(
        "collect", _route_after_collect, ["resolve_dependencies", "synthesize"]
    )
    graph.add_conditional_edges(
        "resolve_dependencies",
        _dispatch_ready,
        ["retrieve_documents", "superhero_agent", "synthesize"],
    )
    graph.add_edge("synthesize", "check_grounding")
    graph.add_conditional_edges(
        "check_grounding",
        _make_route_after_grounding(deps.max_regenerations),
        ["synthesize", END],
    )
    return graph.compile()


async def _collect(state: GraphState) -> dict[str, Any]:
    """Fan-in point. Exists so the parallel branches have one successor."""
    return {}


def _sends_for(state: GraphState, ready: Sequence[SubQuery]) -> list[Send]:
    return [
        Send(
            "retrieve_documents" if sq.source == "dataset" else "superhero_agent",
            {"question": state["question"], "sub_query": sq},
        )
        for sq in ready
    ]


def _route_after_plan(state: GraphState) -> str | list[Send]:
    plan = state.get("plan")
    ready = state.get("ready", [])
    if plan is None or plan.intent != "answerable" or not ready:
        return "respond_directly"
    return _sends_for(state, ready)


def _route_after_collect(state: GraphState) -> str:
    if state.get("pending") and state.get("wave", 1) < MAX_WAVES:
        return "resolve_dependencies"
    return "synthesize"


def _dispatch_ready(state: GraphState) -> str | list[Send]:
    ready = state.get("ready", [])
    return _sends_for(state, ready) if ready else "synthesize"


def _make_route_after_grounding(max_regenerations: int) -> Callable[[GraphState], str]:
    def route(state: GraphState) -> str:
        if state.get("grounded") is False and state.get("regenerations", 0) <= max_regenerations:
            return "synthesize"
        return END

    return route
