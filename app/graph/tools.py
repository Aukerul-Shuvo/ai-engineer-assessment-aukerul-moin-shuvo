"""LangChain tool adapters over the plain tool functions, plus a per-run call budget.

The superhero agent needs its tools as LangChain ``BaseTool`` objects. ``build_superhero_tools``
wraps the functions in ``app.tools.superhero`` without changing them; each returns compact JSON
text, which is what a ``ToolMessage`` carries back to the model.

``with_call_budget`` caps how many tool calls one agent run may make. When the budget is spent,
further calls return a ``ToolError`` telling the model to answer with what it has. That ends the
loop gracefully instead of hitting the graph's hard recursion limit and losing the evidence
gathered so far.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from app.tools.common import ToolError
from app.tools.superhero import SuperheroClient, get_superhero, search_superheroes


class SearchSuperheroesArgs(BaseModel):
    """Arguments of ``search_superheroes``."""

    name: str = Field(min_length=1, description="Character name or alias, e.g. 'batman'.")
    limit: int = Field(default=5, ge=1, le=20, description="Maximum matches to return.")


class GetSuperheroArgs(BaseModel):
    """Arguments of ``get_superhero``."""

    hero_id: str = Field(min_length=1, description="An id returned by search_superheroes.")


def build_superhero_tools(client: SuperheroClient) -> list[BaseTool]:
    """The two superhero tools bound to a client, as LangChain tools returning JSON text."""

    async def _search(name: str, limit: int = 5) -> str:
        return (await search_superheroes(client, name, limit)).model_dump_json()

    async def _get(hero_id: str) -> str:
        return (await get_superhero(client, hero_id)).model_dump_json()

    return [
        StructuredTool.from_function(
            coroutine=_search,
            name="search_superheroes",
            description=(
                "Find comic-book characters by name. Returns id, name, full name, publisher, "
                "alignment and power stats for each match, plus a note when several share the name."
            ),
            args_schema=SearchSuperheroesArgs,
        ),
        StructuredTool.from_function(
            coroutine=_get,
            name="get_superhero",
            description=(
                "Full record for one character id: biography, aliases, appearance, work, "
                "connections and image."
            ),
            args_schema=GetSuperheroArgs,
        ),
    ]


def with_call_budget(tools: Sequence[BaseTool], max_calls: int) -> list[BaseTool]:
    """Wrap tools so that, together, they accept at most ``max_calls`` calls per run."""
    remaining = {"calls": max_calls}

    def wrap(tool: BaseTool) -> BaseTool:
        async def _guarded(**kwargs: Any) -> str:
            if remaining["calls"] <= 0:
                return ToolError(
                    error="Tool call budget for this question is exhausted.",
                    hint="Answer with the facts already retrieved.",
                ).model_dump_json()
            remaining["calls"] -= 1
            result = await tool.ainvoke(kwargs)
            return result if isinstance(result, str) else str(result)

        return StructuredTool.from_function(
            coroutine=_guarded,
            name=tool.name,
            description=tool.description,
            args_schema=tool.args_schema,
        )

    return [wrap(tool) for tool in tools]
