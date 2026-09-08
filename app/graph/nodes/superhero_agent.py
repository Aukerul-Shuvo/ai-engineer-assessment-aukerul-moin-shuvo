"""superhero_agent: a small ReAct agent for one superhero sub-query.

Agent, not pipeline, because navigating the API is unpredictable: a name may match several
characters, an alias may be needed, details may require a second call. The agent gets the two
superhero tools under a call budget, runs until it answers or the budget ends, and its text reply
is discarded. Evidence is built from what the tools actually returned, parsed out of the
``ToolMessage`` contents, so the response can only cite characters that were really fetched.

Without tools (no token configured) the branch records a note and returns no evidence.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import BaseTool

from app.api.errors import UpstreamUnavailableError
from app.graph import prompts
from app.graph.nodes.common import message_text
from app.graph.schemas import Evidence
from app.graph.state import BranchInput
from app.graph.tools import with_call_budget
from app.llm.providers import ChatModels, call_with_failover

log = structlog.get_logger(__name__)

_STAT_KEYS = ("intelligence", "strength", "speed", "durability", "power", "combat")


def make_superhero_agent(
    models: ChatModels | None, tools: Sequence[BaseTool], *, max_agent_steps: int
) -> Callable[[BranchInput], Awaitable[dict[str, Any]]]:
    """Build the node."""

    async def superhero_agent(branch: BranchInput) -> dict[str, Any]:
        sub_query = branch["sub_query"]
        if not tools:
            return {
                "notes": [f"{sub_query.id}: the superhero source is not configured"],
                "evidence": [],
            }
        if models is None:
            return {"notes": [f"{sub_query.id}: no model to drive the superhero lookup"]}

        budgeted = with_call_budget(tools, max_agent_steps)
        question = sub_query.text
        if sub_query.hero_name:
            question += f"\n(Character to look up: {sub_query.hero_name})"
        try:
            result, provider = await call_with_failover(
                models.agents(budgeted, prompts.SUPERHERO_AGENT),
                {"messages": [HumanMessage(question)]},
                config={"recursion_limit": 2 * max_agent_steps + 4},
            )
        except UpstreamUnavailableError as exc:
            return {"notes": [f"{sub_query.id}: superhero lookup failed ({exc.describe()})"]}

        evidence, errors = evidence_from_messages(sub_query.id, result.get("messages", []))
        notes = [f"{sub_query.id}: {error}" for error in errors]
        log.info("superhero_agent_done", sub_query=sub_query.id, heroes=len(evidence))
        return {"evidence": evidence, "notes": notes, "providers": [f"agent:{provider}"]}

    return superhero_agent


def evidence_from_messages(
    sub_query_id: str, messages: Sequence[Any]
) -> tuple[list[Evidence], list[str]]:
    """Turn every tool result in an agent transcript into evidence, one item per character.

    A full record from ``get_superhero`` replaces the summary of the same character from a
    search. Tool errors are reported as notes rather than evidence.
    """
    by_id: dict[str, Evidence] = {}
    errors: list[str] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        try:
            payload = json.loads(message_text(message))
        except ValueError:
            continue
        if not isinstance(payload, dict):
            continue
        if "error" in payload and "results" not in payload:
            errors.append(str(payload["error"]))
            continue
        if "results" in payload:
            for record in payload.get("results", []):
                by_id.setdefault(str(record["id"]), _hero_evidence(sub_query_id, record))
            if payload.get("count") == 0 and payload.get("note"):
                errors.append(str(payload["note"]))
        elif "id" in payload and "powerstats" in payload:
            by_id[str(payload["id"])] = _hero_evidence(sub_query_id, payload)
    return list(by_id.values()), errors


def _hero_evidence(sub_query_id: str, record: dict[str, Any]) -> Evidence:
    hero_id = str(record["id"])
    name = str(record.get("name", "Unknown"))
    full_name = record.get("full_name")
    title = f"{name} ({full_name})" if full_name else name
    return Evidence(
        sub_query_id=sub_query_id,
        kind="superhero",
        title=title,
        reference=f"Superhero API - character id {hero_id} - GET /api/{{token}}/{hero_id}",
        url=f"https://superheroapi.com/api/{{token}}/{hero_id}",
        excerpt=describe_hero(record),
        data=record,
    )


def describe_hero(record: dict[str, Any]) -> str:
    """A compact factual paragraph about one character, for the model and for excerpts."""
    parts: list[str] = []
    name = record.get("name", "Unknown")
    identity = f"{name}" + (f", real name {record['full_name']}" if record.get("full_name") else "")
    publisher = record.get("publisher")
    alignment = record.get("alignment")
    parts.append(
        identity
        + (f", published by {publisher}" if publisher else "")
        + (f", alignment {alignment}" if alignment else "")
        + "."
    )
    stats = record.get("powerstats") or {}
    rendered = ", ".join(
        f"{key} {stats[key] if stats.get(key) is not None else 'unknown'}"
        for key in _STAT_KEYS
        if key in stats
    )
    if rendered:
        parts.append(f"Power stats: {rendered}.")
    for label, key in (
        ("First appearance", "first_appearance"),
        ("Place of birth", "place_of_birth"),
        ("Occupation", "occupation"),
        ("Base", "base"),
        ("Affiliations", "group_affiliation"),
        ("Relatives", "relatives"),
        ("Gender", "gender"),
        ("Race", "race"),
    ):
        if record.get(key):
            parts.append(f"{label}: {record[key]}.")
    if record.get("aliases"):
        parts.append(f"Aliases: {', '.join(record['aliases'])}.")
    if record.get("height"):
        parts.append(f"Height: {' / '.join(record['height'])}.")
    if record.get("weight"):
        parts.append(f"Weight: {' / '.join(record['weight'])}.")
    return " ".join(parts)
