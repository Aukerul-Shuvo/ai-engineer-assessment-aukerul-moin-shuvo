"""Contract tests for POST /ask.

The app starts with its real lifespan (real corpus, real wiring) and then the ask service is
swapped for one driven by a scripted model, so every test is offline and exact.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from pydantic import SecretStr

from app.api.cache import ResponseCache
from app.api.service import AskService
from app.api.sessions import SessionStore
from app.config import Settings
from app.graph.build import GraphDependencies, build_graph
from app.graph.schemas import GroundingVerdict, QueryPlan, RetrievalGrade, SubQuery
from app.llm.providers import ChatModels, Provider
from app.main import create_app
from tests.fakes import ScriptedChatModel

SUPER_BOWL = "super-bowl-50-000"


def scripted_service(
    app: FastAPI,
    *,
    structured: list[object],
    chat: list[AIMessage],
    models: bool = True,
    timeout_s: float = 30.0,
) -> ScriptedChatModel:
    """Replace the app's ask service with one whose graph runs a scripted model."""
    model = ScriptedChatModel(structured_script=structured, script=chat)
    chat_models = (
        ChatModels([Provider(name="fake", model_id="scripted", model=model)]) if models else None
    )
    graph = build_graph(
        GraphDependencies(
            models=chat_models,
            store=app.state.store,
            document_titles=[d.title for d in app.state.store.list_documents()],
        )
    )
    app.state.ask_service = AskService(
        graph=graph,
        sessions=SessionStore(ttl_s=60),
        cache=ResponseCache(ttl_s=60, max_size=16),
        max_question_chars=app.state.settings.max_question_chars,
        timeout_s=timeout_s,
        metrics=app.state.ask_metrics,
    )
    return model


def happy_path_script() -> tuple[list[object], list[AIMessage]]:
    return (
        [
            QueryPlan(
                intent="answerable",
                sub_queries=[SubQuery(id="q1", text="Who won Super Bowl 50?", source="dataset")],
            ),
            RetrievalGrade(relevant_ids=[SUPER_BOWL], sufficient=True),
            GroundingVerdict(supported=True),
        ],
        [AIMessage(content="The Denver Broncos won Super Bowl 50 [S1].")],
    )


# ---------------------------------------------------------------- happy path
def test_ask_returns_answer_sources_plan_and_meta(app: FastAPI, client: TestClient) -> None:
    structured, chat = happy_path_script()
    scripted_service(app, structured=structured, chat=chat)

    response = client.post("/ask", json={"question": "Who won Super Bowl 50?"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"] == "The Denver Broncos won Super Bowl 50 [S1]."
    assert body["citations"] == ["S1"]

    source = body["sources"][0]
    assert source["id"] == "S1"
    assert source["type"] == "dataset"
    assert source["title"] == "Super Bowl 50"
    assert source["url"] == "https://en.wikipedia.org/wiki/Super_Bowl_50"
    assert source["locator"] == {"file": "data/paragraphs.jsonl", "paragraph_id": SUPER_BOWL}
    assert source["retrieval"]["dense_rank"] == 1
    assert source["retrieval"]["dense_score"] > 0
    assert source["retrieval"]["rerank_score"] is None, "the reranker is off in tests"
    assert "Denver Broncos" in source["excerpt"]
    assert source["reference"].startswith("SQuAD v1.1 dev")

    assert body["plan"]["intent"] == "answerable"
    assert body["plan"]["sub_queries"][0]["source"] == "dataset"

    meta = body["meta"]
    assert meta["request_id"] == response.headers["x-request-id"]
    assert meta["grounded"] is True
    assert meta["degraded"] is False
    assert meta["retrieval_mode"] == "dense"
    assert meta["cached"] is False
    assert {"plan:fake", "grade:fake", "synth:fake", "ground:fake"} <= set(meta["providers"])
    assert meta["latency_ms"] >= 0


def test_out_of_scope_returns_direct_reply_without_sources(
    app: FastAPI, client: TestClient
) -> None:
    scripted_service(
        app,
        structured=[QueryPlan(intent="out_of_scope", direct_reply="I cover other topics.")],
        chat=[],
    )

    body = client.post("/ask", json={"question": "Weather in Doha?"}).json()

    assert body["answer"] == "I cover other topics."
    assert body["sources"] == [] and body["citations"] == []
    assert body["plan"]["intent"] == "out_of_scope"
    assert body["meta"]["grounded"] is None


def test_ungrounded_answer_carries_a_caveat(app: FastAPI, client: TestClient) -> None:
    structured, chat = happy_path_script()
    structured[-1] = GroundingVerdict(supported=False, unsupported_claims=["score"])
    structured.append(GroundingVerdict(supported=False, unsupported_claims=["score"]))
    chat.append(AIMessage(content="The Broncos won 30-10 [S1]."))
    scripted_service(app, structured=structured, chat=chat)

    body = client.post("/ask", json={"question": "Who won Super Bowl 50?"}).json()

    assert body["meta"]["grounded"] is False
    assert body["answer"].endswith("(Some statements could not be verified against the sources.)")


# ---------------------------------------------------------------- degraded mode
def test_degraded_mode_returns_sources_with_null_answer(app: FastAPI, client: TestClient) -> None:
    scripted_service(app, structured=[], chat=[], models=False)

    response = client.post("/ask", json={"question": "Denver Broncos"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] is None
    assert body["sources"], "retrieval still ran"
    assert body["meta"]["degraded"] is True
    reason = body["meta"]["degraded_reason"]
    assert "planner unavailable" in reason and "synthesis unavailable" in reason, (
        "both degradations are reported, not just the last one"
    )


# ---------------------------------------------------------------- validation
@pytest.mark.parametrize(
    "payload",
    [
        {"question": "   "},
        {"question": ""},
        {"question": "ok", "unexpected": 1},
        {"question": "ok", "session_id": "has space"},
        {"question": "bad\x00byte"},
        {},
    ],
)
def test_invalid_bodies_return_422_envelopes(client: TestClient, payload: dict[str, Any]) -> None:
    response = client.post("/ask", json=payload)

    assert response.status_code == 422
    assert response.json()["error"]["type"] == "validation_error"


def test_question_over_the_configured_limit_is_rejected(app: FastAPI, client: TestClient) -> None:
    scripted_service(app, structured=[], chat=[])
    too_long = "x" * (app.state.settings.max_question_chars + 1)

    response = client.post("/ask", json={"question": too_long})

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["type"] == "validation_error"
    assert str(app.state.settings.max_question_chars) in error["message"]


# ---------------------------------------------------------------- auth and rate limit
def test_api_key_is_enforced_when_configured(test_settings: Settings) -> None:
    settings = test_settings.model_copy(update={"api_key": SecretStr("s3cret")})
    app = create_app(settings)
    with TestClient(app, raise_server_exceptions=False) as client:
        structured, chat = happy_path_script()
        scripted_service(app, structured=structured, chat=chat)

        denied = client.post("/ask", json={"question": "Who won Super Bowl 50?"})
        wrong = client.post(
            "/ask", json={"question": "Who won Super Bowl 50?"}, headers={"X-API-Key": "nope"}
        )
        allowed = client.post(
            "/ask", json={"question": "Who won Super Bowl 50?"}, headers={"X-API-Key": "s3cret"}
        )

    assert denied.status_code == 401 and denied.json()["error"]["type"] == "unauthorized"
    assert wrong.status_code == 401
    assert allowed.status_code == 200


def test_rate_limit_returns_429_with_retry_after(test_settings: Settings) -> None:
    settings = test_settings.model_copy(update={"rate_limit": "2/minute"})
    app = create_app(settings)
    with TestClient(app, raise_server_exceptions=False) as client:
        scripted_service(app, structured=[], chat=[], models=False)

        first = client.post("/ask", json={"question": "Denver Broncos"})
        second = client.post("/ask", json={"question": "Denver Broncos"})
        third = client.post("/ask", json={"question": "Denver Broncos"})

    assert first.status_code == 200 and second.status_code == 200
    assert third.status_code == 429
    assert third.json()["error"]["type"] == "rate_limited"
    assert "retry-after" in third.headers


# ---------------------------------------------------------------- sessions and cache
def test_session_history_reaches_the_planner(app: FastAPI, client: TestClient) -> None:
    structured, chat = happy_path_script()
    structured2, chat2 = happy_path_script()
    model = scripted_service(app, structured=structured + structured2, chat=chat + chat2)

    client.post("/ask", json={"question": "Who won Super Bowl 50?", "session_id": "s1"})
    client.post("/ask", json={"question": "By how much?", "session_id": "s1"})

    # The planner's payload for the second question is the fifth model call
    # (plan, grade, synth, ground, plan). It must include the prior exchange.
    second_plan_messages = model.calls[4]
    texts = [getattr(m, "content", "") for m in second_plan_messages]
    assert "Who won Super Bowl 50?" in texts
    assert "The Denver Broncos won Super Bowl 50 [S1]." in texts
    assert texts[-1] == "By how much?"


def test_identical_stateless_questions_are_served_from_cache(
    app: FastAPI, client: TestClient
) -> None:
    structured, chat = happy_path_script()
    model = scripted_service(app, structured=structured, chat=chat)

    first = client.post("/ask", json={"question": "Who won Super Bowl 50?"}).json()
    second = client.post("/ask", json={"question": "  who won super bowl 50?  "}).json()

    assert first["meta"]["cached"] is False
    assert second["meta"]["cached"] is True
    assert second["answer"] == first["answer"]
    assert second["meta"]["request_id"] != first["meta"]["request_id"]
    assert not model.structured_script and not model.script, "no second run happened"


# ---------------------------------------------------------------- timeout
def test_timeout_maps_to_504(app: FastAPI, client: TestClient) -> None:
    class SlowGraph:
        async def ainvoke(self, input: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
            await asyncio.sleep(1)
            return {}

        def astream(self, input: dict[str, Any], **kwargs: Any) -> AsyncIterator[Any]:
            raise NotImplementedError

    app.state.ask_service = AskService(
        graph=SlowGraph(),
        sessions=SessionStore(ttl_s=60),
        cache=ResponseCache(ttl_s=60, max_size=4),
        max_question_chars=1000,
        timeout_s=0.05,
    )

    response = client.post("/ask", json={"question": "anything"})

    assert response.status_code == 504
    assert response.json()["error"]["type"] == "timeout"


# ---------------------------------------------------------------- streaming
def _parse_sse(text: str) -> Iterator[tuple[str, dict[str, Any]]]:
    # SSE permits CRLF or LF line endings; normalise so the split is exact either way.
    for block in text.replace("\r\n", "\n").strip().split("\n\n"):
        event, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
        if event and data is not None:
            yield event, data


def test_event_stream_reports_progress_and_ends_with_the_full_response(
    app: FastAPI, client: TestClient
) -> None:
    structured, chat = happy_path_script()
    scripted_service(app, structured=structured, chat=chat)

    with client.stream(
        "POST",
        "/ask",
        json={"question": "Who won Super Bowl 50?"},
        headers={"Accept": "text/event-stream"},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = list(_parse_sse(response.read().decode()))

    names = [name for name, _ in events]
    assert names == ["plan", "evidence", "answer", "grounding", "done"]
    assert events[0][1]["intent"] == "answerable"
    assert events[1][1]["count"] == 1
    assert events[2][1]["citations"] == ["S1"]
    done = events[-1][1]
    assert done["answer"] == "The Denver Broncos won Super Bowl 50 [S1]."
    assert done["sources"][0]["id"] == "S1"
    assert done["meta"]["grounded"] is True
