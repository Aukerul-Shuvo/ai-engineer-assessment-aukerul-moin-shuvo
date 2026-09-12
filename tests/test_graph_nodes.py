from pathlib import Path

import httpx
import pytest
import respx
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool

from app.graph.build import GraphDependencies
from app.graph.nodes.analyze_query import make_analyze_query, sanitize_plan
from app.graph.nodes.check_grounding import make_check_grounding
from app.graph.nodes.retrieve_documents import make_retrieve_documents
from app.graph.nodes.superhero_agent import evidence_from_messages, make_superhero_agent
from app.graph.nodes.synthesize import clean_citations, make_synthesize
from app.graph.schemas import Evidence, GroundingVerdict, QueryPlan, RetrievalGrade, SubQuery
from app.graph.tools import build_superhero_tools, with_call_budget
from app.llm.providers import ChatModels, Provider
from app.tools.circuit_breaker import CircuitBreaker
from app.tools.superhero import SuperheroClient
from tests.fakes import FakeEmbedder, ScriptedChatModel
from tests.fakes import build_test_corpus as build_corpus
from tests.fakes import load_test_store as load
from tests.test_superhero import BASE, BATMEN, TOKEN

TITLES = ["Super_Bowl_50", "Nikola_Tesla", "Oxygen"]


def models_from(script: list[object]) -> tuple[ChatModels, ScriptedChatModel]:
    model = ScriptedChatModel(script=script)
    return ChatModels([Provider(name="fake", model_id="scripted", model=model)]), model


def evidence(sub_query_id: str, title: str, text: str) -> Evidence:
    return Evidence(
        sub_query_id=sub_query_id,
        kind="dataset",
        title=title,
        reference=f"ref {title}",
        url=None,
        excerpt=text,
    )


def branch(sub_query: SubQuery) -> dict[str, object]:
    return {"question": "q", "sub_query": sub_query}


# ---------------------------------------------------------------- analyze_query
async def test_analyze_query_splits_ready_and_pending_and_records_provider() -> None:
    plan = QueryPlan(
        intent="answerable",
        sub_queries=[
            SubQuery(id="q1", text="Where is the Amazon rainforest?", source="dataset"),
            SubQuery(
                id="q2",
                text="Name a superhero from <country from q1>",
                source="superhero",
                depends_on=["q1"],
            ),
        ],
    )
    models, _ = models_from([plan])
    node = make_analyze_query(models, TITLES, max_sub_queries=6)

    out = await node({"question": "Where is the Amazon, and name a superhero from there?"})

    assert [sq.id for sq in out["ready"]] == ["q1"]
    assert [sq.id for sq in out["pending"]] == ["q2"]
    assert out["providers"] == ["plan:fake"]
    assert out["degraded"] is False


async def test_analyze_query_degrades_to_a_dataset_search_without_models() -> None:
    node = make_analyze_query(None, TITLES, max_sub_queries=6)

    out = await node({"question": "Who won Super Bowl 50?"})

    assert out["degraded"] is True
    assert out["ready"][0].source == "dataset"
    assert out["ready"][0].text == "Who won Super Bowl 50?"


async def test_analyze_query_degrades_when_every_provider_fails() -> None:
    model = ScriptedChatModel(fail_with=RuntimeError("quota"))
    models = ChatModels([Provider(name="fake", model_id="x", model=model)])
    node = make_analyze_query(models, TITLES, max_sub_queries=6)

    out = await node({"question": "anything"})

    assert out["degraded"] is True
    assert "quota" in out["degraded_reason"]


def test_sanitize_plan_fixes_ids_dependencies_and_cycles() -> None:
    plan = QueryPlan(
        intent="answerable",
        sub_queries=[
            SubQuery(id="q1", text="a", source="dataset", depends_on=["q2"]),
            SubQuery(id="q1", text="b", source="superhero", depends_on=["q1", "nope"]),
            SubQuery(id="q3", text="   ", source="dataset"),
        ],
    )

    cleaned = sanitize_plan(plan, max_sub_queries=6)

    assert [sq.id for sq in cleaned.sub_queries] == ["q1", "q2"], "duplicate renamed, blank gone"
    assert all(sq.depends_on == [] for sq in cleaned.sub_queries), "cycle broken"


def test_sanitize_plan_caps_the_number_of_sub_queries() -> None:
    plan = QueryPlan(
        intent="answerable",
        sub_queries=[SubQuery(id=f"q{i}", text=f"t{i}", source="dataset") for i in range(10)],
    )

    assert len(sanitize_plan(plan, max_sub_queries=3).sub_queries) == 3


@pytest.mark.parametrize("blank", ["", "   "])
def test_sanitize_plan_with_only_blank_sub_queries_becomes_out_of_scope(blank: str) -> None:
    plan = QueryPlan(
        intent="answerable", sub_queries=[SubQuery(id="q1", text=blank, source="dataset")]
    )

    assert sanitize_plan(plan, max_sub_queries=6).intent == "out_of_scope"


# ---------------------------------------------------------------- retrieve_documents
async def test_retrieve_documents_keeps_graded_passages_with_provenance(tmp_path: Path) -> None:
    embedder = FakeEmbedder(16)
    store = load(await build_corpus(tmp_path, embedder=embedder), embedder=embedder)
    models, _ = models_from([RetrievalGrade(relevant_ids=["super-bowl-50-000"], sufficient=True)])
    node = make_retrieve_documents(
        models, store, grade_top_k=4, evidence_per_sub_query=6, max_query_rewrites=1
    )

    out = await node(branch(SubQuery(id="q1", text="Who won Super Bowl 50?", source="dataset")))

    assert [e.locator["paragraph_id"] for e in out["evidence"]] == ["super-bowl-50-000"]
    item = out["evidence"][0]
    assert item.kind == "dataset"
    assert item.url == "https://en.wikipedia.org/wiki/Super_Bowl_50"
    assert item.retrieval["dense_rank"] == 1
    assert item.retrieval["rerank_score"] is None
    assert out["providers"] == ["grade:fake"]
    assert out["notes"] == []


async def test_retrieve_documents_rewrites_once_when_insufficient(tmp_path: Path) -> None:
    embedder = FakeEmbedder(16)
    store = load(await build_corpus(tmp_path, embedder=embedder), embedder=embedder)
    models, model = models_from(
        [
            RetrievalGrade(relevant_ids=[], sufficient=False),
            {"query": "Tesla Morgan Wardenclyffe transmitter funds"},
            RetrievalGrade(relevant_ids=["nikola-tesla-000"], sufficient=True),
        ]
    )
    node = make_retrieve_documents(
        models, store, grade_top_k=4, evidence_per_sub_query=6, max_query_rewrites=1
    )

    out = await node(
        branch(SubQuery(id="q1", text="What did Tesla ask Morgan for?", source="dataset"))
    )

    assert out["providers"] == ["grade:fake", "rewrite:fake", "grade:fake"]
    assert any("rewritten" in note for note in out["notes"])
    assert [e.locator["paragraph_id"] for e in out["evidence"]] == ["nikola-tesla-000"]
    assert not model.script, "the whole script was consumed"


async def test_retrieve_documents_without_models_returns_top_passages_ungraded(
    tmp_path: Path,
) -> None:
    embedder = FakeEmbedder(16)
    store = load(await build_corpus(tmp_path, embedder=embedder), embedder=embedder)
    node = make_retrieve_documents(
        None, store, grade_top_k=4, evidence_per_sub_query=2, max_query_rewrites=1
    )

    # Both Super Bowl paragraphs rank above the rest; the cap keeps the top two.
    out = await node(branch(SubQuery(id="q1", text="Panthers Super Bowl", source="dataset")))

    assert len(out["evidence"]) == 2
    assert any("not graded" in note for note in out["notes"])


# ---------------------------------------------------------------- superhero_agent
@respx.mock
async def test_superhero_agent_builds_evidence_from_tool_results() -> None:
    respx.get(f"{BASE}/{TOKEN}/search/batman").mock(
        return_value=httpx.Response(200, json={"response": "success", "results": BATMEN})
    )
    respx.get(f"{BASE}/{TOKEN}/70").mock(
        return_value=httpx.Response(200, json={"response": "success", **BATMEN[1]})
    )
    async with httpx.AsyncClient(follow_redirects=True) as http:
        client = SuperheroClient(http, base_url=BASE, token=TOKEN, breaker=CircuitBreaker())
        models, _ = models_from(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "search_superheroes", "args": {"name": "batman"}, "id": "c1"}
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "get_superhero", "args": {"hero_id": "70"}, "id": "c2"}],
                ),
                AIMessage(content="Batman, Bruce Wayne, DC Comics."),
            ]
        )
        node = make_superhero_agent(models, build_superhero_tools(client), max_agent_steps=4)

        out = await node(
            branch(
                SubQuery(
                    id="q2", text="How strong is Batman?", source="superhero", hero_name="Batman"
                )
            )
        )

    ids = sorted(e.data["id"] for e in out["evidence"])
    assert ids == ["69", "70", "71"], "one evidence item per character the tools returned"
    bruce = next(e for e in out["evidence"] if e.data["id"] == "70")
    assert bruce.data["first_appearance"] == "Detective Comics #27", "full record won"
    assert "{token}" in bruce.url and TOKEN not in bruce.url
    assert "strength 26" in bruce.excerpt
    assert out["providers"] == ["agent:fake"]


@respx.mock
async def test_superhero_agent_budget_stops_the_loop_gracefully() -> None:
    route = respx.get(f"{BASE}/{TOKEN}/search/batman").mock(
        return_value=httpx.Response(200, json={"response": "success", "results": BATMEN})
    )
    async with httpx.AsyncClient(follow_redirects=True) as http:
        client = SuperheroClient(http, base_url=BASE, token=TOKEN, breaker=CircuitBreaker())
        call = {"name": "search_superheroes", "args": {"name": "batman"}}
        models, _ = models_from(
            [
                AIMessage(content="", tool_calls=[{**call, "id": "c1"}]),
                AIMessage(content="", tool_calls=[{**call, "id": "c2"}]),
                AIMessage(content="done"),
            ]
        )
        node = make_superhero_agent(models, build_superhero_tools(client), max_agent_steps=1)

        out = await node(branch(SubQuery(id="q1", text="Batman?", source="superhero")))

    assert route.call_count == 1, "the second call was refused by the budget"
    assert len(out["evidence"]) == 3
    assert any("budget" in note for note in out["notes"])


async def test_superhero_agent_without_tools_leaves_a_note() -> None:
    models, _ = models_from([])
    node = make_superhero_agent(models, [], max_agent_steps=4)

    out = await node(branch(SubQuery(id="q1", text="Batman?", source="superhero")))

    assert out["evidence"] == []
    assert "not configured" in out["notes"][0]


def test_evidence_from_messages_reports_tool_errors_and_empty_searches() -> None:
    messages = [
        ToolMessage(content='{"error": "Superhero API rate limit reached"}', tool_call_id="1"),
        ToolMessage(
            content=(
                '{"query": "nobody", "count": 0, "results": [], '
                '"note": "No character named nobody."}'
            ),
            tool_call_id="2",
        ),
        ToolMessage(content="not json", tool_call_id="3"),
    ]

    items, errors = evidence_from_messages("q1", messages)

    assert items == []
    assert errors == ["Superhero API rate limit reached", "No character named nobody."]


async def test_with_call_budget_is_shared_across_tools() -> None:
    async def a() -> str:
        return "a"

    async def b() -> str:
        return "b"

    tools = with_call_budget(
        [
            StructuredTool.from_function(coroutine=a, name="a", description="a"),
            StructuredTool.from_function(coroutine=b, name="b", description="b"),
        ],
        max_calls=1,
    )

    assert await tools[0].ainvoke({}) == "a"
    assert "budget" in await tools[1].ainvoke({})


# ---------------------------------------------------------------- synthesize
async def test_synthesize_labels_evidence_deterministically_and_cleans_citations() -> None:
    plan = QueryPlan(
        intent="answerable",
        sub_queries=[
            SubQuery(id="q1", text="a", source="dataset"),
            SubQuery(id="q2", text="b", source="superhero"),
        ],
    )
    models, model = models_from([AIMessage(content="Fact one [S1]. Fact two [S2]. Bogus [S7].")])
    node = make_synthesize(models)
    state = {
        "question": "compound?",
        "plan": plan,
        # Arrives in the "wrong" order, as parallel branches may finish in any order.
        "evidence": [evidence("q2", "Flash", "speed 100"), evidence("q1", "Kangaroo", "33 days")],
    }

    out = await node(state)

    assert [e.title for e in out["cited_evidence"]] == ["Kangaroo", "Flash"], "plan order wins"
    assert out["answer"] == "Fact one [S1]. Fact two [S2]. Bogus ."
    assert out["citations"] == ["S1", "S2"]
    assert out["providers"] == ["synth:fake"]
    assert "[S1] Kangaroo (dataset): 33 days" in model.calls[0][0].content


async def test_synthesize_without_evidence_answers_honestly_without_a_model_call() -> None:
    models, model = models_from([])
    node = make_synthesize(models)

    out = await node(
        {
            "question": "q",
            "plan": None,
            "evidence": [],
            "caveats": ["the superhero source is not configured"],
        }
    )

    assert out["answer"].startswith("I could not find information")
    assert "Note: the superhero source is not configured." in out["answer"]
    assert out["citations"] == []
    assert model.calls == []


async def test_synthesize_marks_degraded_when_all_providers_fail() -> None:
    model = ScriptedChatModel(fail_with=RuntimeError("down"))
    models = ChatModels([Provider(name="fake", model_id="x", model=model)])
    node = make_synthesize(models)

    out = await node({"question": "q", "plan": None, "evidence": [evidence("q1", "T", "x")]})

    assert out["answer"] is None
    assert out["degraded"] is True
    assert out["cited_evidence"], "sources survive so the API can return them"


def test_clean_citations_keeps_order_and_drops_unknown_labels() -> None:
    text, used = clean_citations("A [S3] B [S1] C [S3] D [S9]", {"S1", "S2", "S3"})

    assert text == "A [S3] B [S1] C [S3] D"
    assert used == ["S1", "S3"]


# ---------------------------------------------------------------- check_grounding
async def test_check_grounding_records_issues_and_counts_regenerations() -> None:
    models, _ = models_from(
        [GroundingVerdict(supported=False, unsupported_claims=["Fact two is wrong"])]
    )
    node = make_check_grounding(models)

    out = await node(
        {"answer": "Fact one [S1]. Fact two [S1].", "cited_evidence": [evidence("q1", "T", "x")]}
    )

    assert out["grounded"] is False
    assert out["grounding_issues"] == ["Fact two is wrong"]
    assert out["regenerations"] == 1


async def test_check_grounding_skips_when_there_is_nothing_to_check() -> None:
    models, model = models_from([])

    out = await make_check_grounding(models)({"answer": None, "cited_evidence": []})

    assert out["grounded"] is None
    assert model.calls == []


def test_graph_dependencies_defaults_match_settings_defaults() -> None:
    deps = GraphDependencies(models=None, store=None)

    assert (deps.max_agent_steps, deps.max_query_rewrites, deps.max_regenerations) == (4, 1, 1)
    assert (deps.max_sub_queries, deps.grade_top_k, deps.evidence_per_sub_query) == (6, 10, 6)


async def test_routine_notes_stay_out_of_the_answer_but_caveats_are_appended() -> None:
    """A rewritten query is the operator's business; a missing source is the reader's."""
    models, _ = models_from(["The Denver Broncos won [S1]."])
    node = make_synthesize(models)

    out = await node(
        {
            "question": "Who won and how strong is Batman?",
            "plan": None,
            "evidence": [evidence("q1", "Super Bowl 50", "The Broncos won.")],
            "notes": [
                "q1: query rewritten once",
                "q1: relevance grading unavailable, kept the top passages ungraded",
            ],
            "caveats": ["the superhero source is not configured"],
        }
    )

    assert out["answer"] == (
        "The Denver Broncos won [S1]. Note: the superhero source is not configured."
    )
    assert "rewritten" not in out["answer"]
    assert "grading" not in out["answer"]
