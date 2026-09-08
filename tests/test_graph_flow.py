"""End-to-end runs of the compiled graph with a scripted model, a tiny corpus and a mocked API.

Structured-output calls and chat calls are scripted in separate queues. Parallel branches make
the global call order non-deterministic, but the graph fixes the order within each kind.
"""

from pathlib import Path

import httpx
import pytest
import respx
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

from app.graph.build import GraphDependencies, build_graph
from app.graph.schemas import (
    GroundingVerdict,
    QueryPlan,
    ResolvedSubQueries,
    RetrievalGrade,
    SubQuery,
)
from app.graph.tools import build_superhero_tools
from app.llm.providers import ChatModels, Provider
from app.retrieval.store import HybridStore
from app.tools.circuit_breaker import CircuitBreaker
from app.tools.superhero import SuperheroClient
from tests.fakes import FakeEmbedder, ScriptedChatModel
from tests.test_store import build_corpus, load
from tests.test_superhero import BASE, BATMEN, TOKEN

TITLES = ["Super_Bowl_50", "Nikola_Tesla", "Oxygen"]


@pytest.fixture
async def store(tmp_path: Path) -> HybridStore:
    embedder = FakeEmbedder(16)
    return load(await build_corpus(tmp_path, embedder=embedder), embedder=embedder)


def make_deps(
    *,
    store: HybridStore | None,
    structured: list[object],
    chat: list[AIMessage],
    tools: list[BaseTool] | None = None,
) -> tuple[GraphDependencies, ScriptedChatModel]:
    model = ScriptedChatModel(structured_script=structured, script=chat)
    models = ChatModels([Provider(name="fake", model_id="scripted", model=model)])
    deps = GraphDependencies(
        models=models, store=store, superhero_tools=tools or [], document_titles=TITLES
    )
    return deps, model


def dataset_plan(*texts: str) -> QueryPlan:
    return QueryPlan(
        intent="answerable",
        sub_queries=[
            SubQuery(id=f"q{i}", text=text, source="dataset") for i, text in enumerate(texts, 1)
        ],
    )


async def test_out_of_scope_question_gets_a_direct_reply_and_no_sources(
    store: HybridStore,
) -> None:
    deps, model = make_deps(
        store=store,
        structured=[QueryPlan(intent="out_of_scope", direct_reply="I cover the corpus only.")],
        chat=[],
    )

    final = await build_graph(deps).ainvoke({"question": "What's the weather in Doha?"})

    assert final["answer"] == "I cover the corpus only."
    assert final["cited_evidence"] == []
    assert final["citations"] == []
    assert final["grounded"] is None
    assert not model.structured_script and not model.script


@respx.mock
async def test_compound_question_runs_both_branches_and_cites_both(store: HybridStore) -> None:
    respx.get(f"{BASE}/{TOKEN}/search/batman").mock(
        return_value=httpx.Response(200, json={"response": "success", "results": BATMEN})
    )
    async with httpx.AsyncClient(follow_redirects=True) as http:
        client = SuperheroClient(http, base_url=BASE, token=TOKEN, breaker=CircuitBreaker())
        deps, model = make_deps(
            store=store,
            tools=build_superhero_tools(client),
            structured=[
                QueryPlan(
                    intent="answerable",
                    sub_queries=[
                        SubQuery(id="q1", text="Who won Super Bowl 50?", source="dataset"),
                        SubQuery(
                            id="q2",
                            text="How strong is Batman?",
                            source="superhero",
                            hero_name="Batman",
                        ),
                    ],
                ),
                RetrievalGrade(relevant_ids=["super-bowl-50-000"], sufficient=True),
                GroundingVerdict(supported=True),
            ],
            chat=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "search_superheroes", "args": {"name": "batman"}, "id": "c1"}
                    ],
                ),
                AIMessage(content="Found Batman."),
                AIMessage(content="The Denver Broncos won [S1]. Batman's strength is 26 [S3]."),
            ],
        )

        final = await build_graph(deps).ainvoke(
            {"question": "Who won Super Bowl 50, and how strong is Batman?"}
        )

    kinds = [e.kind for e in final["cited_evidence"]]
    assert kinds[0] == "dataset", "plan order: the dataset sub-query is labelled first"
    assert kinds.count("superhero") == 3, "one evidence item per Batman the tool returned"
    assert final["citations"] == ["S1", "S3"]
    assert final["grounded"] is True
    assert final["answer"].startswith("The Denver Broncos won [S1].")
    assert {"plan:fake", "grade:fake", "agent:fake", "synth:fake", "ground:fake"} <= set(
        final["providers"]
    )
    assert not model.structured_script and not model.script, "every scripted call was used"


async def test_grounding_failure_regenerates_once_with_feedback(store: HybridStore) -> None:
    deps, model = make_deps(
        store=store,
        structured=[
            dataset_plan("Who won Super Bowl 50?"),
            RetrievalGrade(relevant_ids=["super-bowl-50-000"], sufficient=True),
            GroundingVerdict(supported=False, unsupported_claims=["The score was 30-10"]),
            GroundingVerdict(supported=True),
        ],
        chat=[
            AIMessage(content="The Broncos won 30-10 [S1]."),
            AIMessage(content="The Broncos won 24-10 [S1]."),
        ],
    )

    final = await build_graph(deps).ainvoke({"question": "Who won Super Bowl 50?"})

    assert final["answer"] == "The Broncos won 24-10 [S1]."
    assert final["grounded"] is True
    assert final["regenerations"] == 1
    second_synthesis_prompt = str(model.calls[-2])
    assert "30-10" in second_synthesis_prompt, "the regeneration saw the grounding feedback"


async def test_dependent_sub_query_runs_in_a_second_wave(store: HybridStore) -> None:
    deps, _ = make_deps(
        store=store,
        structured=[
            QueryPlan(
                intent="answerable",
                sub_queries=[
                    SubQuery(id="q1", text="What is oxygen's atomic number?", source="dataset"),
                    SubQuery(
                        id="q2",
                        text="Which article mentions the number <from q1>?",
                        source="dataset",
                        depends_on=["q1"],
                    ),
                ],
            ),
            RetrievalGrade(relevant_ids=["oxygen-000"], sufficient=True),
            ResolvedSubQueries(
                sub_queries=[
                    SubQuery(id="q2", text="Which article mentions the number 8?", source="dataset")
                ]
            ),
            RetrievalGrade(relevant_ids=["oxygen-000"], sufficient=True),
            GroundingVerdict(supported=True),
        ],
        chat=[AIMessage(content="Oxygen's atomic number is 8 [S1].")],
    )

    final = await build_graph(deps).ainvoke({"question": "compound"})

    assert final["wave"] == 2
    assert final["pending"] == []
    assert "resolve:fake" in final["providers"]
    assert {e.sub_query_id for e in final["evidence"]} == {"q1", "q2"}


async def test_degraded_run_without_any_model_still_returns_sources(store: HybridStore) -> None:
    deps = GraphDependencies(models=None, store=store, document_titles=TITLES)

    final = await build_graph(deps).ainvoke({"question": "Denver Broncos"})

    assert final["degraded"] is True
    assert final["answer"] is None
    assert final["cited_evidence"], "retrieval still ran and sources are available"
    assert final["cited_evidence"][0].locator["paragraph_id"] == "super-bowl-50-000"
