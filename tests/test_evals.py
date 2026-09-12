"""The evaluation harness must itself be correct, or its numbers mean nothing."""

import json
from pathlib import Path

from app.retrieval.corpus import CorpusPaths, GoldQuestion
from evals.common import (
    contains_gold,
    exact_match,
    f1,
    load_gold,
    normalize_answer,
    percentile,
)
from evals.retrieval import RankingScores, evaluate_retrieval, render
from evals.routing import GOLDEN_PATH, load_golden
from tests.fakes import TEST_CORPUS, FakeEmbedder
from tests.fakes import build_test_corpus as build_corpus
from tests.fakes import load_test_store as load


# ---------------------------------------------------------------- SQuAD scoring
def test_normalisation_matches_the_squad_script() -> None:
    assert normalize_answer("The Denver Broncos.") == "denver broncos"
    assert normalize_answer("  An  apple, a day ") == "apple day"
    en_dash_score = f"24{chr(0x2013)}10"  # SQuAD strips ASCII punctuation only; an en dash survives
    assert normalize_answer(en_dash_score) == en_dash_score


def test_exact_match_and_f1_take_the_best_gold_answer() -> None:
    golds = ["Denver Broncos", "the Denver Broncos"]

    assert exact_match("The Denver Broncos", golds) == 1.0
    assert exact_match("Broncos", golds) == 0.0
    assert f1("Broncos", golds) == 2 * (1.0 * 0.5) / 1.5
    assert f1("nothing relevant", golds) == 0.0
    assert f1("", golds) == 0.0


def test_contains_gold_is_normalised_substring_match() -> None:
    assert contains_gold("...champion Denver Broncos defeated...", ["the Denver Broncos"])
    assert not contains_gold("Carolina Panthers", ["Denver Broncos"])
    assert not contains_gold("anything", [""]), "blank gold answers never match"


def test_percentile_is_nearest_rank() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    assert percentile(values, 50) == 3.0
    assert percentile(values, 95) == 5.0
    assert percentile([], 50) == 0.0


# ---------------------------------------------------------------- retrieval scoring
def test_ranking_scores_count_hits_at_each_k_and_mrr() -> None:
    scores = RankingScores(ks=[1, 3, 5])

    scores.add(["a", "b", "c"], gold_id="a")  # rank 1
    scores.add(["a", "b", "c"], gold_id="c")  # rank 3
    scores.add(["a", "b", "c"], gold_id="zz")  # miss

    summary = scores.summary()
    assert summary["recall@1"] == round(1 / 3, 4)
    assert summary["recall@3"] == round(2 / 3, 4)
    assert summary["recall@5"] == round(2 / 3, 4)
    assert summary["mrr"] == round((1 + 1 / 3 + 0) / 3, 4)


async def test_evaluate_retrieval_on_a_tiny_corpus_reports_every_ranking(tmp_path: Path) -> None:
    embedder = FakeEmbedder(16)
    paths = await build_corpus(tmp_path, embedder=embedder)
    store = load(paths, embedder=embedder)
    gold = [
        GoldQuestion(
            id="g1",
            question="Who defeated the Carolina Panthers in Super Bowl 50?",
            answers=["Denver Broncos"],
            paragraph_id="super-bowl-50-000",
            title="Super_Bowl_50",
        ),
        GoldQuestion(
            id="g2",
            question="Where did Tesla want to build a transmitter?",
            answers=["Wardenclyffe"],
            paragraph_id="nikola-tesla-000",
            title="Nikola_Tesla",
        ),
    ]

    report = await evaluate_retrieval(store, gold, ks=(1, 3), pool=3)

    assert set(report["results"]) == {"dense"}, "no reranker was passed"
    assert report["results"]["dense"]["recall@1"] == 1.0
    assert report["results"]["dense"]["mrr"] == 1.0
    assert report["questions"] == 2 and report["pool"] == 3
    assert "| dense |" in render(report)


async def test_dense_ranking_is_ids_in_vector_order(tmp_path: Path) -> None:
    embedder = FakeEmbedder(16)
    store = load(await build_corpus(tmp_path, embedder=embedder), embedder=embedder)

    ranking = await store.dense_ranking("Denver Broncos", k=3)

    assert ranking[0] == "super-bowl-50-000"
    assert len(ranking) == 3


# ---------------------------------------------------------------- gold sets
def test_load_gold_samples_reproducibly(tmp_path: Path) -> None:
    paths = CorpusPaths(tmp_path)
    tmp_path.mkdir(exist_ok=True)
    rows = [
        GoldQuestion(id=str(i), question=f"q{i}", answers=["a"], paragraph_id="p", title="t")
        for i in range(20)
    ]
    paths.questions.write_text(
        "".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8"
    )

    first = load_gold(paths, limit=5, seed=7)
    second = load_gold(paths, limit=5, seed=7)
    everything = load_gold(paths, limit=0)

    assert [q.id for q in first] == [q.id for q in second]
    assert len(first) == 5 and len(everything) == 20


def test_routing_golden_set_is_well_formed() -> None:
    cases = load_golden(GOLDEN_PATH)

    assert len(cases) >= 25
    assert {c.intent for c in cases} == {"answerable", "out_of_scope", "chitchat"}
    assert any(set(c.sources) == {"dataset", "superhero"} for c in cases), "has 'both' cases"
    for case in cases:
        assert set(case.sources) <= {"dataset", "superhero"}
        assert (case.intent == "answerable") == bool(case.sources)
    # Every line is valid JSON with exactly these keys.
    for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines():
        assert set(json.loads(line)) == {"question", "intent", "sources", "min_sub_queries"}


def test_corpus_fixture_titles_are_the_ones_the_tests_assume() -> None:
    assert [a["title"] for a in TEST_CORPUS["data"]] == ["Super_Bowl_50", "Nikola_Tesla", "Oxygen"]
