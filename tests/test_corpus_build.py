import json
from pathlib import Path

import numpy as np

from app.retrieval.build import (
    build_embeddings,
    parse_squad,
    read_manifest,
    run_build,
    run_embed_only,
)
from app.retrieval.corpus import (
    CorpusPaths,
    EmbeddingsMeta,
    GoldQuestion,
    Paragraph,
    indexed_text,
    paragraph_id,
    paragraphs_fingerprint,
    read_jsonl,
    wikipedia_url,
)
from tests.fakes import FakeEmbedder

MINI_SQUAD = {
    "version": "1.1",
    "data": [
        {
            "title": "Super_Bowl_50",
            "paragraphs": [
                {
                    "context": (
                        "Super Bowl 50 was an American football game. The American Football "
                        "Conference champion Denver Broncos defeated the National Football "
                        "Conference champion Carolina Panthers 24-10."
                    ),
                    "qas": [
                        {
                            "id": "q1",
                            "question": "Which NFL team represented the AFC at Super Bowl 50?",
                            "answers": [
                                {"text": "Denver Broncos", "answer_start": 100},
                                {"text": "Denver Broncos", "answer_start": 100},
                                {"text": "the Denver Broncos", "answer_start": 96},
                            ],
                        }
                    ],
                },
                {
                    "context": (
                        "The game was played on February 7, 2016, at Levi's Stadium in "
                        "Santa Clara, California."
                    ),
                    "qas": [
                        {
                            "id": "q2",
                            "question": "Where did Super Bowl 50 take place?",
                            "answers": [{"text": "Santa Clara, California", "answer_start": 60}],
                        }
                    ],
                },
            ],
        },
        {
            "title": "Nikola_Tesla",
            "paragraphs": [
                {
                    "context": (
                        "Nikola Tesla was a Serbian-American inventor best known for his "
                        "contributions to the alternating current induction motor."
                    ),
                    "qas": [
                        {
                            "id": "q3",
                            "question": "What motor did Tesla contribute to?",
                            "answers": [{"text": "induction motor", "answer_start": 100}],
                        }
                    ],
                }
            ],
        },
    ],
}


def test_parse_squad_keeps_order_and_links_questions_to_paragraphs() -> None:
    paragraphs, questions = parse_squad(MINI_SQUAD)

    assert [p.id for p in paragraphs] == [
        "super-bowl-50-000",
        "super-bowl-50-001",
        "nikola-tesla-000",
    ]
    assert paragraphs[0].title == "Super_Bowl_50"
    assert paragraphs[0].display_title == "Super Bowl 50"
    assert paragraphs[0].url == "https://en.wikipedia.org/wiki/Super_Bowl_50"
    assert [q.paragraph_id for q in questions] == [
        "super-bowl-50-000",
        "super-bowl-50-001",
        "nikola-tesla-000",
    ]


def test_parse_squad_collapses_duplicate_answers_preserving_order() -> None:
    _, questions = parse_squad(MINI_SQUAD)

    assert questions[0].answers == ["Denver Broncos", "the Denver Broncos"]


def test_ids_and_urls_are_url_safe_for_awkward_titles() -> None:
    assert paragraph_id("Sky_(United_Kingdom)", 7) == "sky-united-kingdom-007"
    assert paragraph_id("Fresno,_California", 0) == "fresno-california-000"
    assert (
        wikipedia_url("Sky_(United_Kingdom)")
        == "https://en.wikipedia.org/wiki/Sky_(United_Kingdom)"
    )
    assert wikipedia_url("Fresno,_California") == "https://en.wikipedia.org/wiki/Fresno,_California"


def test_indexed_text_prepends_the_title_header() -> None:
    paragraphs, _ = parse_squad(MINI_SQUAD)

    assert indexed_text(paragraphs[2]).startswith("Nikola Tesla. Nikola Tesla was")


async def test_build_embeddings_produces_float16_unit_vectors_with_meta() -> None:
    paragraphs, _ = parse_squad(MINI_SQUAD)
    embedder = FakeEmbedder(dimensions=16)

    vectors, meta = await build_embeddings(paragraphs, embedder, batch_size=2)

    assert vectors.shape == (3, 16)
    assert vectors.dtype == np.float16
    assert np.allclose(np.linalg.norm(vectors.astype(np.float32), axis=1), 1.0, atol=1e-2)
    assert embedder.calls == 2, "3 texts at batch size 2 means two calls"
    assert meta == EmbeddingsMeta(
        model="fake-embedder",
        dimensions=16,
        count=3,
        paragraphs_sha256=paragraphs_fingerprint(paragraphs),
        created_at=meta.created_at,
    )


async def test_run_build_without_embedder_writes_the_corpus_and_manifest(tmp_path: Path) -> None:
    source = tmp_path / "dev.json"
    source.write_text(json.dumps(MINI_SQUAD), encoding="utf-8")
    paths = CorpusPaths(tmp_path / "data")

    manifest = await run_build(source=str(source), paths=paths, embedder=None)

    assert paths.paragraphs.exists() and paths.questions.exists()
    assert not paths.embeddings.exists()
    assert manifest.articles == 2
    assert manifest.paragraphs == 3
    assert manifest.questions == 3
    assert manifest.embeddings is None
    assert manifest.squad_version == "1.1"
    assert len(manifest.source_sha256) == 64

    assert read_jsonl(paths.paragraphs, Paragraph) == parse_squad(MINI_SQUAD)[0]
    assert read_jsonl(paths.questions, GoldQuestion) == parse_squad(MINI_SQUAD)[1]
    assert read_manifest(paths) == manifest


async def test_run_embed_only_adds_vectors_to_an_existing_corpus(tmp_path: Path) -> None:
    source = tmp_path / "dev.json"
    source.write_text(json.dumps(MINI_SQUAD), encoding="utf-8")
    paths = CorpusPaths(tmp_path / "data")
    await run_build(source=str(source), paths=paths, embedder=None)

    manifest = await run_embed_only(paths=paths, embedder=FakeEmbedder(dimensions=8))

    assert manifest.embeddings is not None
    assert manifest.embeddings.count == 3
    assert manifest.embeddings.dimensions == 8
    assert np.load(paths.embeddings).shape == (3, 8)
    assert read_manifest(paths).embeddings == manifest.embeddings
    assert read_manifest(paths).paragraphs == 3, "the rest of the manifest is preserved"
