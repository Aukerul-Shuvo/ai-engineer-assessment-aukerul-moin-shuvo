from pathlib import Path

import httpx
import pytest
import respx

from app.retrieval.build import run_build
from app.retrieval.corpus import CorpusPaths

TINY_SQUAD = {
    "version": "1.1",
    "data": [
        {
            "title": "Oxygen",
            "paragraphs": [
                {
                    "context": "Oxygen is a chemical element with symbol O and atomic number 8.",
                    "qas": [
                        {
                            "id": "q1",
                            "question": "What is the atomic number of oxygen?",
                            "answers": [{"text": "8", "answer_start": 60}],
                        }
                    ],
                }
            ],
        }
    ],
}

SOURCE_URL = "https://example.test/dev-v1.1.json"


@respx.mock
async def test_run_build_downloads_the_source_when_given_a_url(tmp_path: Path) -> None:
    route = respx.get(SOURCE_URL).mock(return_value=httpx.Response(200, json=TINY_SQUAD))
    paths = CorpusPaths(tmp_path / "data")

    manifest = await run_build(source=SOURCE_URL, paths=paths, embedder=None)

    assert route.called
    assert manifest.source_url == SOURCE_URL
    assert manifest.paragraphs == 1
    assert manifest.questions == 1
    assert paths.paragraphs.exists()


@respx.mock
async def test_run_build_fails_loudly_when_the_source_is_unreachable(tmp_path: Path) -> None:
    respx.get(SOURCE_URL).mock(return_value=httpx.Response(404))

    with pytest.raises(httpx.HTTPStatusError):
        await run_build(source=SOURCE_URL, paths=CorpusPaths(tmp_path / "data"), embedder=None)
