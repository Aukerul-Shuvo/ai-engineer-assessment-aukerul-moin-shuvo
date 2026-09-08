"""Runs the real FlashRank model. Downloads 22 MB of weights the first time, then uses the cache."""

from pathlib import Path

import pytest

from app.retrieval.rerank import FlashRankReranker

ANSWER = (
    "Super Bowl 50. The AFC champion Denver Broncos defeated the NFC champion Carolina Panthers."
)
DISTRACTOR = "Super Bowl 50. The Panthers finished the regular season with a 15-1 record."
UNRELATED = "Nikola Tesla. Tesla approached Morgan to ask for more funds."


@pytest.mark.slow
async def test_cross_encoder_puts_the_answer_paragraph_first() -> None:
    reranker = await FlashRankReranker.create(
        model_name="ms-marco-MiniLM-L-12-v2", cache_dir=Path("data/models"), max_length=512
    )

    ranked = await reranker.rerank(
        "Which NFL team represented the AFC at Super Bowl 50?",
        [DISTRACTOR, ANSWER, UNRELATED],
    )

    assert [index for index, _ in ranked] == [1, 0, 2]
    assert ranked[0][1] > 0.9
    assert ranked[2][1] < 0.1
    assert reranker.model_id == "ms-marco-MiniLM-L-12-v2"


async def test_empty_input_short_circuits_without_loading_anything() -> None:
    class NoRanker:
        def rerank(self, request: object) -> list[dict[str, object]]:
            raise AssertionError("should not be called")

    reranker = FlashRankReranker(NoRanker(), "stub")  # type: ignore[arg-type]

    assert await reranker.rerank("q", []) == []
