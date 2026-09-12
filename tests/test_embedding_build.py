"""The embedding build under a per-text quota: pacing, retry-after, daily stop, resume."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from app.retrieval.build import EmbeddingQuotaError, build_embeddings
from app.retrieval.corpus import CorpusPaths, Paragraph
from tests.fakes import FakeEmbedder

MINUTE_LIMIT = (
    "429 RESOURCE_EXHAUSTED. quotaId: EmbedContentRequestsPerMinutePerUserPerProjectPerModel"
    "-FreeTier. Please retry in 0.5s."
)
DAILY_LIMIT = (
    "429 RESOURCE_EXHAUSTED. quotaId: EmbedContentRequestsPerDayPerProjectPerModel-FreeTier."
)


class FlakyEmbedder(FakeEmbedder):
    """Raises the queued exceptions first, then behaves like the fake."""

    def __init__(self, failures: list[Exception]) -> None:
        super().__init__()
        self._failures = failures

    async def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if self._failures:
            raise self._failures.pop(0)
        return await super().embed_documents(texts)


class StopsAfterOneBatch(FakeEmbedder):
    """Embeds one batch, then reports the daily quota spent, like the free tier at 1,000 texts."""

    async def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if self.calls >= 1:
            raise RuntimeError(DAILY_LIMIT)
        return await super().embed_documents(texts)


def paragraphs(count: int, prefix: str = "p") -> list[Paragraph]:
    return [
        Paragraph(
            id=f"{prefix}-{i:03d}",
            title="Article",
            index=i,
            text=f"{prefix} paragraph number {i} about topic {i}",
            url="https://en.wikipedia.org/wiki/Article",
        )
        for i in range(count)
    ]


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []

    async def record(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("app.retrieval.build.asyncio.sleep", record)
    return slept


async def test_daily_quota_saves_progress_and_a_rerun_resumes(tmp_path: Path) -> None:
    paths = CorpusPaths(tmp_path)
    corpus = paragraphs(5)

    with pytest.raises(EmbeddingQuotaError):
        await build_embeddings(
            corpus, FlakyEmbedder([RuntimeError(DAILY_LIMIT)]), batch_size=2, paths=paths
        )
    assert not paths.embeddings_partial.exists(), "nothing embedded, nothing saved"

    with pytest.raises(EmbeddingQuotaError) as info:
        await build_embeddings(corpus, StopsAfterOneBatch(), batch_size=2, paths=paths)
    assert (info.value.done, info.value.total) == (2, 5)
    assert np.load(paths.embeddings_partial).shape == (2, 32)

    resumed = FakeEmbedder()
    vectors, meta = await build_embeddings(corpus, resumed, batch_size=2, paths=paths)
    assert vectors.shape == (5, 32)
    assert vectors.dtype == np.float16
    assert meta.count == 5
    assert resumed.calls == 2, "only the three remaining texts were embedded, in two batches"


async def test_partial_progress_is_discarded_when_the_corpus_changes(tmp_path: Path) -> None:
    paths = CorpusPaths(tmp_path)
    with pytest.raises(EmbeddingQuotaError):
        await build_embeddings(paragraphs(4), StopsAfterOneBatch(), batch_size=2, paths=paths)
    assert paths.embeddings_partial.exists()

    fresh = FakeEmbedder()
    vectors, _ = await build_embeddings(paragraphs(4, prefix="q"), fresh, batch_size=2, paths=paths)
    assert vectors.shape == (4, 32)
    assert fresh.calls == 2, "a different corpus starts from scratch"


async def test_per_minute_limit_waits_the_advertised_delay_and_retries(
    no_sleep: list[float],
) -> None:
    embedder = FlakyEmbedder([RuntimeError(MINUTE_LIMIT)])
    vectors, _ = await build_embeddings(paragraphs(3), embedder, batch_size=3)
    assert vectors.shape == (3, 32)
    assert no_sleep == [1.5], "retry-after of 0.5 s plus a one-second margin"


async def test_pacing_sleeps_between_batches_but_not_after_the_last(
    no_sleep: list[float],
) -> None:
    await build_embeddings(paragraphs(6), FakeEmbedder(), batch_size=2, texts_per_minute=120)
    assert len(no_sleep) == 2, "three batches, two pauses"
    assert all(0.9 < pause <= 1.0 for pause in no_sleep), "2 texts at 120/min is one second each"


async def test_non_quota_errors_propagate_unchanged() -> None:
    embedder = FlakyEmbedder([ValueError("bad request")])
    with pytest.raises(ValueError, match="bad request"):
        await build_embeddings(paragraphs(2), embedder, batch_size=2)
