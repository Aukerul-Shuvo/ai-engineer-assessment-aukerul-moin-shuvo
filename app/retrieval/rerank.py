"""Cross-encoder reranking of retrieval candidates.

First-stage retrievers score query and passage separately, which is fast but coarse. A
cross-encoder reads the query and one passage together and scores the pair, which is far more
precise but too slow to run over the whole corpus. So it runs only over the fused candidates.

The production reranker is FlashRank's ``ms-marco-MiniLM-L-12-v2``, a 34 MB ONNX model that runs
on CPU in tens of milliseconds for a hundred passages. It is synchronous, so it is offloaded to a
worker thread to keep the event loop free. Model weights download on first use into
``Settings.reranker_cache_dir``; the Docker image pre-warms them.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from flashrank import Ranker, RerankRequest


class Reranker(Protocol):
    """Score every passage against the query; return all of them, best first."""

    @property
    def model_id(self) -> str:
        """Identifier recorded in retrieval provenance."""
        ...

    async def rerank(self, query: str, passages: Sequence[str]) -> list[tuple[int, float]]:
        """``(index into passages, score)`` for every passage, sorted by score descending."""
        ...


class FlashRankReranker:
    """FlashRank cross-encoder, constructed off the event loop, scored in a worker thread."""

    def __init__(self, ranker: Ranker, model_name: str) -> None:
        self._ranker = ranker
        self._model_name = model_name

    @classmethod
    async def create(
        cls, *, model_name: str, cache_dir: Path, max_length: int = 512
    ) -> FlashRankReranker:
        """Load (and on first use download) the model without blocking the event loop."""

        def build() -> Ranker:
            cache_dir.mkdir(parents=True, exist_ok=True)
            return Ranker(
                model_name=model_name,
                cache_dir=str(cache_dir),
                max_length=max_length,
                log_level="WARNING",
            )

        return cls(await asyncio.to_thread(build), model_name)

    @property
    def model_id(self) -> str:
        """The FlashRank model name."""
        return self._model_name

    async def rerank(self, query: str, passages: Sequence[str]) -> list[tuple[int, float]]:
        """Score all passages in a worker thread."""
        if not passages:
            return []
        return await asyncio.to_thread(self._rerank_sync, query, list(passages))

    def _rerank_sync(self, query: str, passages: list[str]) -> list[tuple[int, float]]:
        request = RerankRequest(
            query=query, passages=[{"id": i, "text": text} for i, text in enumerate(passages)]
        )
        return [(int(item["id"]), float(item["score"])) for item in self._ranker.rerank(request)]
