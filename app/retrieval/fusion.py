"""Reciprocal rank fusion: combine rankings from different retrievers.

Each retriever produces an ordered list of paragraph rows. A paragraph's fused score is the sum
over the lists it appears in of ``1 / (k + rank)``. Appearing near the top of several lists
beats appearing at the very top of one, which is what makes a paragraph found by both BM25 and
dense search rise above one found by only one of them.

``k`` is 60 by default, the value from Cormack, Clarke and Büttcher (SIGIR 2009) and the default
in Elasticsearch and OpenSearch. Ties break on row number so results are deterministic.
"""

from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]], *, k: int = 60
) -> list[tuple[int, float]]:
    """Fuse ordered row lists into one ranking of ``(row, score)``, best first."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, row in enumerate(ranking, start=1):
            scores[row] = scores.get(row, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
