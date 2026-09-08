"""Dense retrieval over precomputed embeddings.

The corpus vectors are built once and committed as ``embeddings.npy`` (float16, unit length).
At startup they are loaded into a float32 matrix. A query is embedded with the same model and
the query task type, and similarity is one matrix-vector product: 2,067 by 768 floats, well
under a millisecond. No vector database is needed at this size; the ADR names the scale-up path.

The index refuses to load when the metadata disagrees with the paragraphs on disk (count or
fingerprint) or with the configured embedding model, so a stale vector file can never silently
return wrong neighbours.
"""

from __future__ import annotations

import numpy as np

from app.retrieval.corpus import CorpusPaths, EmbeddingsMeta


class DenseIndexMismatchError(ValueError):
    """The vectors on disk do not belong to the current paragraphs or model."""


class DenseIndex:
    """Cosine similarity search over unit vectors held in memory."""

    def __init__(self, matrix: np.ndarray, meta: EmbeddingsMeta) -> None:
        self._matrix = matrix
        self._meta = meta

    @property
    def size(self) -> int:
        """Number of indexed paragraphs."""
        return int(self._matrix.shape[0])

    @property
    def meta(self) -> EmbeddingsMeta:
        """Provenance of the vectors."""
        return self._meta

    @classmethod
    def load(
        cls,
        paths: CorpusPaths,
        *,
        expected_fingerprint: str,
        expected_count: int,
        expected_model: str,
        expected_dimensions: int,
    ) -> DenseIndex:
        """Load and verify. Raises ``DenseIndexMismatchError`` on any disagreement."""
        meta = EmbeddingsMeta.model_validate_json(paths.embeddings_meta.read_text(encoding="utf-8"))
        problems = []
        if meta.paragraphs_sha256 != expected_fingerprint:
            problems.append("paragraph fingerprint differs from paragraphs.jsonl")
        if meta.count != expected_count:
            problems.append(f"vector count {meta.count} != paragraph count {expected_count}")
        if meta.model != expected_model:
            problems.append(f"built with {meta.model!r}, configured {expected_model!r}")
        if meta.dimensions != expected_dimensions:
            problems.append(f"built with {meta.dimensions} dims, configured {expected_dimensions}")
        if problems:
            raise DenseIndexMismatchError("; ".join(problems))

        matrix = np.load(paths.embeddings).astype(np.float32)
        if matrix.shape != (expected_count, expected_dimensions):
            raise DenseIndexMismatchError(f"unexpected array shape {matrix.shape}")
        return cls(matrix, meta)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Top ``k`` rows by cosine similarity as ``(row, score)``, best first."""
        limit = min(k, self.size)
        if limit <= 0:
            return []
        scores = self._matrix @ query_vector.astype(np.float32)
        top = np.argpartition(-scores, limit - 1)[:limit]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(int(row), float(scores[row])) for row in top]
