"""Lexical retrieval with BM25.

BM25 is the exact-match half of hybrid retrieval. It scores a paragraph by how many query terms
it contains, weighted by how rare each term is across the corpus. It is what finds
"Denver Broncos" or "1901" when an embedding would blur them into general meaning.

The implementation is bm25s: sparse numpy scoring, English stopword removal and Snowball
stemming so "patents" matches "patented". The same tokenizer runs at build time and at query
time. Changing it means rebuilding the index, which is why it is defined once here.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import bm25s
import Stemmer

_STEMMER = Stemmer.Stemmer("english")


def _tokenize(texts: Sequence[str]) -> bm25s.tokenization.Tokenized:
    return bm25s.tokenize(list(texts), stopwords="english", stemmer=_STEMMER, show_progress=False)


class LexicalIndex:
    """A saved-and-loadable BM25 index over paragraph texts."""

    def __init__(self, retriever: bm25s.BM25, size: int) -> None:
        self._retriever = retriever
        self._size = size

    @property
    def size(self) -> int:
        """Number of indexed paragraphs."""
        return self._size

    @classmethod
    def build(cls, texts: Sequence[str]) -> LexicalIndex:
        """Index ``texts``; result row ``i`` refers to ``texts[i]``."""
        retriever = bm25s.BM25()
        retriever.index(_tokenize(texts), show_progress=False)
        return cls(retriever, len(texts))

    def save(self, directory: Path) -> None:
        """Persist to ``directory`` as bm25s' numpy and JSON files."""
        directory.mkdir(parents=True, exist_ok=True)
        self._retriever.save(str(directory), show_progress=False)

    @classmethod
    def load(cls, directory: Path) -> LexicalIndex:
        """Load an index previously written by ``save``."""
        retriever = bm25s.BM25.load(str(directory), load_corpus=False, show_progress=False)
        size = int(retriever.scores["num_docs"])
        return cls(retriever, size)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Top ``k`` rows for ``query`` as ``(row, score)``, best first. Zero scores are dropped."""
        tokens = _tokenize([query])
        if not tokens.ids or not tokens.ids[0]:
            return []
        limit = min(k, self._size)
        rows, scores = self._retriever.retrieve(tokens, k=limit, show_progress=False)
        return [
            (int(row), float(score))
            for row, score in zip(rows[0], scores[0], strict=True)
            if score > 0
        ]
