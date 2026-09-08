"""Corpus files: what the build script writes and the retrieval store reads.

The corpus lives in ``data/`` as plain, inspectable files so a reviewer can open them:

* ``paragraphs.jsonl``      one JSON object per paragraph: id, article title, text, Wikipedia URL
* ``questions.jsonl``       the gold questions, each with the paragraph it was written from
* ``bm25/``                 the lexical index as saved by bm25s
* ``embeddings.npy``        dense vectors, one row per paragraph, same order as paragraphs.jsonl
* ``embeddings.meta.json``  which model produced them and a fingerprint of the paragraphs
* ``manifest.json``         provenance: source URL, hash, counts, build time

``indexed_text`` is the single definition of what gets indexed: the article title prepended to
the paragraph. Both the lexical and the dense index use it, so both retrievers see the same
text. Excerpts shown to users come from the raw ``text`` field, never the indexed form.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict

WIKIPEDIA_BASE = "https://en.wikipedia.org/wiki/"


class Paragraph(BaseModel):
    """One retrievable unit of the corpus: a Wikipedia paragraph as it appears in SQuAD."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    title: str
    index: int
    text: str
    url: str

    @property
    def display_title(self) -> str:
        """Human-readable article title: ``Super_Bowl_50`` becomes ``Super Bowl 50``."""
        return self.title.replace("_", " ")


class GoldQuestion(BaseModel):
    """A human-written question with its accepted answers and the paragraph it came from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    question: str
    answers: list[str]
    paragraph_id: str
    title: str


class EmbeddingsMeta(BaseModel):
    """Describes ``embeddings.npy`` so the store can refuse a stale or mismatched index."""

    model_config = ConfigDict(extra="forbid")

    model: str
    dimensions: int
    count: int
    dtype: str = "float16"
    normalized: bool = True
    task_type: str = "RETRIEVAL_DOCUMENT"
    paragraphs_sha256: str
    created_at: str


class BuildManifest(BaseModel):
    """Provenance of everything in ``data/``: where it came from and what it contains."""

    model_config = ConfigDict(extra="forbid")

    source_url: str
    source_sha256: str
    source_bytes: int
    squad_version: str
    articles: int
    paragraphs: int
    questions: int
    built_at: str
    embeddings: EmbeddingsMeta | None = None


@dataclass(frozen=True)
class CorpusPaths:
    """Every file location under one data directory, so no path is spelled twice."""

    data_dir: Path

    @property
    def paragraphs(self) -> Path:
        """One paragraph per line."""
        return self.data_dir / "paragraphs.jsonl"

    @property
    def questions(self) -> Path:
        """One gold question per line."""
        return self.data_dir / "questions.jsonl"

    @property
    def bm25_dir(self) -> Path:
        """Directory holding the saved bm25s index."""
        return self.data_dir / "bm25"

    @property
    def embeddings(self) -> Path:
        """Dense vectors as a numpy array."""
        return self.data_dir / "embeddings.npy"

    @property
    def embeddings_meta(self) -> Path:
        """Model id and fingerprint for the dense vectors."""
        return self.data_dir / "embeddings.meta.json"

    @property
    def manifest(self) -> Path:
        """Build provenance."""
        return self.data_dir / "manifest.json"


def indexed_text(paragraph: Paragraph) -> str:
    """The text both retrievers index: title header plus paragraph body."""
    return f"{paragraph.display_title}. {paragraph.text}"


def paragraph_id(title: str, index: int) -> str:
    """Stable id from the article slug and the paragraph's position, e.g. ``super-bowl-50-003``."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return f"{slug}-{index:03d}"


def wikipedia_url(title: str) -> str:
    """Article URL. SQuAD titles are already Wikipedia path segments."""
    return WIKIPEDIA_BASE + quote(title, safe="(),_'!:")


def paragraphs_fingerprint(paragraphs: Iterable[Paragraph]) -> str:
    """SHA-256 over ids and texts. Lets the store verify vectors match the paragraphs."""
    digest = hashlib.sha256()
    for paragraph in paragraphs:
        digest.update(paragraph.id.encode())
        digest.update(b"\t")
        digest.update(paragraph.text.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def read_jsonl[ModelT: BaseModel](path: Path, model: type[ModelT]) -> list[ModelT]:
    """Load and validate every line of a JSONL file as ``model``."""
    with path.open("r", encoding="utf-8") as handle:
        return [model.model_validate_json(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[BaseModel]) -> None:
    """Write models one per line. Creates parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(row.model_dump_json())
            handle.write("\n")
