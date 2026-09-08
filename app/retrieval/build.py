"""Turn the SQuAD dev file into the corpus files the service reads.

Input: the canonical ``dev-v1.1.json`` from the SQuAD project, fetched from its URL or read
from disk. Output: the files described in ``corpus.py``.

The transformation deliberately does nothing clever. Paragraphs are kept verbatim in source
order, each gets a stable id and its Wikipedia URL, and every gold question records the
paragraph it was written from. That last fact is what makes retrieval evaluation exact rather
than approximate. Dense embeddings are optional at build time so the corpus and lexical index
can be produced without a model key; the store degrades to BM25-only when they are absent.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import structlog

from app.llm.embeddings import Embedder
from app.retrieval.corpus import (
    BuildManifest,
    CorpusPaths,
    EmbeddingsMeta,
    GoldQuestion,
    Paragraph,
    indexed_text,
    paragraph_id,
    paragraphs_fingerprint,
    read_jsonl,
    wikipedia_url,
    write_jsonl,
)
from app.retrieval.lexical import LexicalIndex

SQUAD_DEV_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"

log = structlog.get_logger(__name__)


async def load_source(source: str) -> tuple[bytes, dict[str, Any]]:
    """Fetch the SQuAD JSON from a URL or a local path. Returns raw bytes and parsed document."""
    if source.startswith(("http://", "https://")):
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            response = await client.get(source)
            response.raise_for_status()
            raw = response.content
    else:
        raw = await asyncio.to_thread(Path(source).read_bytes)
    document: dict[str, Any] = json.loads(raw)
    return raw, document


def parse_squad(document: dict[str, Any]) -> tuple[list[Paragraph], list[GoldQuestion]]:
    """Flatten SQuAD's article, paragraph, question nesting into two lists.

    Order is preserved so ids are reproducible. Duplicate answer strings from multiple
    annotators are collapsed while keeping first-seen order.
    """
    paragraphs: list[Paragraph] = []
    questions: list[GoldQuestion] = []
    for article in document["data"]:
        title = article["title"]
        url = wikipedia_url(title)
        for index, entry in enumerate(article["paragraphs"]):
            pid = paragraph_id(title, index)
            paragraphs.append(
                Paragraph(id=pid, title=title, index=index, text=entry["context"], url=url)
            )
            for qa in entry["qas"]:
                answers = list(dict.fromkeys(a["text"] for a in qa["answers"]))
                questions.append(
                    GoldQuestion(
                        id=qa["id"],
                        question=qa["question"],
                        answers=answers,
                        paragraph_id=pid,
                        title=title,
                    )
                )
    return paragraphs, questions


def build_lexical_index(paragraphs: Sequence[Paragraph]) -> LexicalIndex:
    """BM25 over the indexed form of every paragraph."""
    return LexicalIndex.build([indexed_text(p) for p in paragraphs])


async def build_embeddings(
    paragraphs: Sequence[Paragraph], embedder: Embedder, *, batch_size: int = 100
) -> tuple[np.ndarray, EmbeddingsMeta]:
    """Embed every paragraph in batches. Returns float16 unit vectors and their metadata."""
    texts = [indexed_text(p) for p in paragraphs]
    chunks: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        chunks.append(await embedder.embed_documents(batch))
        log.info("embedded", done=min(start + batch_size, len(texts)), total=len(texts))
    vectors = np.vstack(chunks).astype(np.float16)
    meta = EmbeddingsMeta(
        model=embedder.model_id,
        dimensions=embedder.dimensions,
        count=len(paragraphs),
        paragraphs_sha256=paragraphs_fingerprint(paragraphs),
        created_at=_now(),
    )
    return vectors, meta


def write_embeddings(paths: CorpusPaths, vectors: np.ndarray, meta: EmbeddingsMeta) -> None:
    """Persist the dense index and its metadata side by side."""
    np.save(paths.embeddings, vectors)
    paths.embeddings_meta.write_text(meta.model_dump_json(indent=2) + "\n", encoding="utf-8")


def write_manifest(paths: CorpusPaths, manifest: BuildManifest) -> None:
    """Persist build provenance."""
    paths.manifest.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")


def read_manifest(paths: CorpusPaths) -> BuildManifest:
    """Load build provenance."""
    return BuildManifest.model_validate_json(paths.manifest.read_text(encoding="utf-8"))


async def run_build(
    *,
    source: str,
    paths: CorpusPaths,
    embedder: Embedder | None,
    batch_size: int = 100,
) -> BuildManifest:
    """Full build: fetch, parse, write corpus files, build BM25, optionally embed."""
    raw, document = await load_source(source)
    paragraphs, questions = parse_squad(document)
    log.info(
        "parsed",
        articles=len(document["data"]),
        paragraphs=len(paragraphs),
        questions=len(questions),
    )

    paths.data_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(paths.paragraphs, paragraphs)
    write_jsonl(paths.questions, questions)

    build_lexical_index(paragraphs).save(paths.bm25_dir)
    log.info("bm25_saved", directory=str(paths.bm25_dir))

    embeddings_meta: EmbeddingsMeta | None = None
    if embedder is not None:
        vectors, embeddings_meta = await build_embeddings(
            paragraphs, embedder, batch_size=batch_size
        )
        write_embeddings(paths, vectors, embeddings_meta)
        log.info("embeddings_saved", shape=list(vectors.shape), model=embedder.model_id)

    manifest = BuildManifest(
        source_url=source,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        source_bytes=len(raw),
        squad_version=str(document.get("version", "unknown")),
        articles=len(document["data"]),
        paragraphs=len(paragraphs),
        questions=len(questions),
        built_at=_now(),
        embeddings=embeddings_meta,
    )
    write_manifest(paths, manifest)
    return manifest


async def run_embed_only(
    *, paths: CorpusPaths, embedder: Embedder, batch_size: int = 100
) -> BuildManifest:
    """Add or refresh dense vectors for an existing corpus without touching anything else."""
    paragraphs = read_jsonl(paths.paragraphs, Paragraph)
    vectors, meta = await build_embeddings(paragraphs, embedder, batch_size=batch_size)
    write_embeddings(paths, vectors, meta)
    manifest = read_manifest(paths).model_copy(update={"embeddings": meta})
    write_manifest(paths, manifest)
    log.info("embeddings_saved", shape=list(vectors.shape), model=embedder.model_id)
    return manifest


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
