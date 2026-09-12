"""Turn the SQuAD dev file into the corpus files the service reads.

Input: the canonical ``dev-v1.1.json`` from the SQuAD project, fetched from its URL or read
from disk. Output: the files described in ``corpus.py``.

The transformation deliberately does nothing clever. Paragraphs are kept verbatim in source
order, each gets a stable id and its Wikipedia URL, and every gold question records the
paragraph it was written from. That last fact is what makes retrieval evaluation exact rather
than approximate. The dense vectors are the index, so a corpus without them cannot be searched:
``--skip-embeddings`` writes the text files only, for inspection or for a later
``--only-embeddings`` run once a key is available.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
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


def parse_squad(
    document: dict[str, Any], *, max_articles: int | None = None
) -> tuple[list[Paragraph], list[GoldQuestion]]:
    """Flatten SQuAD's article, paragraph, question nesting into two lists.

    Order is preserved so ids are reproducible. Duplicate answer strings from multiple
    annotators are collapsed while keeping first-seen order.

    ``max_articles`` keeps only the first few articles in source order, which is how the
    corpus is sized to one day of the free embedding quota.
    """
    paragraphs: list[Paragraph] = []
    questions: list[GoldQuestion] = []
    for article in document["data"][:max_articles]:
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


class EmbeddingQuotaError(RuntimeError):
    """The provider's daily embedding quota ran out. Progress is saved; rerun later to resume."""

    def __init__(self, done: int, total: int) -> None:
        super().__init__(f"embedding quota exhausted after {done} of {total} texts")
        self.done = done
        self.total = total


_RETRY_DELAY = re.compile(r"retry in ([0-9.]+)s", re.IGNORECASE)
_MAX_BATCH_ATTEMPTS = 6


def _is_rate_limit(exc: BaseException) -> bool:
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text or "429" in text


def _retry_delay_seconds(exc: BaseException) -> float | None:
    match = _RETRY_DELAY.search(str(exc))
    return float(match.group(1)) if match else None


def _is_daily_quota(exc: BaseException) -> bool:
    # Google names the violated quota; the per-day one contains "PerDay", the per-minute one
    # "PerMinute". A retry-after longer than a couple of minutes means the same thing: stop and
    # resume later rather than sleep inside the build.
    return "PerDay" in str(exc) or (_retry_delay_seconds(exc) or 0.0) > 120.0


async def _embed_batch(
    embedder: Embedder, batch: Sequence[str], *, done: int, total: int
) -> np.ndarray:
    """One batch, honouring the provider's retry-after on a per-minute limit."""
    for attempt in range(1, _MAX_BATCH_ATTEMPTS + 1):
        try:
            return await embedder.embed_documents(batch)
        except Exception as exc:
            if _is_daily_quota(exc):
                raise EmbeddingQuotaError(done, total) from exc
            if not _is_rate_limit(exc) or attempt == _MAX_BATCH_ATTEMPTS:
                raise
            delay = (_retry_delay_seconds(exc) or 30.0) + 1.0
            log.warning("embedding_rate_limited", attempt=attempt, sleep_s=round(delay, 1))
            await asyncio.sleep(delay)
    raise RuntimeError("unreachable")


def _load_partial(paths: CorpusPaths, embedder: Embedder, fingerprint: str) -> np.ndarray | None:
    """Vectors saved by an earlier run of the same corpus and model, or ``None``."""
    if not (paths.embeddings_partial.exists() and paths.embeddings_partial_meta.exists()):
        return None
    meta = EmbeddingsMeta.model_validate_json(
        paths.embeddings_partial_meta.read_text(encoding="utf-8")
    )
    same = (
        meta.paragraphs_sha256 == fingerprint
        and meta.model == embedder.model_id
        and meta.dimensions == embedder.dimensions
    )
    if not same:
        log.warning("embeddings_partial_discarded", reason="corpus or model changed")
        return None
    loaded: np.ndarray = np.load(paths.embeddings_partial)
    if loaded.shape[0] != meta.count:
        log.warning(
            "embeddings_partial_discarded", reason="progress file does not match its record"
        )
        return None
    return loaded


def _save_partial(
    paths: CorpusPaths, vectors: np.ndarray, embedder: Embedder, fingerprint: str
) -> None:
    np.save(paths.embeddings_partial, vectors)
    meta = EmbeddingsMeta(
        model=embedder.model_id,
        dimensions=embedder.dimensions,
        count=int(vectors.shape[0]),
        paragraphs_sha256=fingerprint,
        created_at=_now(),
    )
    paths.embeddings_partial_meta.write_text(
        meta.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def discard_partial(paths: CorpusPaths) -> None:
    """Remove the progress files once the full index is written."""
    paths.embeddings_partial.unlink(missing_ok=True)
    paths.embeddings_partial_meta.unlink(missing_ok=True)


async def build_embeddings(
    paragraphs: Sequence[Paragraph],
    embedder: Embedder,
    *,
    batch_size: int = 100,
    paths: CorpusPaths | None = None,
    texts_per_minute: int | None = None,
) -> tuple[np.ndarray, EmbeddingsMeta]:
    """Embed every paragraph in batches. Returns float16 unit vectors and their metadata.

    With ``paths`` the build is resumable: progress is saved after every batch and a rerun
    continues where the last one stopped, as long as the corpus and model are unchanged. With
    ``texts_per_minute`` it paces itself to a quota that counts each text as a request. When the
    provider reports the daily quota spent, it saves progress and raises
    ``EmbeddingQuotaError`` instead of losing the work done so far.
    """
    texts = [indexed_text(p) for p in paragraphs]
    fingerprint = paragraphs_fingerprint(paragraphs)
    resumed = _load_partial(paths, embedder, fingerprint) if paths is not None else None
    chunks: list[np.ndarray] = [resumed] if resumed is not None else []
    done = int(resumed.shape[0]) if resumed is not None else 0
    if done:
        log.info("embeddings_resumed", done=done, total=len(texts))

    while done < len(texts):
        batch = texts[done : done + batch_size]
        batch_started = time.monotonic()
        try:
            chunks.append(await _embed_batch(embedder, batch, done=done, total=len(texts)))
        except EmbeddingQuotaError:
            if paths is not None and chunks:
                _save_partial(paths, np.vstack(chunks).astype(np.float16), embedder, fingerprint)
            raise
        done += len(batch)
        if paths is not None:
            _save_partial(paths, np.vstack(chunks).astype(np.float16), embedder, fingerprint)
        log.info("embedded", done=done, total=len(texts))
        if texts_per_minute and done < len(texts):
            pause = 60.0 * len(batch) / texts_per_minute - (time.monotonic() - batch_started)
            if pause > 0:
                log.info("embedding_paced", sleep_s=round(pause, 1))
                await asyncio.sleep(pause)

    vectors = np.vstack(chunks).astype(np.float16)
    meta = EmbeddingsMeta(
        model=embedder.model_id,
        dimensions=embedder.dimensions,
        count=len(paragraphs),
        paragraphs_sha256=fingerprint,
        created_at=_now(),
    )
    return vectors, meta


def write_embeddings(paths: CorpusPaths, vectors: np.ndarray, meta: EmbeddingsMeta) -> None:
    """Persist the dense index and its metadata side by side."""
    np.save(paths.embeddings, vectors)
    paths.embeddings_meta.write_text(
        meta.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def write_manifest(paths: CorpusPaths, manifest: BuildManifest) -> None:
    """Persist build provenance. LF line endings so the committed file is identical on Windows."""
    paths.manifest.write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def read_manifest(paths: CorpusPaths) -> BuildManifest:
    """Load build provenance."""
    return BuildManifest.model_validate_json(paths.manifest.read_text(encoding="utf-8"))


async def run_build(
    *,
    source: str,
    paths: CorpusPaths,
    embedder: Embedder | None,
    batch_size: int = 100,
    texts_per_minute: int | None = None,
    max_articles: int | None = None,
) -> BuildManifest:
    """Full build: fetch, parse, write the corpus files, then embed them."""
    raw, document = await load_source(source)
    source_articles = len(document["data"])
    paragraphs, questions = parse_squad(document, max_articles=max_articles)
    articles = min(source_articles, max_articles) if max_articles else source_articles
    log.info(
        "parsed",
        articles=articles,
        source_articles=source_articles,
        paragraphs=len(paragraphs),
        questions=len(questions),
    )

    paths.data_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(paths.paragraphs, paragraphs)
    write_jsonl(paths.questions, questions)

    embeddings_meta: EmbeddingsMeta | None = None
    if embedder is not None:
        vectors, embeddings_meta = await build_embeddings(
            paragraphs,
            embedder,
            batch_size=batch_size,
            paths=paths,
            texts_per_minute=texts_per_minute,
        )
        write_embeddings(paths, vectors, embeddings_meta)
        discard_partial(paths)
        log.info("embeddings_saved", shape=list(vectors.shape), model=embedder.model_id)

    manifest = BuildManifest(
        source_url=source,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        source_bytes=len(raw),
        squad_version=str(document.get("version", "unknown")),
        articles=articles,
        source_articles=source_articles,
        paragraphs=len(paragraphs),
        questions=len(questions),
        built_at=_now(),
        embeddings=embeddings_meta,
    )
    write_manifest(paths, manifest)
    return manifest


async def run_embed_only(
    *,
    paths: CorpusPaths,
    embedder: Embedder,
    batch_size: int = 100,
    texts_per_minute: int | None = None,
) -> BuildManifest:
    """Add or refresh dense vectors for an existing corpus without touching anything else.

    Resumable and quota-paced; see ``build_embeddings``.
    """
    paragraphs = read_jsonl(paths.paragraphs, Paragraph)
    vectors, meta = await build_embeddings(
        paragraphs,
        embedder,
        batch_size=batch_size,
        paths=paths,
        texts_per_minute=texts_per_minute,
    )
    write_embeddings(paths, vectors, meta)
    discard_partial(paths)
    manifest = read_manifest(paths).model_copy(update={"embeddings": meta})
    write_manifest(paths, manifest)
    log.info("embeddings_saved", shape=list(vectors.shape), model=embedder.model_id)
    return manifest


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
