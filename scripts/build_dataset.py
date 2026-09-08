"""Build the corpus files in ``data/`` from the SQuAD dev set.

Usage, from the repository root::

    python -m scripts.build_dataset                     # corpus, BM25 index, embeddings
    python -m scripts.build_dataset --skip-embeddings   # no model key needed
    python -m scripts.build_dataset --only-embeddings   # add vectors to an existing corpus

The heavy lifting is in ``app.retrieval.build``; this file only parses arguments, decides
whether an embedder is available, and reports what was written. The outputs are committed to
the repository so a reviewer can clone and run without repeating this step.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import structlog

from app.config import get_settings
from app.llm.embeddings import Embedder, GeminiEmbedder
from app.observability.logging import configure_logging
from app.retrieval.build import SQUAD_DEV_URL, run_build, run_embed_only
from app.retrieval.corpus import CorpusPaths

log = structlog.get_logger("scripts.build_dataset")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", default=SQUAD_DEV_URL, help="URL or path of dev-v1.1.json")
    parser.add_argument("--data-dir", default=None, help="Output directory (default: DATA_DIR)")
    parser.add_argument("--batch-size", type=int, default=100, help="Texts per embedding request")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--skip-embeddings", action="store_true", help="Build without dense vectors")
    mode.add_argument("--only-embeddings", action="store_true", help="Only (re)build dense vectors")
    return parser.parse_args(argv)


def _make_embedder() -> Embedder | None:
    settings = get_settings()
    if settings.gemini_api_key is None:
        return None
    return GeminiEmbedder(
        api_key=settings.gemini_api_key.get_secret_value(),
        model=settings.gemini_embedding_model,
        dimensions=settings.embedding_dimensions,
        timeout_s=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
    )


async def _main(argv: list[str]) -> int:
    args = _parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level, as_json=False)
    paths = CorpusPaths(settings.data_dir if args.data_dir is None else Path(args.data_dir))

    embedder = None if args.skip_embeddings else _make_embedder()
    if args.only_embeddings:
        if embedder is None:
            log.error("no_embedder", hint="set GEMINI_API_KEY to build embeddings")
            return 2
        manifest = await run_embed_only(paths=paths, embedder=embedder, batch_size=args.batch_size)
    else:
        if embedder is None and not args.skip_embeddings:
            log.warning(
                "no_embedder", note="GEMINI_API_KEY not set; building without dense vectors"
            )
        manifest = await run_build(
            source=args.source, paths=paths, embedder=embedder, batch_size=args.batch_size
        )

    log.info(
        "build_complete",
        data_dir=str(paths.data_dir),
        paragraphs=manifest.paragraphs,
        questions=manifest.questions,
        embeddings=manifest.embeddings.model if manifest.embeddings else None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main(sys.argv[1:])))
