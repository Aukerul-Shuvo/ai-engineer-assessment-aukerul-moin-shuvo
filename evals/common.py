"""Shared evaluation helpers: gold loading, sampling, SQuAD scoring, report writing.

Scoring follows the official SQuAD evaluation script: answers are lowercased, punctuation and
the articles a/an/the are removed, whitespace is collapsed, and exact match and token F1 are
taken as the maximum over the accepted gold answers.
"""

from __future__ import annotations

import json
import random
import re
import string
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.retrieval.corpus import CorpusPaths, GoldQuestion, read_jsonl

REPORTS_DIR = Path("evals/reports")

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def load_gold(paths: CorpusPaths, *, limit: int = 0, seed: int = 42) -> list[GoldQuestion]:
    """All gold questions, or a reproducible random sample of ``limit`` of them."""
    questions = read_jsonl(paths.questions, GoldQuestion)
    if limit and limit < len(questions):
        rng = random.Random(seed)  # noqa: S311 - reproducible sampling, not security
        questions = rng.sample(questions, limit)
    return questions


def normalize_answer(text: str) -> str:
    """The SQuAD normalisation."""
    lowered = text.lower()
    no_punct = "".join(ch for ch in lowered if ch not in _PUNCT)
    no_articles = _ARTICLES.sub(" ", no_punct)
    return " ".join(no_articles.split())


def exact_match(prediction: str, golds: Sequence[str]) -> float:
    """1.0 if the normalised prediction equals any normalised gold answer."""
    normalised = normalize_answer(prediction)
    return float(any(normalised == normalize_answer(gold) for gold in golds))


def f1(prediction: str, golds: Sequence[str]) -> float:
    """Token F1, maximised over gold answers."""
    pred_tokens = normalize_answer(prediction).split()
    best = 0.0
    for gold in golds:
        gold_tokens = normalize_answer(gold).split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def contains_gold(text: str, golds: Sequence[str]) -> bool:
    """Whether any normalised gold answer appears inside the normalised text."""
    haystack = normalize_answer(text)
    return any(normalize_answer(gold) in haystack for gold in golds if gold.strip())


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile; 0.0 for an empty sequence."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * len(ordered) + 0.5) - 1))
    return ordered[index]


def write_report(name: str, payload: dict[str, Any]) -> Path:
    """Persist a run as timestamped JSON under ``evals/reports``."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = REPORTS_DIR / f"{name}_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """A GitHub-flavoured markdown table."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)
