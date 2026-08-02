"""Pure utility functions with no external service dependencies."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Iterator

import tiktoken


def daterange(start_date: _dt.date, end_date: _dt.date) -> Iterator[_dt.date]:
    """Yield dates from ``start_date`` up to and including ``end_date``.

    >>> list(daterange(date(2025, 11, 10), date(2025, 11, 12)))
    [date(2025, 11, 10), date(2025, 11, 11), date(2025, 11, 12)]
    """
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + _dt.timedelta(n)


def ensure_directory(path: Path) -> None:
    """Ensure that ``path`` exists, creating it and all parents if necessary."""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def read_human_text(path: Path) -> str:
    """Read text files that may contain legacy IRC/archive encodings.

    Prefer strict UTF-8, then fall back to Windows-1252 (common for smart quotes
    like byte 0x91), and finally Latin-1 so one malformed byte cannot abort a
    batch job.
    """
    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def token_count(text: str, model: str = "o200k_base") -> int:
    """Return the number of tokens in ``text`` for the specified tokeniser.

    The tiktoken library ships with multiple encodings.  The OpenAI
    cookbook suggests using the ``encoding_for_model`` helper for
    supported chat models.  Here we expose the same via an explicit
    model parameter.  See the tiktoken README for examples.
    """
    enc = tiktoken.get_encoding(model)
    return len(enc.encode(text))


def estimate_embedding_cost(texts, price_per_m: float = 0.01) -> float:
    """Estimate the cost in USD of embedding ``texts`` at $price_per_m per 1M tokens.

    OpenRouter currently charges $0.01 per million input tokens for the
    Qwen3 embedding model.  This function sums the token
    counts for all input texts and scales accordingly.  It returns the
    estimated cost in US dollars.
    """
    total_tokens = sum(token_count(t) for t in texts)
    return (total_tokens / 1_000_000.0) * price_per_m
