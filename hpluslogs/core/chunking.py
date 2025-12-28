"""Text chunking and parsing utilities."""

from __future__ import annotations

from typing import Iterable, List, Tuple

from hpluslogs.core.utils import token_count


def parse_log_lines(raw_text: str) -> Iterable[Tuple[str, str]]:
    """Parse raw IRC log text into a stream of (timestamp, message) tuples.

    The logs use a simple space‑separated format where each line begins
    with a timestamp (HH:MM) followed by IRC events or messages.  This
    parser strips join/quit events and returns only lines containing
    user content (denoted by ``<nickname>``).  Lines without angle
    brackets are ignored.
    """
    for line in raw_text.splitlines():
        parts = line.split("<", 1)
        if len(parts) != 2:
            continue
        timestamp_part, rest = parts
        timestamp_str = timestamp_part.strip().split()[0] if timestamp_part.strip() else ""
        if ">" not in rest:
            continue
        _, message = rest.split(">", 1)
        yield (timestamp_str, message.strip())


def chunk_messages(messages: List[Tuple[str, str]], max_tokens: int = 175, overlap: int = 20) -> List[dict]:
    """Split a list of (timestamp, message) pairs into overlapping chunks.

    Each chunk concatenates messages until the token count reaches
    ``max_tokens`` (defaults to 175) and overlaps with
    the previous chunk by ``overlap`` tokens, following the sliding window
    technique recommended in recent RAG research.  Metadata such as timestamps
    and indices are retained in the returned dictionaries.
    """
    chunks: List[dict] = []
    buffer: List[str] = []
    buffer_tokens = 0
    i = 0
    for ts, msg in messages:
        token_len = token_count(msg)
        if buffer_tokens + token_len > max_tokens and buffer:
            chunk_text = "\n".join(buffer)
            chunks.append({
                "index": len(chunks),
                "start": i - len(buffer),
                "end": i,
                "text": chunk_text,
            })
            overlap_buffer: List[str] = []
            overlap_tokens = 0
            for m in reversed(buffer):
                overlap_tokens += token_count(m)
                overlap_buffer.insert(0, m)
                if overlap_tokens >= overlap:
                    break
            buffer = overlap_buffer.copy()
            buffer_tokens = sum(token_count(m) for m in buffer)
        buffer.append(msg)
        buffer_tokens += token_len
        i += 1
    if buffer:
        chunk_text = "\n".join(buffer)
        chunks.append({
            "index": len(chunks),
            "start": i - len(buffer),
            "end": i,
            "text": chunk_text,
        })
    return chunks


def enrich_chunk(text: str, model: str | None = None) -> str:
    """Optionally call a large LLM to enrich or summarise a chunk.

    In practice you might call a bigger Qwen3 model, K2 or Moonshot to
    generate a summary or extract key information.  The RAG best
    practices study notes that summarisation reduces redundant
    information and improves downstream generation.  This stub
    simply returns the input text unchanged; override it when ready.
    """
    return text
