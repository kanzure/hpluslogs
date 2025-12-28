"""Preprocessing service for chunking IRC logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import click

from hpluslogs.core.chunking import chunk_messages, enrich_chunk, parse_log_lines
from hpluslogs.core.utils import ensure_directory


def save_jsonl(chunks: List[dict], path: Path) -> None:
    """Write a list of chunk dictionaries to a JSON Lines file."""
    with path.open("w", encoding="utf-8") as f:
        for obj in chunks:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def run(
    data_dir: Path,
    max_tokens: int = 175,
    overlap: int = 20,
    enrich: bool = False,
) -> None:
    """Convert raw logs into JSONL files of overlapping message chunks.

    For each downloaded ``.log`` file in ``data_dir/raw``, this function
    parses the messages, segments them into overlapping sliding windows,
    optionally calls a summariser to enrich the text, and writes the result
    to ``data_dir/chunks/<YYYY-MM-DD>.jsonl``.
    """
    raw_dir = data_dir / "raw"
    chunk_dir = data_dir / "chunks"
    ensure_directory(chunk_dir)
    
    for log_file in sorted(raw_dir.glob("*.log"))[-22:]:
        out_file = chunk_dir / (log_file.stem + ".jsonl")
        click.echo(f"Processing {log_file.name} …")
        lines = parse_log_lines(log_file.read_text(encoding="utf-8", errors="replace"))
        messages = list(lines)
        chunks = chunk_messages(messages, max_tokens=max_tokens, overlap=overlap)
        if enrich:
            for c in chunks:
                c["enriched_text"] = enrich_chunk(c["text"])
        save_jsonl(chunks, out_file)
        click.echo(f"Wrote {len(chunks)} chunks to {out_file}")
