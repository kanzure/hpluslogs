"""Embedding service for vectorizing chunks and storing in Chroma."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List, Set

import click

from hpluslogs.core.utils import ensure_directory, estimate_embedding_cost
from hpluslogs.integrations import chroma, openrouter


def load_embedding_progress(progress_file: Path) -> Set[str]:
    """Load the set of already-embedded chunk IDs from the progress file."""
    if not progress_file.exists():
        return set()
    completed = set()
    with progress_file.open("r", encoding="utf-8") as f:
        for line in f:
            completed.add(line.strip())
    return completed


def save_embedding_progress(progress_file: Path, chunk_id: str) -> None:
    """Append a completed chunk ID to the progress file."""
    with progress_file.open("a", encoding="utf-8") as f:
        f.write(chunk_id + "\n")


async def embed_and_store_batches(
    model: str,
    texts: List[dict],
    batch_size: int,
    concurrency: int,
    collection,
    progress_file: Path,
    completed_ids: Set[str],
) -> int:
    """Embed texts in batches and incrementally store them in the vector database.

    Returns the number of newly embedded chunks.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def embed_and_add_batch(batch_idx: int, batch_texts: List[dict]) -> int:
        async with semaphore:
            # Filter out already-completed chunks
            filtered_texts = []
            filtered_meta = []
            ids = []
            for text in batch_texts:
                chunk_id = text["chunk_id"]
                if chunk_id not in completed_ids:
                    filtered_texts.append(text["text"])
                    filtered_meta.append(text["meta"])
                    ids.append(chunk_id)

            if not filtered_texts:
                click.echo(f"Batch {batch_idx + 1}: all chunks already embedded, skipping")
                return 0

            click.echo(f"Embedding batch {batch_idx + 1} ({len(filtered_texts)} chunks)…")
            embeddings = await openrouter.embed_batch_async(filtered_texts, model)
            click.echo(f"Received embeddings for batch {batch_idx + 1}")

            # Add to collection immediately
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: chroma.add_embeddings(
                    collection,
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=filtered_meta,
                    documents=filtered_texts
                )
            )

            # Record progress
            for chunk_id in ids:
                save_embedding_progress(progress_file, chunk_id)
                completed_ids.add(chunk_id)

            click.echo(f"Batch {batch_idx + 1}: added {len(ids)} embeddings to database")

            del embeddings
            del filtered_texts
            del filtered_meta

            return len(ids)

    # Create batches
    tasks = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_idx = i // batch_size
        tasks.append(embed_and_add_batch(batch_idx, batch_texts))

    results = await asyncio.gather(*tasks)
    total = sum(results)

    del tasks
    del results

    return total


def run(
    data_dir: Path,
    cost_limit: float = 1.0,
    model: str = "qwen/qwen3-embedding-8b",
    concurrency: int = 80,
) -> None:
    """Embed all chunk files into a vector database using OpenRouter embeddings."""
    index_dir = data_dir / "index"
    ensure_directory(index_dir)
    chunk_dir = data_dir / "chunks"
    progress_file = data_dir / "embedding_progress.txt"

    # Load already-completed chunks
    completed_ids = load_embedding_progress(progress_file)
    click.echo(f"Found {len(completed_ids)} already-embedded chunks")

    # Load all chunks
    text_len = 0
    remaining_texts: List[dict] = []
    for f in sorted(chunk_dir.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            text_len += 1
            obj_j = json.loads(line)
            meta = {"file": f.stem, "index": obj_j["index"], "start": obj_j["start"], "end": obj_j["end"]}
            chunk_id = f"{meta['file']}-{meta['index']}"
            if chunk_id not in completed_ids:
                text_data = obj_j.get("enriched_text", obj_j["text"])
                remaining_texts.append({"text": text_data, "meta": meta, "chunk_id": chunk_id})

    if not remaining_texts:
        click.echo("All chunks have already been embedded!")
        return

    click.echo(f"Total chunks: {text_len}, remaining to embed: {len(remaining_texts)}")
    est_cost = estimate_embedding_cost([x["text"] for x in remaining_texts])
    click.echo(f"Estimated embedding cost for remaining chunks: ${est_cost:.4f} at $0.01/M tokens")
    if est_cost > cost_limit:
        click.echo(f"Aborting: estimated cost ${est_cost:.4f} exceeds cost limit ${cost_limit:.2f}")
        return

    collection = chroma.get_collection(index_dir)
    batch_size = 1000

    newly_embedded = asyncio.run(
        embed_and_store_batches(
            model, remaining_texts, batch_size, concurrency,
            collection, progress_file, completed_ids
        )
    )

    del remaining_texts

    click.echo(f"Successfully embedded {newly_embedded} new chunks and saved to {index_dir}")
    click.echo(f"Total embedded chunks: {len(completed_ids)}")
