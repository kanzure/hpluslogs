"""Chroma vector database adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

_client = None


def get_client(index_dir: Path):
    """Get or create Chroma persistent client."""
    global _client
    import chromadb
    if _client is None:
        _client = chromadb.PersistentClient(path=str(index_dir))
    return _client


def get_collection(index_dir: Path, name: str = "hplus_index"):
    """Get or create a Chroma collection."""
    client = get_client(index_dir)
    return client.get_or_create_collection(name=name)


def add_embeddings(
    collection,
    ids: List[str],
    embeddings: List[List[float]],
    documents: List[str],
    metadatas: List[Dict[str, Any]],
) -> None:
    """Add embeddings to collection."""
    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


def search(collection, query_embedding: List[float], n_results: int) -> Dict[str, Any]:
    """Search collection, return results dict."""
    return collection.query(query_embeddings=[query_embedding], n_results=n_results)
