"""Search service - retrieves relevant chunks from any backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from hpluslogs.integrations import chroma, openrouter, xai

BackendType = Literal["chroma", "xai"]


def retrieve(
    data_dir: Path,
    query: str,
    top_k: int,
    backend: BackendType = "chroma",
    collection_id: Optional[str] = None,
    search_mode: str = "hybrid",
    filter_str: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve relevant chunks for a query.
    
    Returns list of dicts with 'content', 'score', 'metadata' keys.
    """
    if backend == "chroma":
        return _retrieve_chroma(data_dir, query, top_k)
    elif backend == "xai":
        return _retrieve_xai(collection_id, query, top_k, search_mode, filter_str)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _retrieve_chroma(data_dir: Path, query: str, top_k: int) -> List[Dict[str, Any]]:
    """Retrieve from local Chroma database."""
    index_dir = data_dir / "index"
    collection = chroma.get_collection(index_dir)
    query_embedding = openrouter.embed_single(query)
    results = chroma.search(collection, query_embedding, top_k)
    
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    
    return [
        {"content": doc, "score": None, "metadata": meta}
        for doc, meta in zip(documents, metadatas)
    ]


def _retrieve_xai(
    collection_id: Optional[str],
    query: str,
    top_k: int,
    mode: str,
    filter_str: Optional[str],
) -> List[Dict[str, Any]]:
    """Retrieve from xAI Collections."""
    if not collection_id:
        raise ValueError("collection_id required for xAI backend")
    return xai.search_collection(collection_id, query, top_k, mode, filter_str)
