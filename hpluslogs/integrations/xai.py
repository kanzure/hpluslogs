"""xAI Collections API adapter."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_sync_client():
    """Get xAI sync client."""
    import xai_sdk
    return xai_sdk.Client(
        api_key=os.environ.get("XAI_API_KEY"),
        management_api_key=os.environ.get("XAI_MANAGEMENT_API_KEY"),
        timeout=3600,
    )


def get_async_client():
    """Get xAI async client."""
    import xai_sdk
    return xai_sdk.AsyncClient(
        api_key=os.environ.get("XAI_API_KEY"),
        management_api_key=os.environ.get("XAI_MANAGEMENT_API_KEY"),
        timeout=3600,
    )


def create_collection(
    name: str,
    chunk_size: int,
    chunk_overlap: int,
    field_definitions: List[Dict[str, Any]],
) -> str:
    """Create collection, return collection_id."""
    client = get_sync_client()
    collection = client.collections.create(
        name=name,
        chunk_configuration={
            "tokens_configuration": {
                "max_chunk_size_tokens": chunk_size,
                "chunk_overlap_tokens": chunk_overlap,
                "encoding_name": "o200k_base",
            },
            "strip_whitespace": True,
        },
        field_definitions=field_definitions,
    )
    return collection.collection_id


async def upload_document_async(
    client,
    collection_id: str,
    file_path: Path,
    fields: Dict[str, Any],
    wait_for_indexing: bool = False,
) -> str:
    """Upload single document asynchronously, return file_id."""
    data = file_path.read_bytes()
    doc = await client.collections.upload_document(
        collection_id=collection_id,
        name=file_path.name,
        data=data,
        fields=fields,
        wait_for_indexing=wait_for_indexing,
    )
    return doc.file_metadata.file_id


def search_collection(
    collection_id: str,
    query: str,
    limit: int,
    mode: str = "hybrid",
    filter_str: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search collection, return list of matches."""
    client = get_sync_client()
    kwargs = {
        "query": query,
        "collection_ids": [collection_id],
        "limit": limit,
        "retrieval_mode": mode,
    }
    if filter_str:
        kwargs["filter"] = filter_str
    results = client.collections.search(**kwargs)
    return [
        {
            "content": m.chunk_content,
            "score": m.score,
            "metadata": getattr(m, 'metadata', {}),
        }
        for m in results.matches
    ]
