"""Generation service - builds prompts and calls LLMs."""

from __future__ import annotations

from typing import Any, Dict, List

from hpluslogs.core.prompts import (
    FIGHTAGING_SEARCH_QUERY_PROMPT,
    FIGHTAGING_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT,
    SEARCH_QUERY_GENERATION_PROMPT,
    build_rag_prompt,
)
from hpluslogs.integrations import openrouter


def generate_answer(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    model: str = "openrouter/x-ai/grok-4-fast",
    prompt_fragment: str = "",
    system_prompt: str = RAG_SYSTEM_PROMPT,
) -> str:
    """Generate an answer using retrieved context."""
    context = format_context(retrieved_chunks)
    prompt = build_rag_prompt(query, context, prompt_fragment, system_prompt)
    answer = openrouter.complete(prompt, model)
    return f"Search query: {query}\n\n{answer}"


def generate_search_query(prompt_fragment: str, model: str, for_fightaging: bool = False) -> str:
    """Generate search terms from a user request."""
    if for_fightaging:
        prompt = FIGHTAGING_SEARCH_QUERY_PROMPT.format(prompt_fragment=prompt_fragment)
    else:
        prompt = SEARCH_QUERY_GENERATION_PROMPT.format(prompt_fragment=prompt_fragment)
    return openrouter.complete(prompt, model).strip()


def format_context(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into context string."""
    parts = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        score = chunk.get("score")
        
        if "file" in meta:
            header = f"[From {meta['file']} lines {meta.get('start', '?')}-{meta.get('end', '?')}]"
        elif "date" in meta:
            score_str = f"{score:.4f}" if score is not None else "?"
            header = f"[Date: {meta['date']}, Score: {score_str}]"
        elif "source_type" in meta:
            filename = meta.get('filename', 'unknown')
            score_str = f"{score:.4f}" if score is not None else "?"
            header = f"[Source: {meta['source_type']}, File: {filename}, Score: {score_str}]"
        else:
            score_str = f"{score:.4f}" if score is not None else "?"
            header = f"[Score: {score_str}]"
        
        parts.append(f"{header}\n{chunk['content']}")
    return "\n\n".join(parts)
