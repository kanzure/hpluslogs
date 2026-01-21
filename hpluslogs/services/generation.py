"""Generation service - builds prompts and calls LLMs."""

from __future__ import annotations

from typing import Any, Dict, List

from litellm import completion as litellm_completion

from hpluslogs.core.prompts import (
    AAF_SEARCH_QUERY_PROMPT,
    AAF_SYSTEM_PROMPT,
    CONTEXT_CLEANING_PROMPT,
    FIGHTAGING_SEARCH_QUERY_PROMPT,
    FIGHTAGING_SYSTEM_PROMPT,
    GRG_SEARCH_QUERY_PROMPT,
    GRG_SYSTEM_PROMPT,
    LESSWRONG_SEARCH_QUERY_PROMPT,
    LESSWRONG_SYSTEM_PROMPT,
    ORIONSARM_SEARCH_QUERY_PROMPT,
    ORIONSARM_SYSTEM_PROMPT,
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
    system_in_user: bool = True,
    context_override: str = None,
) -> str:
    """Generate an answer using retrieved context.
    
    Args:
        query: The search query.
        retrieved_chunks: List of retrieved chunks with content and metadata.
        model: The LLM model to use.
        prompt_fragment: Additional user instructions.
        system_prompt: The system prompt (or instructions to append if system_in_user=True).
        system_in_user: If True, system_prompt is appended to user message instead of being a system message.
        context_override: If provided, use this instead of formatting retrieved_chunks.
    """
    if context_override is not None:
        context = context_override
    else:
        context = format_context(retrieved_chunks)
    prompt = build_rag_prompt(query, context, prompt_fragment, system_prompt, system_in_user)
    answer = openrouter.complete(prompt, model)
    return f"Search query: {query}\n\n{answer}"


def generate_answer_no_context(
    query: str,
    model: str = "openrouter/x-ai/grok-4-fast",
    prompt_fragment: str = "",
    system_prompt: str = RAG_SYSTEM_PROMPT,
    system_in_user: bool = True,
) -> str:
    """Generate an answer without any RAG context (baseline LLM response).
    
    Args:
        query: The search query.
        model: The LLM model to use.
        prompt_fragment: Additional user instructions.
        system_prompt: The system prompt (or instructions to append if system_in_user=True).
        system_in_user: If True, system_prompt is appended to user message instead of being a system message.
    """
    prompt = build_rag_prompt(query, "", prompt_fragment, system_prompt, system_in_user)
    answer = openrouter.complete(prompt, model)
    return answer


def generate_search_query(
    prompt_fragment: str,
    model: str,
    for_fightaging: bool = False,
    for_lesswrong: bool = False,
    for_orionsarm: bool = False,
    for_grg: bool = False,
    for_aaf: bool = False,
) -> str:
    """Generate search terms from a user request."""
    if for_fightaging:
        prompt = FIGHTAGING_SEARCH_QUERY_PROMPT.format(prompt_fragment=prompt_fragment)
    elif for_lesswrong:
        prompt = LESSWRONG_SEARCH_QUERY_PROMPT.format(prompt_fragment=prompt_fragment)
    elif for_orionsarm:
        prompt = ORIONSARM_SEARCH_QUERY_PROMPT.format(prompt_fragment=prompt_fragment)
    elif for_grg:
        prompt = GRG_SEARCH_QUERY_PROMPT.format(prompt_fragment=prompt_fragment)
    elif for_aaf:
        prompt = AAF_SEARCH_QUERY_PROMPT.format(prompt_fragment=prompt_fragment)
    else:
        prompt = SEARCH_QUERY_GENERATION_PROMPT.format(prompt_fragment=prompt_fragment)
    return openrouter.complete(prompt, model).strip()


def clean_context(
    context: str,
    #model: str = "openrouter/openai/gpt-oss-120b",
    model: str = "openrouter/x-ai/grok-4.1-fast",
) -> str:
    """Clean context by removing redundancy and formatting artifacts.
    
    Uses the specified model with Cerebras provider to clean up the context,
    removing email signatures, duplicate quoted emails, formatting errors,
    and other artifacts while preserving important information.
    """
    response = litellm_completion(
        model=model,
        messages=[
            {"role": "user", "content": f"{context}\n\n{CONTEXT_CLEANING_PROMPT}"}
        ],
        #extra_body={
        #    "provider": {
        #        "only": ["Cerebras"],
        #    }
        #},
    )
    
    return response.choices[0].message.content


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
