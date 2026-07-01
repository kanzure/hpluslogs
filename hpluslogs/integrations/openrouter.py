"""OpenRouter API adapter for embeddings and completions."""

from __future__ import annotations

import os
from typing import List

# Lazy-loaded clients
_async_client = None
_sync_client = None


def get_async_client():
    """Get or create async OpenAI client for OpenRouter."""
    global _async_client
    if _async_client is None:
        import openai
        _async_client = openai.AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            default_headers={"HTTP-Referer": "http://localhost"},
        )
    return _async_client


def get_sync_client():
    """Get or create sync OpenAI client for OpenRouter."""
    global _sync_client
    if _sync_client is None:
        import openai
        _sync_client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            default_headers={"HTTP-Referer": "http://localhost"},
        )
    return _sync_client


async def embed_batch_async(texts: List[str], model: str = "qwen/qwen3-embedding-8b", retry: bool = True) -> List[List[float]]:
    """Asynchronously embed a batch of texts using the OpenRouter API.

    Returns a list of embedding vectors in the same order as the input texts.
    """
    client = get_async_client()
    try:
        resp = await client.embeddings.create(model=model, input=texts)
    except ValueError as err:
        if retry and "No embedding data received" in str(err):
            return await embed_batch_async(texts, model, retry=False)
        raise
    return [item.embedding for item in resp.data]


def embed_single(text: str, model: str = "qwen/qwen3-embedding-8b") -> List[float]:
    """Embed a single text synchronously."""
    client = get_sync_client()
    resp = client.embeddings.create(model=model, input=[text])
    return resp.data[0].embedding


def complete(prompt: str, model: str, timeout: float = 600.0) -> str:
    """Call LLM completion via litellm (through OpenRouter), return response text.

    Since this always routes through OpenRouter (fixed base URL + OPENROUTER_API_KEY),
    the model is normalized to carry the ``openrouter/`` prefix. This means callers
    can pass either ``openrouter/google/gemini-3.5-flash`` or the bare
    ``google/gemini-3.5-flash`` — without the prefix litellm would otherwise try to
    reach Google's API directly and fail with a "Provider List" error.

    ``timeout`` (seconds) caps how long we wait for the API so a hung connection
    fails cleanly instead of blocking forever; the caller can then retry.
    """
    import litellm
    # Silence litellm's noisy "Provider List: ..." / "Give Feedback" stderr spam,
    # which otherwise looks like a fatal error even when the call succeeds.
    litellm.suppress_debug_info = True
    litellm.set_verbose = False

    if not model.startswith("openrouter/"):
        model = "openrouter/" + model

    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        timeout=timeout,
    )
    return response['choices'][0]['message']['content']
