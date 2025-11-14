"""
A minimal Tornado web application for querying the hplusroadmap RAG index.

This module defines a simple HTTP API with a single endpoint, ``/search``,
that accepts a JSON payload containing a ``query`` field.  It embeds
the query, retrieves the most relevant chat chunks from the vector
database, and uses an LLM via litellm/OpenRouter to generate a
response.  The response is returned as JSON.

The application is intentionally minimal: it does not include any
templating or JavaScript but serves as a backend for a future user
interface.  Tornado's asynchronous capabilities allow the long
running embedding and LLM calls to be executed without blocking the
event loop.

Prerequisites:

* ``openai`` – to access the OpenRouter embeddings endpoint via the
  OpenAI client.  See the Snyk example for how to configure the
  client with ``base_url`` and ``api_key``.
* ``litellm`` – to proxy chat completions through OpenRouter.
* ``chromadb`` – used here as the vector store.  You can substitute
  any other vector database that implements a similar API.
* ``tornado`` – the web framework.

Set the ``OPENROUTER_API_KEY`` environment variable to your
OpenRouter API key before starting the server.  For local
development you can supply a dummy referer in the ``HTTP-Referer``
header to satisfy OpenRouter's requirements.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List

import tornado.ioloop
import tornado.web

try:
    import openai
    import litellm
    import chromadb
    from chromadb.config import Settings
except ImportError as exc:
    raise RuntimeError(
        "Required packages are missing; please install openai, litellm and chromadb before running the web app"
    ) from exc


class SearchHandler(tornado.web.RequestHandler):
    """Handle POST /search requests to perform semantic search and LLM answer generation."""

    def initialize(self, index_dir: Path, top_k: int, chat_model: str) -> None:
        self.index_dir = index_dir
        self.top_k = top_k
        self.chat_model = chat_model
        settings = Settings(chroma_db_impl="duckdb", persist_directory=str(self.index_dir))
        self.chroma = chromadb.Client(settings)
        self.collection = self.chroma.get_or_create_collection(name="hplus_index")
        self.embed_client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            default_headers={"HTTP-Referer": "http://localhost"},
        )

    async def post(self) -> None:
        try:
            body: Any = json.loads(self.request.body.decode("utf-8"))
        except json.JSONDecodeError:
            self.set_status(400)
            self.finish({"error": "Invalid JSON"})
            return
        query: str | None = body.get("query")
        if not query:
            self.set_status(400)
            self.finish({"error": "Missing 'query' field"})
            return
        embed_resp = self.embed_client.embeddings.create(model="qwen/qwen3-embedding-8b", input=[query])
        query_embedding = embed_resp.data[0].embedding
        results = self.collection.query(query_embeddings=[query_embedding], n_results=self.top_k)
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        context_parts: List[str] = []
        for doc, meta in zip(documents, metadatas):
            context_parts.append(f"[From {meta['file']} lines {meta['start']}-{meta['end']}]\n{doc}")
        context = "\n\n".join(context_parts)
        prompt = (
            "You are an assistant with access to the hplusroadmap IRC logs.\n"
            "Answer the following question using the retrieved chat excerpts.\n"
            "If the logs do not contain the answer, say so.\n\n"
            f"Question: {query}\n\n"
            f"Retrieved Context:\n{context}\n\n"
            "Answer:"
        )
        llm_response = await tornado.ioloop.IOLoop.current().run_in_executor(
            None,
            lambda: litellm.completion(
                model=self.chat_model,
                messages=[{"role": "user", "content": prompt}],
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
            ),
        )
        answer = llm_response['choices'][0]['message']['content']
        self.finish({"answer": answer.strip()})


def make_app(index_dir: str | Path, top_k: int = 5, chat_model: str = "moonshotai/k2") -> tornado.web.Application:
    """Create and return the Tornado application."""
    return tornado.web.Application([
        (r"/search", SearchHandler, dict(index_dir=Path(index_dir), top_k=top_k, chat_model=chat_model)),
    ])


def run_server(port: int = 8888, index_dir: str = "data/index", top_k: int = 5, chat_model: str = "moonshotai/k2") -> None:
    """Run the Tornado web server on the specified port."""
    app = make_app(index_dir, top_k=top_k, chat_model=chat_model)
    app.listen(port)
    print(f"Server running at http://localhost:{port}/search")
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the hplusroadmap RAG web server.")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--index-dir", type=str, default="data/index")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chat-model", type=str, default="moonshotai/k2")
    args = parser.parse_args()
    run_server(port=args.port, index_dir=args.index_dir, top_k=args.top_k, chat_model=args.chat_model)

