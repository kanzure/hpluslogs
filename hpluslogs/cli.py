"""
Command line utilities for building and querying a retrieval‑augmented
generation (RAG) system over the public IRC logs of the ``#hplusroadmap``
channel.  These utilities are designed to work with the rest of the
project (see ``README.md``) and illustrate how to download, preprocess,
embed and query chat data.  They rely on ``click`` for the command line
interface, ``requests`` for HTTP downloads, ``tiktoken`` for token
estimation and ``openai``/``litellm`` for OpenRouter integration.

Key features:

* **Download logs** – fetch raw IRC logs from https://gnusha.org/logs/ with
  optional start/end date filters.  Existing files are reused so that
  interrupted downloads can be resumed without re‑downloading.
* **Preprocess logs** – normalise the raw logs into a simple JSONL
  representation and cut them into overlapping sliding windows.  The
  overlap size and chunk length are configurable.  Optionally the user
  can request summarisation or enrichment via an LLM; a stub is
  provided here for future integration.
* **Embed chunks** – compute Qwen3 embedding vectors for each chunk via
  OpenRouter.  A cost estimation routine counts tokens using
  ``tiktoken`` and warns the user if the requested work exceeds a user
  provided budget.  Embeddings and associated metadata are stored in
  a local vector database via Chroma (or Qdrant if preferred).
* **Query** – simple search utility that embeds a user query, retrieves
  the nearest neighbours from the vector database, then calls an
  LLM to generate a response using the retrieved context.  This
  leverages ``litellm`` and OpenRouter for flexible model selection.

The functions below are intentionally verbose and include inline
documentation so that they can serve as a starting point for your own
customisation.  Several parts of the pipeline – particularly the
``enrich_chunk`` function used to add LLM‑generated metadata to each
chunk – are implemented as no‑ops by default.  They can be extended
with calls to the larger Qwen or Moonshot models via litellm once
appropriate API keys are configured.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import os
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

import click
import requests
import tiktoken

# Conditionally import openai and litellm to avoid hard dependency at
# module import time.  These packages are only needed for embedding
# and generation steps.  Users can install them via ``uv pip
# install openai litellm`` once they are ready to run those parts of
# the pipeline.
try:
    import openai
except ImportError:
    openai = None  # type: ignore
try:
    import litellm
except ImportError:
    litellm = None  # type: ignore


###############################################################################
# Helper utilities
###############################################################################

def daterange(start_date: _dt.date, end_date: _dt.date) -> Iterator[_dt.date]:
    """Yield dates from ``start_date`` up to and including ``end_date``.

    >>> list(daterange(date(2025, 11, 10), date(2025, 11, 12)))
    [date(2025, 11, 10), date(2025, 11, 11), date(2025, 11, 12)]
    """
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + _dt.timedelta(n)


def ensure_directory(path: Path) -> None:
    """Ensure that ``path`` exists, creating it and all parents if necessary."""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def token_count(text: str, model: str = "o200k_base") -> int:
    """Return the number of tokens in ``text`` for the specified tokeniser.

    The tiktoken library ships with multiple encodings.  The OpenAI
    cookbook suggests using the ``encoding_for_model`` helper for
    supported chat models.  Here we expose the same via an explicit
    model parameter.  See the tiktoken README for examples.
    """
    enc = tiktoken.get_encoding(model)
    return len(enc.encode(text))


def estimate_embedding_cost(texts: Iterable[str], price_per_m: float = 0.01) -> float:
    """Estimate the cost in USD of embedding ``texts`` at $price_per_m per 1M tokens.

    OpenRouter currently charges $0.01 per million input tokens for the
    Qwen3 embedding model.  This function sums the token
    counts for all input texts and scales accordingly.  It returns the
    estimated cost in US dollars.
    """
    total_tokens = sum(token_count(t) for t in texts)
    return (total_tokens / 1_000_000.0) * price_per_m


def fetch_log(date: _dt.date) -> str:
    """Download a single day's IRC log as text.

    The logs are hosted at ``https://gnusha.org/logs/`` with filenames
    formatted as ``YYYY-MM-DD.log``.  This function retrieves the
    plain text version.  It raises ``requests.HTTPError`` on failure.
    """
    url = f"https://gnusha.org/logs/{date.isoformat()}.log"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


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
    ``max_tokens`` (defaults to 175) and overlaps with the previous
    chunk by ``overlap`` tokens, following the sliding window technique
    recommended in recent RAG research.  Metadata such as
    timestamps and indices are retained in the returned dictionaries.
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


def save_jsonl(chunks: List[dict], path: Path) -> None:
    """Write a list of chunk dictionaries to a JSON Lines file."""
    with path.open("w", encoding="utf-8") as f:
        for obj in chunks:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


###############################################################################
# CLI Commands
###############################################################################

@click.group()
@click.option("--data-dir", type=click.Path(file_okay=False, dir_okay=True), default="data",
              help="Directory where logs, processed data and embeddings are stored.")
@click.pass_context
def cli(ctx: click.Context, data_dir: str) -> None:
    """Entry point for the hplusroadmap RAG CLI."""
    ctx.obj = {}
    ctx.obj["data_dir"] = Path(data_dir)
    ensure_directory(ctx.obj["data_dir"])


@cli.command()
@click.option("--start", "start_date", type=click.DateTime(formats=["%Y-%m-%d"]),
              help="First date to download (YYYY-MM-DD). Defaults to today if omitted.")
@click.option("--end", "end_date", type=click.DateTime(formats=["%Y-%m-%d"]),
              help="Last date to download (YYYY-MM-DD). Defaults to start date if omitted.")
@click.option("--resume/--no-resume", default=True,
              help="Skip files that already exist on disk (resume).")
@click.pass_obj
def download(obj: dict, start_date: Optional[_dt.datetime], end_date: Optional[_dt.datetime], resume: bool) -> None:
    """Download raw IRC logs from gnusha.org for a date range.

    The logs are saved in the ``data_dir/raw`` folder with filenames like
    ``YYYY-MM-DD.log``.  If ``resume`` is enabled (the default) then
    existing files will not be re‑downloaded.  Dates should be
    specified in ISO format (YYYY-MM-DD).  When no start date is
    provided this command defaults to today's date.
    """
    data_dir: Path = obj["data_dir"]
    raw_dir = data_dir / "raw"
    ensure_directory(raw_dir)
    today = _dt.date.today()
    if start_date is None:
        start = today
    else:
        start = start_date.date()
    if end_date is None:
        end = start
    else:
        end = end_date.date()
    for day in daterange(start, end):
        fname = raw_dir / f"{day.isoformat()}.log"
        if fname.exists() and resume:
            click.echo(f"✓ {fname.name} exists, skipping")
            continue
        click.echo(f"Downloading {day} …")
        try:
            text = fetch_log(day)
        except Exception as e:
            click.echo(f"Error downloading {day}: {e}")
            continue
        fname.write_text(text, encoding="utf-8")
        click.echo(f"Saved {fname}")


@cli.command()
@click.option("--max-tokens", default=175, help="Maximum tokens per chunk (sliding window).")
@click.option("--overlap", default=20, help="Token overlap between consecutive chunks.")
@click.option("--enrich/--no-enrich", default=False,
              help="Call an LLM to enrich or summarise each chunk.")
@click.pass_obj
def preprocess(obj: dict, max_tokens: int, overlap: int, enrich: bool) -> None:
    """Convert raw logs into JSONL files of overlapping message chunks.

    For each downloaded ``.log`` file in ``data_dir/raw``, this command
    parses the messages, segments them into overlapping sliding windows,
    optionally calls a summariser to enrich the text, and writes the result
    to ``data_dir/chunks/<YYYY-MM-DD>.jsonl``.
    """
    data_dir: Path = obj["data_dir"]
    raw_dir = data_dir / "raw"
    chunk_dir = data_dir / "chunks"
    ensure_directory(chunk_dir)
    for log_file in sorted(raw_dir.glob("*.log")):
        out_file = chunk_dir / (log_file.stem + ".jsonl")
        click.echo(f"Processing {log_file.name} …")
        lines = parse_log_lines(log_file.read_text(encoding="utf-8"))
        messages = list(lines)
        chunks = chunk_messages(messages, max_tokens=max_tokens, overlap=overlap)
        if enrich:
            for c in chunks:
                c["enriched_text"] = enrich_chunk(c["text"])
        save_jsonl(chunks, out_file)
        click.echo(f"Wrote {len(chunks)} chunks to {out_file}")


async def embed_batch_async(client: "openai.AsyncOpenAI", model: str, batch_texts: List[str]) -> List[List[float]]:
    """Asynchronously embed a batch of texts using the OpenRouter API.
    
    This helper function is called concurrently to parallelize embedding requests.
    It returns a list of embedding vectors in the same order as the input texts.
    """
    resp = await client.embeddings.create(model=model, input=batch_texts)
    return [item.embedding for item in resp.data]


async def embed_all_batches(
    client: "openai.AsyncOpenAI",
    model: str,
    texts: List[str],
    metadata: List[dict],
    batch_size: int,
    concurrency: int,
) -> Tuple[List[List[float]], List[str], List[dict], List[str]]:
    """Embed all texts in batches with controlled concurrency.
    
    This function splits the input texts into batches and processes up to
    ``concurrency`` batches in parallel using asyncio.Semaphore.  It returns
    four lists: embeddings, ids, metadatas and documents, ready to be added
    to the vector database.
    """
    semaphore = asyncio.Semaphore(concurrency)
    
    async def embed_with_semaphore(batch_idx: int, batch_texts: List[str], batch_meta: List[dict]) -> Tuple[int, List[List[float]]]:
        async with semaphore:
            click.echo(f"Embedding batch {batch_idx + 1} / {((len(texts)-1)//batch_size)+1}…")
            embeddings = await embed_batch_async(client, model, batch_texts)
            return (batch_idx, embeddings)
    
    tasks = []
    batches_info = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_meta = metadata[i:i + batch_size]
        batch_idx = i // batch_size
        batches_info.append((batch_texts, batch_meta))
        tasks.append(embed_with_semaphore(batch_idx, batch_texts, batch_meta))
    
    results = await asyncio.gather(*tasks)
    results_sorted = sorted(results, key=lambda x: x[0])
    
    all_embeddings = []
    all_ids = []
    all_metadatas = []
    all_documents = []
    
    for (batch_texts, batch_meta), (_, embeddings) in zip(batches_info, results_sorted):
        all_embeddings.extend(embeddings)
        all_ids.extend([f"{meta['file']}-{meta['index']}" for meta in batch_meta])
        all_metadatas.extend(batch_meta)
        all_documents.extend(batch_texts)
    
    return all_embeddings, all_ids, all_metadatas, all_documents


@cli.command()
@click.option("--cost-limit", type=float, default=1.0,
              help="Maximum allowable estimated cost in USD for embedding.")
@click.option("--provider", default="openrouter.ai/api/v1", help="Base URL for the OpenRouter API.")
@click.option("--model", default="qwen/qwen3-embedding-8b", help="Embedding model identifier.")
@click.option("--concurrency", type=int, default=5,
              help="Number of concurrent embedding requests to make in parallel.")
@click.pass_obj
def embed(obj: dict, cost_limit: float, provider: str, model: str, concurrency: int) -> None:
    """Embed all chunk files into a vector database using OpenRouter embeddings.

    Before invoking the remote API this command estimates the total
    token count across all chunks using tiktoken and warns the user if
    the estimated cost would exceed the configured ``cost_limit`` (in
    USD).  If the budget is sufficient, the embeddings are computed
    using ``openai.AsyncOpenAI`` with ``api_base`` pointing at OpenRouter.
    
    The ``--concurrency`` parameter controls how many embedding requests
    are made in parallel using asyncio.  Higher values speed up the process
    but may hit rate limits.  The embeddings are saved into a Chroma database
    under ``data_dir/index``.

    The Qwen3 embedding model currently costs $0.01 per million input
    tokens.  Adjust ``cost_limit`` as desired.
    """
    if openai is None:
        raise click.UsageError("The openai package is not installed. Please install it via `uv pip install openai`.")
    data_dir: Path = obj["data_dir"]
    index_dir = data_dir / "index"
    ensure_directory(index_dir)
    chunk_dir = data_dir / "chunks"
    texts: List[str] = []
    metadata: List[dict] = []
    for f in sorted(chunk_dir.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            obj_j = json.loads(line)
            texts.append(obj_j.get("enriched_text", obj_j["text"]))
            meta = {"file": f.stem, "index": obj_j["index"], "start": obj_j["start"], "end": obj_j["end"]}
            metadata.append(meta)
    est_cost = estimate_embedding_cost(texts)
    click.echo(f"Estimated embedding cost: ${est_cost:.4f} at $0.01/M tokens")
    if est_cost > cost_limit:
        click.echo(f"Aborting: estimated cost ${est_cost:.4f} exceeds cost limit ${cost_limit:.2f}")
        return
    client = openai.AsyncOpenAI(
        base_url=f"https://{provider}",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        default_headers={"HTTP-Referer": "http://localhost"},
    )
    try:
        import chromadb
    except ImportError:
        raise click.UsageError("Chroma is not installed. Please install it via `uv pip install chromadb`.")
    clientdb = chromadb.PersistentClient(path=str(index_dir))
    collection = clientdb.get_or_create_collection(name="hplus_index")
    batch_size = 32
    
    all_embeddings, all_ids, all_metadatas, all_documents = asyncio.run(
        embed_all_batches(client, model, texts, metadata, batch_size, concurrency)
    )
    
    click.echo(f"Adding {len(all_embeddings)} embeddings to the vector database…")
    collection.add(
        ids=all_ids,
        embeddings=all_embeddings,
        metadatas=all_metadatas,
        documents=all_documents
    )
    click.echo(f"Embedded {len(texts)} chunks and saved index to {index_dir}")


@cli.command()
@click.option("--model", default="openrouter/moonshotai/kimi-k2", help="Chat model to use for answering queries.")
@click.option("--top-k", default=20, help="Number of nearest neighbours to retrieve.")
@click.option("--nollm", default=False, help="Skip using an LLM (print context only).")
@click.option("--contextlimit", type=int, default=256000, help="Maximum tokens in context before aborting (0 for no limit).")
@click.option("--prompt-fragment", default="", help="Additional instructions to add to the LLM prompt.")
@click.argument("query")
@click.pass_obj
def query(obj: dict, model: str, top_k: int, nollm: bool, contextlimit: int, prompt_fragment: str, query: str) -> None:
    """Answer a user question by retrieving relevant IRC messages and calling an LLM.

    This command embeds the query using the Qwen3 embedding model, retrieves
    the ``top_k`` most similar chunks from the vector database, constructs
    a prompt that includes the retrieved context, and sends the prompt to
    the specified chat model via litellm/openrouter.  The answer is
    printed to stdout.
    """
    if openai is None or litellm is None:
        raise click.UsageError("The openai and litellm packages are required. Please install them via uv pip install openai litellm")
    data_dir: Path = obj["data_dir"]
    index_dir = data_dir / "index"
    try:
        import chromadb
    except ImportError:
        raise click.UsageError("Chroma is not installed. Please install it via `uv pip install chromadb`.")
    clientdb = chromadb.PersistentClient(path=str(index_dir))
    collection = clientdb.get_or_create_collection(name="hplus_index")
    embed_client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        default_headers={"HTTP-Referer": "http://localhost"},
    )
    embed_resp = embed_client.embeddings.create(model="qwen/qwen3-embedding-8b", input=[query])
    query_embedding = embed_resp.data[0].embedding
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    context_parts = []
    for doc, meta in zip(documents, metadatas):
        context_parts.append(f"[From {meta['file']} lines {meta['start']}-{meta['end']}]\n{doc}")
    context = "\n\n".join(context_parts)
    click.echo("\nContext: <context>" + context + "</context>\n\n")
    if nollm == True:
        click.echo("\nExiting due to nollm flag.")
        return
    if contextlimit > 0:
        context_tokens = token_count(context)
        if context_tokens > contextlimit:
            click.echo(f"Context too long: {context_tokens} tokens exceeds limit of {contextlimit}")
            return
        else:
            click.echo(f"Context length: {context_tokens}\n\n\n")

    prompt = (
        "You are an assistant with access to the hplusroadmap IRC logs.\n"
        "Answer the following question using the retrieved chat excerpts. Where possible, please include next to a specific reference a link to the IRC log that mentioned that or informed that line of your output based off of the date of the IRC log mapped to the following URL format in year, month, day format: https://gnusha.org/logs/2016-11-01.log which is for 2016-11-01 (November 1st, 2016) as an example. Please use markdown format and backticks around quotes from the IRC log excerpt (next to the URL that you provide).\n"
        "If the logs do not contain the answer, say so.\n\n"
    )
    if prompt_fragment:
        prompt_fragment2 = f"Extra prompt for you: <prompt>{prompt_fragment}</prompt>\n\n"
    else:
        prompt_fragment2 = ""
    prompt += (
        f"Search query: <prompt>{query}</prompt>\n\n"
        f"{prompt_fragment2}"
        f"Retrieved Context:\n<context>{context}</context>\n\n"
        "Answer:"
    )
    click.echo("Asking the LLM...\n")
    llm_response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    answer = llm_response['choices'][0]['message']['content']
    click.echo("\n" + answer.strip())


def main() -> None:
    """Invoke the CLI when run as a script."""
    cli()


if __name__ == "__main__":
    main()
