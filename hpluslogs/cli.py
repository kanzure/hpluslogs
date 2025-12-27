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
import subprocess
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
try:
    import xai_sdk
except ImportError:
    xai_sdk = None  # type: ignore


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
    ``max_tokens`` (defaults to 175) and overlaps with
    the previous chunk by ``overlap`` tokens, following the sliding window
    technique recommended in recent RAG research.  Metadata such as timestamps
    and indices are retained in the returned dictionaries.
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


def load_embedding_progress(progress_file: Path) -> set:
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


async def embed_batch_async(client: "openai.AsyncOpenAI", model: str, batch_texts: List[str], retry: bool = True) -> List[List[float]]:
    """Asynchronously embed a batch of texts using the OpenRouter API.

    This helper function is called concurrently to parallelize embedding requests.
    It returns a list of embedding vectors in the same order as the input texts.
    """
    try:
        resp = await client.embeddings.create(model=model, input=batch_texts)
    except ValueError as err:
        if retry and "No embedding data received" in str(err):
            #return []
            # one-shot retry
            return await embed_batch_async(client, model, batch_texts, retry=False)
        raise  # re-raise any other ValueError
    return [item.embedding for item in resp.data]


async def embed_and_store_batches(
    client: "openai.AsyncOpenAI",
    model: str,
    texts: List[str],
    batch_size: int,
    concurrency: int,
    collection: "chromadb.Collection",
    progress_file: Path,
    completed_ids: set,
) -> int:
    """Embed texts in batches and incrementally store them in the vector database.

    This function processes batches with controlled concurrency and adds each
    batch to the Chroma collection immediately after embedding, reducing memory
    usage.  It also tracks progress so the script can resume if interrupted.
    Returns the number of newly embedded chunks.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def embed_and_add_batch(batch_idx: int, batch_texts: List[str]) -> int:
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
            embeddings = await embed_batch_async(client, model, filtered_texts)
            click.echo(f"Received embeddings for batch {batch_idx + 1}")

            # Add to collection immediately (runs in thread pool to avoid blocking)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: collection.add(
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

            # Explicitly free memory
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

    # Process all batches
    results = await asyncio.gather(*tasks)
    total = sum(results)

    # Explicitly free memory
    del tasks
    del results

    return total


@cli.command()
@click.option("--cost-limit", type=float, default=1.0,
              help="Maximum allowable estimated cost in USD for embedding.")
@click.option("--provider", default="openrouter.ai/api/v1", help="Base URL for the OpenRouter API.")
@click.option("--model", default="qwen/qwen3-embedding-8b", help="Embedding model identifier.")
@click.option("--concurrency", type=int, default=80,
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
    but may hit rate limits.  The embeddings are saved incrementally into
    a Chroma database under ``data_dir/index``.  Progress is tracked in
    ``data_dir/embedding_progress.txt`` so the script can resume if interrupted.

    The Qwen3 embedding model currently costs $0.01 per million input
    tokens.  Adjust ``cost_limit`` as desired.
    """
    if openai is None:
        raise click.UsageError("The openai package is not installed. Please install it via `uv pip install openai`.")
    data_dir: Path = obj["data_dir"]
    index_dir = data_dir / "index"
    ensure_directory(index_dir)
    chunk_dir = data_dir / "chunks"
    progress_file = data_dir / "embedding_progress.txt"

    # Load already-completed chunks
    completed_ids = load_embedding_progress(progress_file)
    click.echo(f"Found {len(completed_ids)} already-embedded chunks")

    # Load all chunks
    #texts: List[str] = []
    text_len = 0
    remaining_texts: List[dict] = []
    #metadata: List[dict] = []
    for f in sorted(chunk_dir.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            text_len += 1
            obj_j = json.loads(line)
            meta = {"file": f.stem, "index": obj_j["index"], "start": obj_j["start"], "end": obj_j["end"]}
            chunk_id = f"{meta['file']}-{meta['index']}"
            # Filter out already-completed chunks
            if chunk_id not in completed_ids:
                text_data = obj_j.get("enriched_text", obj_j["text"])
                #texts.append(text_data)
                #metadata.append(meta)
                remaining_texts.append({"text": text_data, "meta": meta, "chunk_id": chunk_id})

    len_texts = text_len
    remaining_texts2 = remaining_texts#[50_000:50_000 * 4]
    del remaining_texts
    remaining_texts = remaining_texts2

    if not remaining_texts:
        click.echo("All chunks have already been embedded!")
        return

    click.echo(f"Total chunks: {len_texts}, remaining to embed: {len(remaining_texts)}")
    est_cost = estimate_embedding_cost([x["text"] for x in remaining_texts])
    click.echo(f"Estimated embedding cost for remaining chunks: ${est_cost:.4f} at $0.01/M tokens")
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
    #batch_size = 32 * 3  # 32 seems to be about 5500 tokens, model says 32k input, so maybe 3x?
    #batch_size = 32 * 4 # dunno if this will work. testing with chunk size 500 tokens.
    batch_size = 1000

    # Embed and store incrementally
    newly_embedded = asyncio.run(
        embed_and_store_batches(
            client, model, remaining_texts, batch_size, concurrency,
            collection, progress_file, completed_ids
        )
    )

    # Explicitly free memory after embedding is complete
    #del metadata
    del remaining_texts

    click.echo(f"Successfully embedded {newly_embedded} new chunks and saved to {index_dir}")
    click.echo(f"Total embedded chunks: {len(completed_ids)}")


@cli.command()
#@click.option("--model", default="openrouter/moonshotai/kimi-k2", help="Chat model to use for answering queries.")
@click.option("--model", default="openrouter/x-ai/grok-4-fast", help="Chat model to use for answering queries.")
@click.option("--top-k", default=100, help="Number of nearest neighbours to retrieve.")
@click.option("--nollm", default=False, help="Skip using an LLM (print context only).")
@click.option("--contextlimit", type=int, default=1500000, help="Maximum tokens in context before aborting (0 for no limit).")
@click.option("--prompt-fragment", default="", help="Additional instructions to add to the LLM prompt.")
@click.option("--output-name", default=None, help="Base filename for output (without extension). If not provided, uses timestamp.")
@click.option("--css-file", default="wrap.css", help="CSS file to use with pandoc for HTML generation.")
@click.option("--upload/--no-upload", default=True, help="Upload files to server via scp.")
@click.option("--remote-user", default="bryan", help="Remote SSH user for upload.")
@click.option("--remote-host", default="gnusha.org", help="Remote SSH host for upload.")
@click.option("--remote-path", default="~/public_html/irc/chatgpt/hplusroadmap/", help="Remote path for upload.")
@click.argument("query", required=False, default="")
@click.pass_obj
def query(obj: dict, model: str, top_k: int, nollm: bool, contextlimit: int, prompt_fragment: str,
          output_name: Optional[str], css_file: str, upload: bool, remote_user: str,
          remote_host: str, remote_path: str, query: str) -> None:
    """Answer a user question by retrieving relevant IRC messages and calling an LLM.

    This command embeds the query using the Qwen3 embedding model, retrieves
    the ``top_k`` most similar chunks from the vector database, constructs
    a prompt that includes the retrieved context, and sends the prompt to
    the specified chat model via litellm/openrouter.  The answer is
    printed to stdout.

    If no query is provided but a prompt-fragment is given, the LLM will first
    generate appropriate search terms from the prompt-fragment, then use those
    to query the vector database.
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

    # If no query provided, generate search terms from prompt-fragment using LLM
    search_query = query
    if not query and prompt_fragment:
        click.echo("No query provided. Generating search terms from prompt-fragment using LLM...\n")
        query_generation_prompt = (
            "You are a search query generator for a semantic search system over IRC chat logs.\n"
            "Based on the user's request below, generate an effective search query.\n"
            "The query should be a concise list of key terms, concepts, or phrases that would help retrieve relevant messages.\n"
            "Return ONLY the search query text, nothing else.\n\n"
            f"User request: {prompt_fragment}\n\n"
            "Search query:"
        )
        query_gen_response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": query_generation_prompt}],
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
        search_query = query_gen_response['choices'][0]['message']['content'].strip()
        click.echo(f"Generated search query: {search_query}\n")
    elif not query and not prompt_fragment:
        raise click.UsageError("Either a query argument or --prompt-fragment must be provided.")

    embed_resp = embed_client.embeddings.create(model="qwen/qwen3-embedding-8b", input=[search_query])
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
        "You are an assistant with access to the hplusroadmap IRC logs. Your job is to answer the following question(s) using the retrieved chat excerpts by constructing an in-depth technical report. The question (query) will be provided below.\n"
        "If you choose to include a quote from the IRC logs, then please use markdown format and GitHub markdown formatted four-space block quotes for the IRC log excerpts that you use. Please edit the excerpt to remove extraneous content.\n"
        "Where you see papers referenced, please collect those references and display them in your answer, even if the referenced papers may not be exactly related (e.g. they might be conceptually close, so include them in your output). Where you see companies mentioned, like a new company or a name of a company, or the name of people involved in different projects or ventures, or the name of different involved people, please list those in the answer as well. You are writing for a highly technical audience that is deeply interested in esoteric knowledge, technology, engineering, tech development, research, brainstorming, and speculation.\n"
        "If the logs do not contain the answer, say so. Your job is to extract the most relevant matching results and formulate it into a markdown-formatted document for readability. The question(s) or query that you have to answer is provided below (the user request and the search_query).\n\n"
        "\n\n"
    )
    if prompt_fragment:
        prompt_fragment2 = f"User request: <prompt>{prompt_fragment}</prompt>\n\n"
    else:
        prompt_fragment2 = ""

    # Include the search query used (whether original or generated)
    search_info = f"Search query used: <search_query>{search_query}</search_query>\n\n"

    prompt += (
        f"{prompt_fragment2}"
        f"{search_info}"
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
    answer = f"Search query: {search_query}\n\n"
    answer += llm_response['choices'][0]['message']['content']
    click.echo("\n" + answer.strip())

    # Save output to files
    data_dir: Path = obj["data_dir"]
    output_dir = data_dir / "outputs"
    ensure_directory(output_dir)

    # Generate filename
    if output_name is None:
        timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"query_{timestamp}"
    else:
        base_name = output_name

    md_file = output_dir / f"{base_name}.md"
    html_file = output_dir / f"{base_name}.md.html"

    # Save markdown file
    md_file.write_text(answer.strip(), encoding="utf-8")
    click.echo(f"\n✓ Saved markdown to {md_file}")

    # Generate HTML with pandoc
    try:
        pandoc_cmd = [
            "pandoc",
            "-f", "markdown+autolink_bare_uris",
            "-s",
            "-c", css_file,
            str(md_file),
            "-o", str(html_file)
        ]
        subprocess.run(pandoc_cmd, check=True, capture_output=True)
        click.echo(f"✓ Generated HTML at {html_file}")
    except subprocess.CalledProcessError as e:
        click.echo(f"⚠ Warning: pandoc failed: {e.stderr.decode()}")
    except FileNotFoundError:
        click.echo("⚠ Warning: pandoc not found. Install pandoc to generate HTML.")

    # Upload files via scp
    if upload:
        remote_dest = f"{remote_user}@{remote_host}:{remote_path}"

        # Upload markdown
        try:
            scp_md_cmd = ["scp", "-p", str(md_file), f"{remote_dest}{base_name}.md"]
            subprocess.run(scp_md_cmd, check=True, capture_output=True)
            click.echo(f"✓ Uploaded {md_file.name} to {remote_dest}")
        except subprocess.CalledProcessError as e:
            click.echo(f"⚠ Warning: scp failed for markdown: {e.stderr.decode()}")
        except FileNotFoundError:
            click.echo("⚠ Warning: scp not found.")

        # Upload HTML if it was generated
        if html_file.exists():
            try:
                scp_html_cmd = ["scp", "-p", str(html_file), f"{remote_dest}{base_name}.md.html"]
                subprocess.run(scp_html_cmd, check=True, capture_output=True)
                click.echo(f"✓ Uploaded {html_file.name} to {remote_dest}")
            except subprocess.CalledProcessError as e:
                click.echo(f"⚠ Warning: scp failed for HTML: {e.stderr.decode()}")


###############################################################################
# xAI Collections Commands
###############################################################################

async def upload_file_async(
    client: "xai_sdk.AsyncClient",
    collection_id: str,
    log_file: Path,
    semaphore: asyncio.Semaphore,
    wait_for_indexing: bool = False,
) -> Tuple[str, bool, Optional[str]]:
    """Upload a single file to xAI collection asynchronously.
    
    Returns a tuple of (filename, success, error_message).
    """
    async with semaphore:
        file_date = log_file.stem  # e.g., "2025-01-15"
        try:
            file_data = log_file.read_bytes()
            document = await client.collections.upload_document(
                collection_id=collection_id,
                name=log_file.name,
                data=file_data,
                fields={
                    "date": file_date,
                    "filename": log_file.name,
                },
                wait_for_indexing=wait_for_indexing,
            )
            return (log_file.name, True, document.file_metadata.file_id)
        except Exception as e:
            return (log_file.name, False, str(e))


async def xai_upload_async(
    collection_id: str,
    log_files: List[Path],
    uploaded_files: set,
    concurrency: int,
    wait_for_indexing: bool,
    resume: bool,
) -> Tuple[int, int, set]:
    """Upload multiple files to xAI collection concurrently.
    
    Returns (uploaded_count, failed_count, newly_uploaded_files).
    """
    client = xai_sdk.AsyncClient(
        api_key=os.environ.get("XAI_API_KEY"),
        management_api_key=os.environ.get("XAI_MANAGEMENT_API_KEY"),
        timeout=3600,
    )
    
    semaphore = asyncio.Semaphore(concurrency)
    
    # Filter files to upload
    files_to_upload = []
    for log_file in log_files:
        if resume and log_file.name in uploaded_files:
            click.echo(f"✓ {log_file.name} already uploaded, skipping")
            continue
        files_to_upload.append(log_file)
    
    if not files_to_upload:
        click.echo("No new files to upload.")
        return (0, 0, set())
    
    click.echo(f"Uploading {len(files_to_upload)} files with concurrency={concurrency}...")
    
    # Create upload tasks
    tasks = [
        upload_file_async(client, collection_id, log_file, semaphore, wait_for_indexing)
        for log_file in files_to_upload
    ]
    
    # Process with progress reporting
    uploaded_count = 0
    failed_count = 0
    newly_uploaded = set()
    
    # Use asyncio.as_completed for progress updates
    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
        filename, success, result = await coro
        if success:
            click.echo(f"[{i}/{len(tasks)}] ✓ {filename} -> {result}")
            newly_uploaded.add(filename)
            uploaded_count += 1
        else:
            click.echo(f"[{i}/{len(tasks)}] ✗ {filename}: {result}")
            failed_count += 1
    
    await client.close()
    return (uploaded_count, failed_count, newly_uploaded)


@cli.command("xai-upload")
@click.option("--collection-name", default="hplusroadmap-logs",
              help="Name for the xAI collection.")
@click.option("--chunk-size", default=500, type=int,
              help="Maximum tokens per chunk (server-side chunking).")
@click.option("--chunk-overlap", default=50, type=int,
              help="Token overlap between chunks.")
@click.option("--resume/--no-resume", default=True,
              help="Skip files that have already been uploaded.")
@click.option("--concurrency", type=int, default=100,
              help="Number of concurrent upload requests (default: 100).")
@click.option("--wait-for-indexing/--no-wait-for-indexing", default=False,
              help="Wait for each document to be indexed before continuing (slower but ensures indexing).")
@click.pass_obj
def xai_upload(obj: dict, collection_name: str, chunk_size: int, chunk_overlap: int, 
               resume: bool, concurrency: int, wait_for_indexing: bool) -> None:
    """Upload raw IRC logs to xAI Collections with parallel uploads.

    This command creates an xAI collection (if it doesn't exist) and uploads
    all raw log files from ``data_dir/raw`` using concurrent requests.  xAI 
    handles chunking server-side using the specified token configuration.  
    Each log file is tagged with a ``date`` metadata field for filtered retrieval.

    The ``--concurrency`` parameter controls how many uploads run in parallel.
    Higher values speed up the process significantly (e.g., 100 concurrent uploads
    can reduce total time from hours to minutes for large archives).

    By default, uploads do not wait for indexing to complete (--no-wait-for-indexing).
    This is much faster but means documents may not be immediately searchable.
    Use --wait-for-indexing if you need documents to be searchable right away.

    Requires XAI_API_KEY and XAI_MANAGEMENT_API_KEY environment variables.
    """
    if xai_sdk is None:
        raise click.UsageError("The xai-sdk package is not installed. Please install it via `uv pip install xai-sdk`.")

    data_dir: Path = obj["data_dir"]
    raw_dir = data_dir / "raw"
    xai_config_file = data_dir / "xai_collection.json"

    if not raw_dir.exists():
        raise click.UsageError(f"Raw logs directory does not exist: {raw_dir}")

    log_files = sorted(raw_dir.glob("*.log"))
    if not log_files:
        click.echo("No log files found to upload.")
        return

    # Check if we have an existing collection
    collection_id = None
    uploaded_files: set = set()
    if xai_config_file.exists():
        config = json.loads(xai_config_file.read_text(encoding="utf-8"))
        collection_id = config.get("collection_id")
        uploaded_files = set(config.get("uploaded_files", []))
        click.echo(f"Found existing collection: {collection_id}")
        click.echo(f"Already uploaded: {len(uploaded_files)} files")

    # Create collection if needed (use sync client for this one-time operation)
    if collection_id is None:
        click.echo(f"Creating collection '{collection_name}'...")
        sync_client = xai_sdk.Client(
            api_key=os.environ.get("XAI_API_KEY"),
            management_api_key=os.environ.get("XAI_MANAGEMENT_API_KEY"),
            timeout=3600,
        )
        collection = sync_client.collections.create(
            name=collection_name,
            chunk_configuration={
                "tokens_configuration": {
                    "max_chunk_size_tokens": chunk_size,
                    "chunk_overlap_tokens": chunk_overlap,
                    "encoding_name": "o200k_base",
                },
                "strip_whitespace": True,
            },
            field_definitions=[
                {"key": "date", "required": True, "unique": False, "inject_into_chunk": True},
                {"key": "filename", "required": True, "unique": False, "inject_into_chunk": False},
            ],
        )
        collection_id = collection.collection_id
        click.echo(f"Created collection: {collection_id}")

        # Save config
        config = {"collection_id": collection_id, "collection_name": collection_name, "uploaded_files": []}
        xai_config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Run async uploads
    click.echo(f"\nTotal files: {len(log_files)}, Already uploaded: {len(uploaded_files)}")
    uploaded_count, failed_count, newly_uploaded = asyncio.run(
        xai_upload_async(
            collection_id=collection_id,
            log_files=log_files,
            uploaded_files=uploaded_files,
            concurrency=concurrency,
            wait_for_indexing=wait_for_indexing,
            resume=resume,
        )
    )

    # Update config with newly uploaded files
    if newly_uploaded:
        uploaded_files.update(newly_uploaded)
        config = json.loads(xai_config_file.read_text(encoding="utf-8"))
        config["uploaded_files"] = list(uploaded_files)
        xai_config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")

    click.echo(f"\nDone! Uploaded: {uploaded_count}, Failed: {failed_count}, Total in collection: {len(uploaded_files)}")
    click.echo(f"Collection ID: {collection_id}")
    if not wait_for_indexing and uploaded_count > 0:
        click.echo("\nNote: Documents were uploaded without waiting for indexing.")
        click.echo("They may take a few minutes to become searchable.")


@cli.command("xai-query")
@click.option("--collection-id", default=None,
              help="xAI collection ID. If not provided, reads from data_dir/xai_collection.json.")
@click.option("--model", default="grok-4-fast",
              help="Grok model to use for answering queries.")
@click.option("--top-k", default=100, type=int,
              help="Number of search results to retrieve.")
@click.option("--search-mode", type=click.Choice(["hybrid", "semantic", "keyword"]), default="hybrid",
              help="Search mode: hybrid (default), semantic, or keyword.")
@click.option("--date-filter", default=None,
              help="Filter by date range, e.g., 'date >= \"2024-01-01\"' or 'date >= \"2024-01-01\" AND date <= \"2024-12-31\"'.")
@click.option("--nollm", is_flag=True, default=False,
              help="Skip LLM generation (print retrieved context only).")
@click.option("--prompt-fragment", default="",
              help="Additional instructions to add to the LLM prompt.")
@click.option("--output-name", default=None,
              help="Base filename for output (without extension).")
@click.option("--css-file", default="wrap.css",
              help="CSS file to use with pandoc for HTML generation.")
@click.option("--upload/--no-upload", default=True,
              help="Upload files to server via scp.")
@click.option("--remote-user", default="bryan",
              help="Remote SSH user for upload.")
@click.option("--remote-host", default="gnusha.org",
              help="Remote SSH host for upload.")
@click.option("--remote-path", default="~/public_html/irc/chatgpt/hplusroadmap/",
              help="Remote path for upload.")
@click.argument("query", required=False, default="")
@click.pass_obj
def xai_query(obj: dict, collection_id: Optional[str], model: str, top_k: int, search_mode: str,
              date_filter: Optional[str], nollm: bool, prompt_fragment: str, output_name: Optional[str],
              css_file: str, upload: bool, remote_user: str, remote_host: str, remote_path: str,
              query: str) -> None:
    """Query xAI Collections and generate an answer using Grok.

    This command searches the xAI collection for relevant IRC log excerpts,
    then uses a Grok model to generate a comprehensive answer.  Supports
    hybrid, semantic, or keyword search modes, and optional date filtering.

    If no query is provided but a prompt-fragment is given, the LLM will first
    generate appropriate search terms from the prompt-fragment.

    Requires XAI_API_KEY environment variable.
    """
    if xai_sdk is None:
        raise click.UsageError("The xai-sdk package is not installed. Please install it via `uv pip install xai-sdk`.")

    data_dir: Path = obj["data_dir"]
    xai_config_file = data_dir / "xai_collection.json"

    # Get collection ID
    if collection_id is None:
        if not xai_config_file.exists():
            raise click.UsageError("No collection ID provided and no xai_collection.json found. Run xai-upload first.")
        config = json.loads(xai_config_file.read_text(encoding="utf-8"))
        collection_id = config.get("collection_id")
        if not collection_id:
            raise click.UsageError("No collection_id found in xai_collection.json")

    # Initialize xAI client
    client = xai_sdk.Client(
        api_key=os.environ.get("XAI_API_KEY"),
        timeout=3600,
    )

    # If no query provided, generate search terms from prompt-fragment
    search_query = query
    if not query and prompt_fragment:
        click.echo("No query provided. Generating search terms from prompt-fragment using Grok...\n")
        query_generation_prompt = (
            "You are a search query generator for a semantic search system over IRC chat logs.\n"
            "Based on the user's request below, generate an effective search query.\n"
            "The query should be a concise list of key terms, concepts, or phrases that would help retrieve relevant messages.\n"
            "Return ONLY the search query text, nothing else.\n\n"
            f"User request: {prompt_fragment}\n\n"
            "Search query:"
        )
        query_gen_response = client.chat.create(
            model=model,
            messages=[{"role": "user", "content": query_generation_prompt}],
        )
        # Collect the full response
        full_response = ""
        for chunk in query_gen_response:
            if hasattr(chunk, 'choices') and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    full_response += delta.content
        search_query = full_response.strip()
        click.echo(f"Generated search query: {search_query}\n")
    elif not query and not prompt_fragment:
        raise click.UsageError("Either a query argument or --prompt-fragment must be provided.")

    # Search the collection
    click.echo(f"Searching collection {collection_id}...")
    click.echo(f"Search mode: {search_mode}, Top-K: {top_k}")
    if date_filter:
        click.echo(f"Date filter: {date_filter}")

    search_kwargs = {
        "query": search_query,
        "collection_ids": [collection_id],
        "limit": top_k,
        "retrieval_mode": search_mode, #{"type": search_mode},
    }
    if date_filter:
        search_kwargs["filter"] = date_filter

    results = client.collections.search(**search_kwargs)

    # Build context from search results
    context_parts = []
    for match in results.matches:
        # Extract metadata if available
        meta_str = f"[Score: {match.score:.4f}]"
        if hasattr(match, 'metadata') and match.metadata:
            if 'date' in match.metadata:
                meta_str = f"[Date: {match.metadata['date']}, Score: {match.score:.4f}]"
        context_parts.append(f"{meta_str}\n{match.chunk_content}")

    context = "\n\n".join(context_parts)
    click.echo(f"\nRetrieved {len(results.matches)} matches.")
    click.echo("\nContext: <context>" + context[:2000] + "...</context>\n\n")

    if nollm:
        click.echo("\nFull context:")
        click.echo(context)
        click.echo("\nExiting due to --nollm flag.")
        return

    # Check context length
    context_tokens = token_count(context)
    click.echo(f"Context length: {context_tokens} tokens\n")

    # Build prompt
    prompt = (
        "You are an assistant with access to the hplusroadmap IRC logs. Your job is to answer the following question(s) using the retrieved chat excerpts by constructing an in-depth technical report. The question (query) will be provided below.\n"
        "If you choose to include a quote from the IRC logs, then please use markdown format and GitHub markdown formatted four-space block quotes for the IRC log excerpts that you use. Please edit the excerpt to remove extraneous content.\n"
        "Where you see papers referenced, please collect those references and display them in your answer, even if the referenced papers may not be exactly related (e.g. they might be conceptually close, so include them in your output). Where you see companies mentioned, like a new company or a name of a company, or the name of people involved in different projects or ventures, or the name of different involved people, please list those in the answer as well. You are writing for a highly technical audience that is deeply interested in esoteric knowledge, technology, engineering, tech development, research, brainstorming, and speculation.\n"
        "If the logs do not contain the answer, say so. Your job is to extract the most relevant matching results and formulate it into a markdown-formatted document for readability. The question(s) or query that you have to answer is provided below (the user request and the search_query).\n\n"
    )

    if prompt_fragment:
        prompt += f"User request: <prompt>{prompt_fragment}</prompt>\n\n"

    prompt += (
        f"Search query used: <search_query>{search_query}</search_query>\n\n"
        f"Retrieved Context:\n<context>{context}</context>\n\n"
        "Answer:"
    )

    click.echo("Asking Grok...\n")

    # Use streaming chat completion
    chat_response = client.chat.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    answer_parts = []
    for chunk in chat_response:
        if hasattr(chunk, 'choices') and chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, 'content') and delta.content:
                answer_parts.append(delta.content)
                click.echo(delta.content, nl=False)

    answer = f"Search query: {search_query}\n\n" + "".join(answer_parts)
    click.echo("\n")

    # Save output to files
    output_dir = data_dir / "outputs"
    ensure_directory(output_dir)

    if output_name is None:
        timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"xai_query_{timestamp}"
    else:
        base_name = output_name

    md_file = output_dir / f"{base_name}.md"
    html_file = output_dir / f"{base_name}.md.html"

    md_file.write_text(answer.strip(), encoding="utf-8")
    click.echo(f"\n✓ Saved markdown to {md_file}")

    # Generate HTML with pandoc
    try:
        pandoc_cmd = [
            "pandoc",
            "-f", "markdown+autolink_bare_uris",
            "-s",
            "-c", css_file,
            str(md_file),
            "-o", str(html_file)
        ]
        subprocess.run(pandoc_cmd, check=True, capture_output=True)
        click.echo(f"✓ Generated HTML at {html_file}")
    except subprocess.CalledProcessError as e:
        click.echo(f"⚠ Warning: pandoc failed: {e.stderr.decode()}")
    except FileNotFoundError:
        click.echo("⚠ Warning: pandoc not found. Install pandoc to generate HTML.")

    # Upload files via scp
    if upload:
        remote_dest = f"{remote_user}@{remote_host}:{remote_path}"

        try:
            scp_md_cmd = ["scp", "-p", str(md_file), f"{remote_dest}{base_name}.md"]
            subprocess.run(scp_md_cmd, check=True, capture_output=True)
            click.echo(f"✓ Uploaded {md_file.name} to {remote_dest}")
        except subprocess.CalledProcessError as e:
            click.echo(f"⚠ Warning: scp failed for markdown: {e.stderr.decode()}")
        except FileNotFoundError:
            click.echo("⚠ Warning: scp not found.")

        if html_file.exists():
            try:
                scp_html_cmd = ["scp", "-p", str(html_file), f"{remote_dest}{base_name}.md.html"]
                subprocess.run(scp_html_cmd, check=True, capture_output=True)
                click.echo(f"✓ Uploaded {html_file.name} to {remote_dest}")
            except subprocess.CalledProcessError as e:
                click.echo(f"⚠ Warning: scp failed for HTML: {e.stderr.decode()}")


def main() -> None:
    """Invoke the CLI when run as a script."""
    cli()


if __name__ == "__main__":
    main()
