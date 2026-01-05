"""
Command line interface for the hplusroadmap RAG system.

This module provides CLI commands for downloading, preprocessing, embedding,
and querying IRC logs and other document collections. It delegates to
service modules for business logic.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Optional

import click

from hpluslogs.core.prompts import AAF_SYSTEM_PROMPT, FIGHTAGING_SYSTEM_PROMPT, GRG_SYSTEM_PROMPT, LESSWRONG_SYSTEM_PROMPT, RAG_SYSTEM_PROMPT
from hpluslogs.core.utils import ensure_directory, token_count
from hpluslogs.services import download, embedding, generation, preprocess, publishing, search, xai_upload


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


@cli.command("download")
@click.option("--start", "start_date", type=click.DateTime(formats=["%Y-%m-%d"]),
              help="First date to download (YYYY-MM-DD). Defaults to today if omitted.")
@click.option("--end", "end_date", type=click.DateTime(formats=["%Y-%m-%d"]),
              help="Last date to download (YYYY-MM-DD). Defaults to start date if omitted.")
@click.option("--resume/--no-resume", default=True,
              help="Skip files that already exist on disk (resume).")
@click.pass_obj
def download_cmd(obj: dict, start_date: Optional[_dt.datetime], end_date: Optional[_dt.datetime], resume: bool) -> None:
    """Download raw IRC logs from gnusha.org for a date range."""
    download.run(obj["data_dir"], start_date, end_date, resume)


@cli.command("preprocess")
@click.option("--max-tokens", default=175, help="Maximum tokens per chunk (sliding window).")
@click.option("--overlap", default=20, help="Token overlap between consecutive chunks.")
@click.option("--enrich/--no-enrich", default=False,
              help="Call an LLM to enrich or summarise each chunk.")
@click.pass_obj
def preprocess_cmd(obj: dict, max_tokens: int, overlap: int, enrich: bool) -> None:
    """Convert raw logs into JSONL files of overlapping message chunks."""
    preprocess.run(obj["data_dir"], max_tokens, overlap, enrich)


@cli.command("embed")
@click.option("--cost-limit", type=float, default=1.0,
              help="Maximum allowable estimated cost in USD for embedding.")
@click.option("--model", default="qwen/qwen3-embedding-8b", help="Embedding model identifier.")
@click.option("--concurrency", type=int, default=80,
              help="Number of concurrent embedding requests to make in parallel.")
@click.pass_obj
def embed_cmd(obj: dict, cost_limit: float, model: str, concurrency: int) -> None:
    """Embed all chunk files into a vector database using OpenRouter embeddings."""
    embedding.run(obj["data_dir"], cost_limit, model, concurrency)


@cli.command("query")
@click.option("--model", default="openrouter/x-ai/grok-4-fast", help="Chat model to use for answering queries.")
@click.option("--top-k", default=100, help="Number of nearest neighbours to retrieve.")
@click.option("--nollm", is_flag=True, default=False, help="Skip using an LLM (print context only).")
@click.option("--contextlimit", type=int, default=1500000, help="Maximum tokens in context before aborting (0 for no limit).")
@click.option("--prompt-fragment", default="", help="Additional instructions to add to the LLM prompt.")
@click.option("--output-name", default=None, help="Base filename for output (without extension).")
@click.option("--css-file", default="wrap.css", help="CSS file to use with pandoc for HTML generation.")
@click.option("--upload/--no-upload", default=True, help="Upload files to server via scp.")
@click.option("--remote-user", default="bryan", help="Remote SSH user for upload.")
@click.option("--remote-host", default="gnusha.org", help="Remote SSH host for upload.")
@click.option("--remote-path", default="~/public_html/irc/chatgpt/hplusroadmap/", help="Remote path for upload.")
@click.argument("query", required=False, default="")
@click.pass_obj
def query_cmd(obj: dict, model: str, top_k: int, nollm: bool, contextlimit: int, prompt_fragment: str,
              output_name: Optional[str], css_file: str, upload: bool, remote_user: str,
              remote_host: str, remote_path: str, query: str) -> None:
    """Answer a user question by retrieving relevant IRC messages and calling an LLM."""
    data_dir: Path = obj["data_dir"]
    
    # Determine search query
    search_query = query
    if not query and prompt_fragment:
        click.echo("No query provided. Generating search terms from prompt-fragment using LLM...\n")
        search_query = generation.generate_search_query(prompt_fragment, model)
        click.echo(f"Generated search query: {search_query}\n")
    elif not query and not prompt_fragment:
        raise click.UsageError("Either a query argument or --prompt-fragment must be provided.")

    # Retrieve context
    results = search.retrieve(data_dir, search_query, top_k, backend="chroma")
    context = generation.format_context(results)
    
    click.echo("\nContext: <context>" + context + "</context>\n\n")
    
    # Upload context file before LLM call
    publishing.output_context(
        data_dir, context, search_query,
        output_name=output_name, css_file=css_file, upload=upload,
        remote_user=remote_user, remote_host=remote_host, remote_path=remote_path,
        prefix="query"
    )
    
    # Upload context file before LLM call
    context_content = f"# Context for: {search_query}\n\n{context}"
    publishing.output(
        data_dir, context_content, 
        output_name=(output_name + ".context") if output_name else None,
        css_file=css_file, upload=upload,
        remote_user=remote_user, remote_host=remote_host, remote_path=remote_path,
        prefix="query.context"
    )
    
    if nollm:
        click.echo("\nExiting due to --nollm flag.")
        return
    
    if contextlimit > 0:
        context_tokens = token_count(context)
        if context_tokens > contextlimit:
            click.echo(f"Context too long: {context_tokens} tokens exceeds limit of {contextlimit}")
            return
        click.echo(f"Context length: {context_tokens}\n\n\n")

    # Generate answer
    click.echo("Asking the LLM...\n")
    answer = generation.generate_answer(search_query, results, model, prompt_fragment, RAG_SYSTEM_PROMPT)
    click.echo("\n" + answer.strip())

    # Publish output
    publishing.output(
        data_dir, answer, output_name, css_file, upload,
        remote_user, remote_host, remote_path, prefix="query"
    )


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
              help="Wait for each document to be indexed before continuing.")
@click.pass_obj
def xai_upload_cmd(obj: dict, collection_name: str, chunk_size: int, chunk_overlap: int,
                   resume: bool, concurrency: int, wait_for_indexing: bool) -> None:
    """Upload raw IRC logs to xAI Collections with parallel uploads."""
    xai_upload.run_hplusroadmap(
        obj["data_dir"], collection_name, chunk_size, chunk_overlap,
        resume, concurrency, wait_for_indexing
    )


@cli.command("xai-query")
@click.option("--collection-id", default=None,
              help="xAI collection ID. If not provided, reads from data_dir/xai_collection.json.")
@click.option("--model", default="openrouter/x-ai/grok-4-fast",
              help="LLM model to use for answering queries (via OpenRouter).")
@click.option("--top-k", default=100, type=int,
              help="Number of search results to retrieve.")
@click.option("--search-mode", type=click.Choice(["hybrid", "semantic", "keyword"]), default="semantic",
              help="Search mode: hybrid, semantic (default), or keyword.")
@click.option("--date-filter", default=None,
              help="Filter by date range, e.g., 'date >= \"2024-01-01\"'.")
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
def xai_query_cmd(obj: dict, collection_id: Optional[str], model: str, top_k: int, search_mode: str,
                  date_filter: Optional[str], nollm: bool, prompt_fragment: str, output_name: Optional[str],
                  css_file: str, upload: bool, remote_user: str, remote_host: str, remote_path: str,
                  query: str) -> None:
    """Query xAI Collections and generate an answer using Grok."""
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

    # Determine search query
    search_query = query
    if not query and prompt_fragment:
        click.echo("No query provided. Generating search terms from prompt-fragment using LLM...\n")
        search_query = generation.generate_search_query(prompt_fragment, model)
        click.echo(f"Generated search query: {search_query}\n")
    elif not query and not prompt_fragment:
        raise click.UsageError("Either a query argument or --prompt-fragment must be provided.")

    # Search
    click.echo(f"Searching collection {collection_id}...")
    click.echo(f"Search mode: {search_mode}, Top-K: {top_k}")
    if date_filter:
        click.echo(f"Date filter: {date_filter}")

    results = search.retrieve(
        data_dir, search_query, top_k, backend="xai",
        collection_id=collection_id, search_mode=search_mode, filter_str=date_filter
    )

    context = generation.format_context(results)
    click.echo(f"\nRetrieved {len(results)} matches.")
    click.echo("\nContext: <context>" + context[:2000] + "...</context>\n\n")

    # Upload context file before LLM call
    publishing.output_context(
        data_dir, context, search_query,
        output_name=output_name, css_file=css_file, upload=upload,
        remote_user=remote_user, remote_host=remote_host, remote_path=remote_path,
        prefix="xai_query"
    )

    if nollm:
        click.echo("\nFull context:")
        click.echo(context)
        click.echo("\nExiting due to --nollm flag.")
        return

    context_tokens = token_count(context)
    click.echo(f"Context length: {context_tokens} tokens\n")

    # Generate answer
    click.echo("Asking the LLM...\n")
    answer = generation.generate_answer(search_query, results, model, prompt_fragment, RAG_SYSTEM_PROMPT)
    click.echo("\n" + answer.strip())

    # Publish output
    publishing.output(
        data_dir, answer, output_name, css_file, upload,
        remote_user, remote_host, remote_path, prefix="xai_query"
    )


@cli.command("fightaging-collect")
@click.option("--collection-name", default="fightaging-articles",
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
              help="Wait for each document to be indexed before continuing.")
@click.pass_obj
def fightaging_collect_cmd(obj: dict, collection_name: str, chunk_size: int, chunk_overlap: int,
                           resume: bool, concurrency: int, wait_for_indexing: bool) -> None:
    """Upload Fight Aging! HTML files to xAI Collections."""
    xai_upload.run_fightaging(
        obj["data_dir"], collection_name, chunk_size, chunk_overlap,
        resume, concurrency, wait_for_indexing
    )


@cli.command("fightaging-query")
@click.option("--collection-id", default=None,
              help="xAI collection ID. If not provided, reads from data_dir/fightaging_collection.json.")
@click.option("--model", default="openrouter/x-ai/grok-4-fast",
              help="LLM model to use for answering queries (via OpenRouter).")
@click.option("--top-k", default=100, type=int,
              help="Number of search results to retrieve.")
@click.option("--search-mode", type=click.Choice(["hybrid", "semantic", "keyword"]), default="semantic",
              help="Search mode: hybrid, semantic (default), or keyword.")
@click.option("--source-filter", type=click.Choice(["all", "newsletters", "pages"]), default="all",
              help="Filter by source type: all (default), newsletters, or pages.")
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
@click.option("--remote-path", default="~/public_html/irc/chatgpt/fightaging/",
              help="Remote path for upload.")
@click.argument("query", required=False, default="")
@click.pass_obj
def fightaging_query_cmd(obj: dict, collection_id: Optional[str], model: str, top_k: int, search_mode: str,
                         source_filter: str, nollm: bool, prompt_fragment: str, output_name: Optional[str],
                         css_file: str, upload: bool, remote_user: str, remote_host: str, remote_path: str,
                         query: str) -> None:
    """Query Fight Aging! collection and generate an answer using Grok."""
    data_dir: Path = obj["data_dir"]
    config_file = data_dir / "fightaging_collection.json"

    # Get collection ID
    if collection_id is None:
        if not config_file.exists():
            raise click.UsageError("No collection ID provided and no fightaging_collection.json found. Run fightaging-collect first.")
        config = json.loads(config_file.read_text(encoding="utf-8"))
        collection_id = config.get("collection_id")
        if not collection_id:
            raise click.UsageError("No collection_id found in fightaging_collection.json")

    # Determine search query
    search_query = query
    if not query and prompt_fragment:
        click.echo("No query provided. Generating search terms from prompt-fragment using LLM...\n")
        search_query = generation.generate_search_query(prompt_fragment, model, for_fightaging=True)
        click.echo(f"Generated search query: {search_query}\n")
    elif not query and not prompt_fragment:
        raise click.UsageError("Either a query argument or --prompt-fragment must be provided.")

    # Build filter
    filter_str = None
    if source_filter != "all":
        filter_str = f'source_type="{source_filter}"'

    # Search
    click.echo(f"Searching collection {collection_id}...")
    click.echo(f"Search mode: {search_mode}, Top-K: {top_k}")
    if filter_str:
        click.echo(f"Source filter: {filter_str}")

    results = search.retrieve(
        data_dir, search_query, top_k, backend="xai",
        collection_id=collection_id, search_mode=search_mode, filter_str=filter_str
    )

    context = generation.format_context(results)
    click.echo(f"\nRetrieved {len(results)} matches.")
    click.echo("\nContext preview: <context>" + context[:2000] + "...</context>\n\n")

    # Upload context file before LLM call
    publishing.output_context(
        data_dir, context, search_query,
        output_name=output_name, css_file=css_file, upload=upload,
        remote_user=remote_user, remote_host=remote_host, remote_path=remote_path,
        prefix="fightaging_query"
    )

    if nollm:
        click.echo("\nFull context:")
        click.echo(context)
        click.echo("\nExiting due to --nollm flag.")
        return

    context_tokens = token_count(context)
    click.echo(f"Context length: {context_tokens} tokens\n")

    # Generate answer
    click.echo("Asking the LLM...\n")
    answer = generation.generate_answer(search_query, results, model, prompt_fragment, FIGHTAGING_SYSTEM_PROMPT)
    click.echo("\n" + answer.strip())

    # Publish output
    publishing.output(
        data_dir, answer, output_name, css_file, upload,
        remote_user, remote_host, remote_path, prefix="fightaging_query"
    )


@cli.command("lesswrong-upload")
@click.option("--collection-name", default="lesswrong-logs",
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
              help="Wait for each document to be indexed before continuing.")
@click.pass_obj
def lesswrong_upload_cmd(obj: dict, collection_name: str, chunk_size: int, chunk_overlap: int,
                         resume: bool, concurrency: int, wait_for_indexing: bool) -> None:
    """Upload LessWrong IRC logs to xAI Collections."""
    xai_upload.run_lesswrong(
        obj["data_dir"], collection_name, chunk_size, chunk_overlap,
        resume, concurrency, wait_for_indexing
    )


@cli.command("lesswrong-query")
@click.option("--collection-id", default=None,
              help="xAI collection ID. If not provided, reads from data_dir/lesswrong_collection.json.")
@click.option("--model", default="openrouter/x-ai/grok-4-fast",
              help="LLM model to use for answering queries (via OpenRouter).")
@click.option("--top-k", default=100, type=int,
              help="Number of search results to retrieve.")
@click.option("--search-mode", type=click.Choice(["hybrid", "semantic", "keyword"]), default="semantic",
              help="Search mode: hybrid, semantic (default), or keyword.")
@click.option("--date-filter", default=None,
              help="Filter by date range, e.g., 'date >= \"2024-01-01\"'.")
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
@click.option("--remote-path", default="~/public_html/irc/chatgpt/lesswrong/",
              help="Remote path for upload.")
@click.argument("query", required=False, default="")
@click.pass_obj
def lesswrong_query_cmd(obj: dict, collection_id: Optional[str], model: str, top_k: int, search_mode: str,
                        date_filter: Optional[str], nollm: bool, prompt_fragment: str, output_name: Optional[str],
                        css_file: str, upload: bool, remote_user: str, remote_host: str, remote_path: str,
                        query: str) -> None:
    """Query LessWrong IRC logs collection and generate an answer."""
    data_dir: Path = obj["data_dir"]
    config_file = data_dir / "lesswrong_collection.json"

    # Get collection ID
    if collection_id is None:
        if not config_file.exists():
            raise click.UsageError("No collection ID provided and no lesswrong_collection.json found. Run lesswrong-upload first.")
        config = json.loads(config_file.read_text(encoding="utf-8"))
        collection_id = config.get("collection_id")
        if not collection_id:
            raise click.UsageError("No collection_id found in lesswrong_collection.json")

    # Determine search query
    search_query = query
    if not query and prompt_fragment:
        click.echo("No query provided. Generating search terms from prompt-fragment using LLM...\n")
        search_query = generation.generate_search_query(prompt_fragment, model, for_lesswrong=True)
        click.echo(f"Generated search query: {search_query}\n")
    elif not query and not prompt_fragment:
        raise click.UsageError("Either a query argument or --prompt-fragment must be provided.")

    # Search
    click.echo(f"Searching collection {collection_id}...")
    click.echo(f"Search mode: {search_mode}, Top-K: {top_k}")
    if date_filter:
        click.echo(f"Date filter: {date_filter}")

    results = search.retrieve(
        data_dir, search_query, top_k, backend="xai",
        collection_id=collection_id, search_mode=search_mode, filter_str=date_filter
    )

    context = generation.format_context(results)
    click.echo(f"\nRetrieved {len(results)} matches.")
    click.echo("\nContext preview: <context>" + context[:2000] + "...</context>\n\n")

    # Upload context file before LLM call
    publishing.output_context(
        data_dir, context, search_query,
        output_name=output_name, css_file=css_file, upload=upload,
        remote_user=remote_user, remote_host=remote_host, remote_path=remote_path,
        prefix="lesswrong_query"
    )

    if nollm:
        click.echo("\nFull context:")
        click.echo(context)
        click.echo("\nExiting due to --nollm flag.")
        return

    context_tokens = token_count(context)
    click.echo(f"Context length: {context_tokens} tokens\n")

    # Generate answer
    click.echo("Asking the LLM...\n")
    answer = generation.generate_answer(search_query, results, model, prompt_fragment, LESSWRONG_SYSTEM_PROMPT)
    click.echo("\n" + answer.strip())

    # Publish output
    publishing.output(
        data_dir, answer, output_name, css_file, upload,
        remote_user, remote_host, remote_path, prefix="lesswrong_query"
    )


@cli.command("grg-collect")
@click.option("--collection-name", default="grg-mailing-list",
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
              help="Wait for each document to be indexed before continuing.")
@click.pass_obj
def grg_collect_cmd(obj: dict, collection_name: str, chunk_size: int, chunk_overlap: int,
                    resume: bool, concurrency: int, wait_for_indexing: bool) -> None:
    """Upload GRG (Gerontology Research Group) mailing list files to xAI Collections."""
    xai_upload.run_grg(
        obj["data_dir"], collection_name, chunk_size, chunk_overlap,
        resume, concurrency, wait_for_indexing
    )


@cli.command("grg-query")
@click.option("--collection-id", default=None,
              help="xAI collection ID. If not provided, reads from data_dir/grg_collection.json.")
@click.option("--model", default="openrouter/x-ai/grok-4-fast",
              help="LLM model to use for answering queries (via OpenRouter).")
@click.option("--top-k", default=100, type=int,
              help="Number of search results to retrieve.")
@click.option("--search-mode", type=click.Choice(["hybrid", "semantic", "keyword"]), default="semantic",
              help="Search mode: hybrid, semantic (default), or keyword.")
@click.option("--nollm", is_flag=True, default=False,
              help="Skip LLM generation (print retrieved context only).")
@click.option("--two-pass", is_flag=True, default=False,
              help="First generate LLM answer without context, then use that as part of context for final answer.")
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
@click.option("--remote-path", default="~/public_html/irc/chatgpt/grg/",
              help="Remote path for upload.")
@click.argument("query", required=False, default="")
@click.pass_obj
def grg_query_cmd(obj: dict, collection_id: Optional[str], model: str, top_k: int, search_mode: str,
                  nollm: bool, two_pass: bool, prompt_fragment: str, output_name: Optional[str],
                  css_file: str, upload: bool, remote_user: str, remote_host: str, remote_path: str,
                  query: str) -> None:
    """Query GRG (Gerontology Research Group) mailing list and generate an answer."""
    data_dir: Path = obj["data_dir"]
    config_file = data_dir / "grg_collection.json"

    # Get collection ID
    if collection_id is None:
        if not config_file.exists():
            raise click.UsageError("No collection ID provided and no grg_collection.json found. Run grg-collect first.")
        config = json.loads(config_file.read_text(encoding="utf-8"))
        collection_id = config.get("collection_id")
        if not collection_id:
            raise click.UsageError("No collection_id found in grg_collection.json")

    # Determine search query
    search_query = query
    if not query and prompt_fragment:
        click.echo("No query provided. Generating search terms from prompt-fragment using LLM...\n")
        search_query = generation.generate_search_query(prompt_fragment, model, for_grg=True)
        click.echo(f"Generated search query: {search_query}\n")
    elif not query and not prompt_fragment:
        raise click.UsageError("Either a query argument or --prompt-fragment must be provided.")

    # Search
    click.echo(f"Searching collection {collection_id}...")
    click.echo(f"Search mode: {search_mode}, Top-K: {top_k}")

    results = search.retrieve(
        data_dir, search_query, top_k, backend="xai",
        collection_id=collection_id, search_mode=search_mode
    )

    context = generation.format_context(results)
    click.echo(f"\nRetrieved {len(results)} matches.")
    #click.echo("\nContext preview: <context>" + context[:2000] + "...</context>\n\n")
    click.echo(f"\nContext: <context>{context}</context>\n\n")

    # Upload context file before LLM call
    publishing.output_context(
        data_dir, context, search_query,
        output_name=output_name, css_file=css_file, upload=upload,
        remote_user=remote_user, remote_host=remote_host, remote_path=remote_path,
        prefix="grg_query"
    )

    if nollm:
        click.echo("\nFull context:")
        click.echo(context)
        click.echo("\nExiting due to --nollm flag.")
        return

    context_tokens = token_count(context)
    click.echo(f"Context length: {context_tokens} tokens\n")

    # Two-pass mode: first generate without context, then use that as part of context
    if two_pass:
        click.echo("Two-pass mode: generating baseline answer without context...\n")
        baseline_answer = generation.generate_answer_no_context(
            search_query, model, prompt_fragment, GRG_SYSTEM_PROMPT
        )
        click.echo("Baseline answer:\n" + baseline_answer.strip() + "\n\n")
        
        # Prepend baseline answer to context
        combined_context = baseline_answer + "\n\n" + context
        
        click.echo("Generating final answer with combined context...\n")
        answer = generation.generate_answer(
            search_query, results, model, prompt_fragment, GRG_SYSTEM_PROMPT,
            context_override=combined_context
        )
    else:
        # Generate answer
        click.echo("Asking the LLM...\n")
        answer = generation.generate_answer(search_query, results, model, prompt_fragment, GRG_SYSTEM_PROMPT)
    
    click.echo("\n" + answer.strip())

    # Publish output
    publishing.output(
        data_dir, answer, output_name, css_file, upload,
        remote_user, remote_host, remote_path, prefix="grg_query"
    )


@cli.command("orionsarm-collect")
@click.option("--collection-name", default="orionsarm-encyclopedia",
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
              help="Wait for each document to be indexed before continuing.")
@click.pass_obj
def orionsarm_collect_cmd(obj: dict, collection_name: str, chunk_size: int, chunk_overlap: int,
                          resume: bool, concurrency: int, wait_for_indexing: bool) -> None:
    """Upload Orion's Arm encyclopedia files to xAI Collections."""
    xai_upload.run_orionsarm(
        obj["data_dir"], collection_name, chunk_size, chunk_overlap,
        resume, concurrency, wait_for_indexing
    )


@cli.command("orionsarm-query")
@click.option("--collection-id", default=None,
              help="xAI collection ID. If not provided, reads from data_dir/orionsarm_collection.json.")
@click.option("--model", default="openrouter/x-ai/grok-4-fast",
              help="LLM model to use for answering queries (via OpenRouter).")
@click.option("--top-k", default=100, type=int,
              help="Number of search results to retrieve.")
@click.option("--search-mode", type=click.Choice(["hybrid", "semantic", "keyword"]), default="semantic",
              help="Search mode: hybrid, semantic (default), or keyword.")
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
@click.option("--remote-path", default="~/public_html/irc/chatgpt/orionsarm/",
              help="Remote path for upload.")
@click.argument("query", required=False, default="")
@click.pass_obj
def orionsarm_query_cmd(obj: dict, collection_id: Optional[str], model: str, top_k: int, search_mode: str,
                        nollm: bool, prompt_fragment: str, output_name: Optional[str],
                        css_file: str, upload: bool, remote_user: str, remote_host: str, remote_path: str,
                        query: str) -> None:
    """Query Orion's Arm encyclopedia and generate an answer."""
    from hpluslogs.core.prompts import ORIONSARM_SYSTEM_PROMPT

    data_dir: Path = obj["data_dir"]
    config_file = data_dir / "orionsarm_collection.json"

    # Get collection ID
    if collection_id is None:
        if not config_file.exists():
            raise click.UsageError("No collection ID provided and no orionsarm_collection.json found. Run orionsarm-collect first.")
        config = json.loads(config_file.read_text(encoding="utf-8"))
        collection_id = config.get("collection_id")
        if not collection_id:
            raise click.UsageError("No collection_id found in orionsarm_collection.json")

    # Determine search query
    search_query = query
    if not query and prompt_fragment:
        click.echo("No query provided. Generating search terms from prompt-fragment using LLM...\n")
        search_query = generation.generate_search_query(prompt_fragment, model, for_orionsarm=True)
        click.echo(f"Generated search query: {search_query}\n")
    elif not query and not prompt_fragment:
        raise click.UsageError("Either a query argument or --prompt-fragment must be provided.")

    # Search
    click.echo(f"Searching collection {collection_id}...")
    click.echo(f"Search mode: {search_mode}, Top-K: {top_k}")

    results = search.retrieve(
        data_dir, search_query, top_k, backend="xai",
        collection_id=collection_id, search_mode=search_mode
    )

    context = generation.format_context(results)
    click.echo(f"\nRetrieved {len(results)} matches.")
    #click.echo("\nContext preview: <context>" + context[:2000] + "...</context>\n\n")
    click.echo(f"\nContext: <context>{context}</context>\n\n")

    # Upload context file before LLM call
    publishing.output_context(
        data_dir, context, search_query,
        output_name=output_name, css_file=css_file, upload=upload,
        remote_user=remote_user, remote_host=remote_host, remote_path=remote_path,
        prefix="orionsarm_query"
    )

    if nollm:
        click.echo("\nFull context:")
        click.echo(context)
        click.echo("\nExiting due to --nollm flag.")
        return

    context_tokens = token_count(context)
    click.echo(f"Context length: {context_tokens} tokens\n")

    # Generate answer
    click.echo("Asking the LLM...\n")
    answer = generation.generate_answer(search_query, results, model, prompt_fragment, ORIONSARM_SYSTEM_PROMPT)
    click.echo("\n" + answer.strip())

    # Publish output
    publishing.output(
        data_dir, answer, output_name, css_file, upload,
        remote_user, remote_host, remote_path, prefix="orionsarm_query"
    )


@cli.command("aaf-collect")
@click.option("--collection-name", default="anti-agingfirewalls",
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
              help="Wait for each document to be indexed before continuing.")
@click.pass_obj
def aaf_collect_cmd(obj: dict, collection_name: str, chunk_size: int, chunk_overlap: int,
                    resume: bool, concurrency: int, wait_for_indexing: bool) -> None:
    """Upload Anti-Aging Firewalls files to xAI Collections (recursively)."""
    xai_upload.run_aaf(
        obj["data_dir"], collection_name, chunk_size, chunk_overlap,
        resume, concurrency, wait_for_indexing
    )


@cli.command("aaf-query")
@click.option("--collection-id", default=None,
              help="xAI collection ID. If not provided, reads from data_dir/aaf_collection.json.")
@click.option("--model", default="openrouter/x-ai/grok-4-fast",
              help="LLM model to use for answering queries (via OpenRouter).")
@click.option("--top-k", default=100, type=int,
              help="Number of search results to retrieve.")
@click.option("--search-mode", type=click.Choice(["hybrid", "semantic", "keyword"]), default="semantic",
              help="Search mode: hybrid, semantic (default), or keyword.")
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
@click.option("--remote-path", default="~/public_html/irc/chatgpt/aaf/",
              help="Remote path for upload.")
@click.argument("query", required=False, default="")
@click.pass_obj
def aaf_query_cmd(obj: dict, collection_id: Optional[str], model: str, top_k: int, search_mode: str,
                  nollm: bool, prompt_fragment: str, output_name: Optional[str],
                  css_file: str, upload: bool, remote_user: str, remote_host: str, remote_path: str,
                  query: str) -> None:
    """Query Anti-Aging Firewalls articles and generate an answer."""
    data_dir: Path = obj["data_dir"]
    config_file = data_dir / "aaf_collection.json"

    # Get collection ID
    if collection_id is None:
        if not config_file.exists():
            raise click.UsageError("No collection ID provided and no aaf_collection.json found. Run aaf-collect first.")
        config = json.loads(config_file.read_text(encoding="utf-8"))
        collection_id = config.get("collection_id")
        if not collection_id:
            raise click.UsageError("No collection_id found in aaf_collection.json")

    # Determine search query
    search_query = query
    if not query and prompt_fragment:
        click.echo("No query provided. Generating search terms from prompt-fragment using LLM...\n")
        search_query = generation.generate_search_query(prompt_fragment, model, for_aaf=True)
        click.echo(f"Generated search query: {search_query}\n")
    elif not query and not prompt_fragment:
        raise click.UsageError("Either a query argument or --prompt-fragment must be provided.")

    # Search
    click.echo(f"Searching collection {collection_id}...")
    click.echo(f"Search mode: {search_mode}, Top-K: {top_k}")

    results = search.retrieve(
        data_dir, search_query, top_k, backend="xai",
        collection_id=collection_id, search_mode=search_mode
    )

    context = generation.format_context(results)
    click.echo(f"\nRetrieved {len(results)} matches.")
    click.echo(f"\nContext: <context>{context}</context>\n\n")

    # Upload context file before LLM call
    publishing.output_context(
        data_dir, context, search_query,
        output_name=output_name, css_file=css_file, upload=upload,
        remote_user=remote_user, remote_host=remote_host, remote_path=remote_path,
        prefix="aaf_query"
    )

    if nollm:
        click.echo("\nFull context:")
        click.echo(context)
        click.echo("\nExiting due to --nollm flag.")
        return

    context_tokens = token_count(context)
    click.echo(f"Context length: {context_tokens} tokens\n")

    # Generate answer
    click.echo("Asking the LLM...\n")
    answer = generation.generate_answer(search_query, results, model, prompt_fragment, AAF_SYSTEM_PROMPT)
    click.echo("\n" + answer.strip())

    # Publish output
    publishing.output(
        data_dir, answer, output_name, css_file, upload,
        remote_user, remote_host, remote_path, prefix="aaf_query"
    )


def main() -> None:
    """Invoke the CLI when run as a script."""
    cli()


if __name__ == "__main__":
    main()
