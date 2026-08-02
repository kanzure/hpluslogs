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
from hpluslogs.services import download, embedding, generation, preprocess, publishing, search, summarize, summary_index, xai_upload


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
              help="First date to download (YYYY-MM-DD). If omitted with --end, reconcile missing logs through today.")
@click.option("--end", "end_date", type=click.DateTime(formats=["%Y-%m-%d"]),
              help="Last date to download (YYYY-MM-DD). Defaults to start date if omitted.")
@click.option("--resume/--no-resume", default=True,
              help="Skip files that already exist on disk (resume).")
@click.pass_obj
def download_cmd(obj: dict, start_date: Optional[_dt.datetime], end_date: Optional[_dt.datetime], resume: bool) -> None:
    """Download raw IRC logs from gnusha.org."""
    download.run(obj["data_dir"], start_date, end_date, resume)


@cli.command("preprocess")
@click.option("--max-tokens", default=175, help="Maximum tokens per chunk (sliding window).")
@click.option("--overlap", default=20, help="Token overlap between consecutive chunks.")
@click.option("--enrich/--no-enrich", default=False,
              help="Call an LLM to enrich or summarise each chunk.")
@click.option("--resume/--no-resume", default=True,
              help="Skip files that already have corresponding .jsonl chunks (resume).")
@click.pass_obj
def preprocess_cmd(obj: dict, max_tokens: int, overlap: int, enrich: bool, resume: bool) -> None:
    """Convert raw logs into JSONL files of overlapping message chunks."""
    preprocess.run(obj["data_dir"], max_tokens, overlap, enrich, resume)


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
@click.option("--cleanpass/--no-cleanpass", default=False,
              help="Run a cleaning pass on context to remove redundancy and formatting artifacts.")
@click.argument("query", required=False, default="")
@click.pass_obj
def query_cmd(obj: dict, model: str, top_k: int, nollm: bool, contextlimit: int, prompt_fragment: str,
              output_name: Optional[str], css_file: str, upload: bool, remote_user: str,
              remote_host: str, remote_path: str, cleanpass: bool, query: str) -> None:
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
    
    # Clean pass to remove redundancy and formatting artifacts
    if cleanpass:
        click.echo("Running cleaning pass on context...")
        context = generation.clean_context(context)
        click.echo("Cleaning pass complete.\n")
    
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
@click.option("--cleanpass/--no-cleanpass", default=False,
              help="Run a cleaning pass on context to remove redundancy and formatting artifacts.")
@click.argument("query", required=False, default="")
@click.pass_obj
def xai_query_cmd(obj: dict, collection_id: Optional[str], model: str, top_k: int, search_mode: str,
                  date_filter: Optional[str], nollm: bool, prompt_fragment: str, output_name: Optional[str],
                  css_file: str, upload: bool, remote_user: str, remote_host: str, remote_path: str,
                  cleanpass: bool, query: str) -> None:
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
    
    # Clean pass to remove redundancy and formatting artifacts
    if cleanpass:
        click.echo("Running cleaning pass on context...")
        context = generation.clean_context(context)
        click.echo("Cleaning pass complete.\n")
    
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
    answer = generation.generate_answer(search_query, results, model, prompt_fragment, RAG_SYSTEM_PROMPT, context_override=context)
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
@click.option("--cleanpass/--no-cleanpass", default=False,
              help="Run a cleaning pass on context to remove redundancy and formatting artifacts.")
@click.argument("query", required=False, default="")
@click.pass_obj
def fightaging_query_cmd(obj: dict, collection_id: Optional[str], model: str, top_k: int, search_mode: str,
                         source_filter: str, nollm: bool, prompt_fragment: str, output_name: Optional[str],
                         css_file: str, upload: bool, remote_user: str, remote_host: str, remote_path: str,
                         cleanpass: bool, query: str) -> None:
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
    
    # Clean pass to remove redundancy and formatting artifacts
    if cleanpass:
        click.echo("Running cleaning pass on context...")
        context = generation.clean_context(context)
        click.echo("Cleaning pass complete.\n")
    
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
    answer = generation.generate_answer(search_query, results, model, prompt_fragment, FIGHTAGING_SYSTEM_PROMPT, context_override=context)
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
@click.option("--cleanpass/--no-cleanpass", default=False,
              help="Run a cleaning pass on context to remove redundancy and formatting artifacts.")
@click.argument("query", required=False, default="")
@click.pass_obj
def lesswrong_query_cmd(obj: dict, collection_id: Optional[str], model: str, top_k: int, search_mode: str,
                        date_filter: Optional[str], nollm: bool, prompt_fragment: str, output_name: Optional[str],
                        css_file: str, upload: bool, remote_user: str, remote_host: str, remote_path: str,
                        cleanpass: bool, query: str) -> None:
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
    
    # Clean pass to remove redundancy and formatting artifacts
    if cleanpass:
        click.echo("Running cleaning pass on context...")
        context = generation.clean_context(context)
        click.echo("Cleaning pass complete.\n")
    
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
    answer = generation.generate_answer(search_query, results, model, prompt_fragment, LESSWRONG_SYSTEM_PROMPT, context_override=context)
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
@click.option("--cleanpass/--no-cleanpass", default=False,
              help="Run a cleaning pass on context to remove redundancy and formatting artifacts.")
@click.argument("query", required=False, default="")
@click.pass_obj
def grg_query_cmd(obj: dict, collection_id: Optional[str], model: str, top_k: int, search_mode: str,
                  nollm: bool, two_pass: bool, prompt_fragment: str, output_name: Optional[str],
                  css_file: str, upload: bool, remote_user: str, remote_host: str, remote_path: str,
                  cleanpass: bool, query: str) -> None:
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
    
    # Clean pass to remove redundancy and formatting artifacts
    if cleanpass:
        click.echo("Running cleaning pass on context...")
        context = generation.clean_context(context)
        click.echo("Cleaning pass complete.\n")
    
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
@click.option("--cleanpass/--no-cleanpass", default=False,
              help="Run a cleaning pass on context to remove redundancy and formatting artifacts.")
@click.argument("query", required=False, default="")
@click.pass_obj
def orionsarm_query_cmd(obj: dict, collection_id: Optional[str], model: str, top_k: int, search_mode: str,
                        nollm: bool, prompt_fragment: str, output_name: Optional[str],
                        css_file: str, upload: bool, remote_user: str, remote_host: str, remote_path: str,
                        cleanpass: bool, query: str) -> None:
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
    
    # Clean pass to remove redundancy and formatting artifacts
    if cleanpass:
        click.echo("Running cleaning pass on context...")
        context = generation.clean_context(context)
        click.echo("Cleaning pass complete.\n")
    
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
    answer = generation.generate_answer(search_query, results, model, prompt_fragment, ORIONSARM_SYSTEM_PROMPT, context_override=context)
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
@click.option("--cleanpass/--no-cleanpass", default=False,
              help="Run a cleaning pass on context to remove redundancy and formatting artifacts.")
@click.argument("query", required=False, default="")
@click.pass_obj
def aaf_query_cmd(obj: dict, collection_id: Optional[str], model: str, top_k: int, search_mode: str,
                  nollm: bool, prompt_fragment: str, output_name: Optional[str],
                  css_file: str, upload: bool, remote_user: str, remote_host: str, remote_path: str,
                  cleanpass: bool, query: str) -> None:
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
    
    # Clean pass to remove redundancy and formatting artifacts
    if cleanpass:
        click.echo("Running cleaning pass on context...")
        context = generation.clean_context(context)
        click.echo("Cleaning pass complete.\n")
    
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
    answer = generation.generate_answer(search_query, results, model, prompt_fragment, AAF_SYSTEM_PROMPT, context_override=context)
    click.echo("\n" + answer.strip())

    # Publish output
    publishing.output(
        data_dir, answer, output_name, css_file, upload,
        remote_user, remote_host, remote_path, prefix="aaf_query"
    )


###############################################################################
# Chat log summarization commands (daily / weekly / monthly)
###############################################################################

# Options shared by all three summarization commands.
def _summary_options(func):
    """Decorator bundling the common options for summarization commands."""
    decorators = [
        click.option("--model", default=summarize.DEFAULT_MODEL,
                     help="LLM to use for summarization (via OpenRouter/litellm). Default: DeepSeek."),
        click.option("--fetch-links/--no-fetch-links", default=True,
                     help="Follow links posted in the logs to gather manuscript/article details."),
        click.option("--max-links-daily", "max_links_daily", type=int, default=500,
                     help="Maximum number of unique links to follow per day (daily limit)."),
        click.option("--concurrency", type=int, default=summarize.DEFAULT_CONCURRENCY,
                     help="Number of daily summaries to generate in parallel."),
        click.option("--resume/--no-resume", default=True,
                     help="Resume: skip summaries already written to disk (non-empty); "
                          "only (re)generate missing or empty/corrupt ones. "
                          "--no-resume regenerates everything from scratch."),
        click.option("--max-link-chars", type=int, default=6000,
                     help="Maximum characters of extracted text to keep per followed link."),
        click.option("--css-file", default="wrap.css",
                     help="CSS file to use with pandoc for HTML generation."),
        click.option("--upload/--no-upload", default=True, help="Upload files to server via scp."),
        click.option("--remote-user", default="bryan", help="Remote SSH user for upload."),
        click.option("--remote-host", default="gnusha.org", help="Remote SSH host for upload."),
        click.option("--remote-path", default=summarize.DEFAULT_REMOTE_PATH,
                     help="Remote path for upload (default: chatsummaries/ directory)."),
    ]
    for dec in reversed(decorators):
        func = dec(func)
    return func


@cli.command("summarize-day")
@click.option("--date", "date", type=click.DateTime(formats=["%Y-%m-%d"]),
              help="Single day to summarize (YYYY-MM-DD). Overrides --start/--end.")
@click.option("--start", "start_date", type=click.DateTime(formats=["%Y-%m-%d"]),
              help="First day to summarize (YYYY-MM-DD). Defaults to today.")
@click.option("--end", "end_date", type=click.DateTime(formats=["%Y-%m-%d"]),
              help="Last day to summarize (YYYY-MM-DD). Defaults to start.")
@click.option("--force/--no-force", default=False,
              help="Regenerate even if a daily summary already exists.")
@_summary_options
@click.pass_obj
def summarize_day_cmd(obj: dict, date: Optional[_dt.datetime], start_date: Optional[_dt.datetime],
                      end_date: Optional[_dt.datetime], force: bool, model: str, fetch_links: bool,
                      max_links_daily: int, concurrency: int, resume: bool, max_link_chars: int,
                      css_file: str, upload: bool, remote_user: str, remote_host: str,
                      remote_path: str) -> None:
    """Summarize one or more days of IRC logs, calling out posted manuscripts."""
    if date is not None:
        start = end = date.date()
    else:
        start = start_date.date() if start_date else _dt.date.today()
        end = end_date.date() if end_date else start
    force = force or not resume
    summarize.summarize_days(
        obj["data_dir"], start, end, concurrency=concurrency,
        model=model, fetch_links=fetch_links, max_links=max_links_daily,
        max_link_chars=max_link_chars, force=force, upload=upload, css_file=css_file,
        remote_user=remote_user, remote_host=remote_host, remote_path=remote_path,
    )


@cli.command("summarize-days")
@click.option("--start", "start_date", type=click.DateTime(formats=["%Y-%m-%d"]),
              help="First raw-log day to consider (YYYY-MM-DD). Defaults to all local raw logs.")
@click.option("--end", "end_date", type=click.DateTime(formats=["%Y-%m-%d"]),
              help="Last raw-log day to consider (YYYY-MM-DD). Defaults to all local raw logs.")
@click.option("--force/--no-force", default=False,
              help="Regenerate all matching raw-log days, even if daily summaries exist.")
@_summary_options
@click.pass_obj
def summarize_days_cmd(obj: dict, start_date: Optional[_dt.datetime],
                       end_date: Optional[_dt.datetime], force: bool, model: str,
                       fetch_links: bool, max_links_daily: int, concurrency: int,
                       resume: bool, max_link_chars: int, css_file: str, upload: bool,
                       remote_user: str, remote_host: str, remote_path: str) -> None:
    """Summarize every local raw-log day that lacks a usable daily summary."""
    start = start_date.date() if start_date else None
    end = end_date.date() if end_date else None
    force = force or not resume
    summarize.summarize_missing_days(
        obj["data_dir"], start=start, end=end, concurrency=concurrency,
        model=model, fetch_links=fetch_links, max_links=max_links_daily,
        max_link_chars=max_link_chars, force=force, upload=upload, css_file=css_file,
        remote_user=remote_user, remote_host=remote_host, remote_path=remote_path,
    )


@cli.command("summarize-week")
@click.option("--date", "date", type=click.DateTime(formats=["%Y-%m-%d"]),
              help="Any day within the target ISO week (Mon-Sun). Defaults to today.")
@click.option("--auto/--no-auto", default=True,
              help="Auto-generate any missing daily summaries for the week.")
@click.option("--force/--no-force", default=False,
              help="Regenerate the weekly summary even if it already exists.")
@click.option("--force-daily/--no-force-daily", default=False,
              help="Regenerate underlying daily summaries even if they exist.")
@_summary_options
@click.pass_obj
def summarize_week_cmd(obj: dict, date: Optional[_dt.datetime], auto: bool, force: bool,
                       force_daily: bool, model: str, fetch_links: bool, max_links_daily: int,
                       concurrency: int, resume: bool, max_link_chars: int, css_file: str,
                       upload: bool, remote_user: str, remote_host: str, remote_path: str) -> None:
    """Consolidate a week's daily summaries into a weekly digest."""
    any_day = date.date() if date else _dt.date.today()
    if not resume:
        force = force_daily = True
    summarize.summarize_week(
        obj["data_dir"], any_day, concurrency=concurrency,
        model=model, auto=auto, fetch_links=fetch_links, max_links=max_links_daily,
        max_link_chars=max_link_chars, force=force, force_daily=force_daily,
        upload=upload, css_file=css_file, remote_user=remote_user,
        remote_host=remote_host, remote_path=remote_path,
    )


@cli.command("summarize-weeks")
@click.option("--start", "start_date", type=click.DateTime(formats=["%Y-%m-%d"]),
              help="First daily-summary day to consider (YYYY-MM-DD). Defaults to all cached dailies.")
@click.option("--end", "end_date", type=click.DateTime(formats=["%Y-%m-%d"]),
              help="Last daily-summary day to consider (YYYY-MM-DD). Defaults to all cached dailies.")
@click.option("--force/--no-force", default=False,
              help="Regenerate all matching weekly summaries, even if they exist.")
@_summary_options
@click.pass_obj
def summarize_weeks_cmd(obj: dict, start_date: Optional[_dt.datetime],
                        end_date: Optional[_dt.datetime], force: bool, model: str,
                        fetch_links: bool, max_links_daily: int, concurrency: int,
                        resume: bool, max_link_chars: int, css_file: str, upload: bool,
                        remote_user: str, remote_host: str, remote_path: str) -> None:
    """Consolidate cached daily summaries into weekly digests."""
    start = start_date.date() if start_date else None
    end = end_date.date() if end_date else None
    force = force or not resume
    summarize.summarize_weeks(
        obj["data_dir"], start=start, end=end, concurrency=concurrency,
        model=model, force=force, upload=upload, css_file=css_file,
        remote_user=remote_user, remote_host=remote_host, remote_path=remote_path,
    )


@cli.command("summarize-month")
@click.option("--month", "month", default=None,
              help="Target month as YYYY-MM. Defaults to the current month.")
@click.option("--auto/--no-auto", default=True,
              help="Auto-generate any missing weekly (and daily) summaries for the month.")
@click.option("--force/--no-force", default=False,
              help="Regenerate the monthly summary even if it already exists.")
@click.option("--force-weekly/--no-force-weekly", default=False,
              help="Regenerate underlying weekly summaries even if they exist.")
@click.option("--force-daily/--no-force-daily", default=False,
              help="Regenerate underlying daily summaries even if they exist.")
@_summary_options
@click.pass_obj
def summarize_month_cmd(obj: dict, month: Optional[str], auto: bool, force: bool,
                        force_weekly: bool, force_daily: bool, model: str, fetch_links: bool,
                        max_links_daily: int, concurrency: int, resume: bool, max_link_chars: int,
                        css_file: str, upload: bool, remote_user: str, remote_host: str,
                        remote_path: str) -> None:
    """Consolidate a month's weekly summaries into a monthly digest."""
    if month:
        try:
            year_i, month_i = (int(x) for x in month.split("-", 1))
        except ValueError:
            raise click.UsageError("--month must be formatted as YYYY-MM, e.g. 2026-04.")
    else:
        today = _dt.date.today()
        year_i, month_i = today.year, today.month
    if not resume:
        force = force_weekly = force_daily = True
    summarize.summarize_month(
        obj["data_dir"], year_i, month_i, concurrency=concurrency,
        model=model, auto=auto, fetch_links=fetch_links, max_links=max_links_daily,
        max_link_chars=max_link_chars, force=force, force_weekly=force_weekly,
        force_daily=force_daily, upload=upload, css_file=css_file,
        remote_user=remote_user, remote_host=remote_host, remote_path=remote_path,
    )


@cli.command("summarize-months")
@click.option("--start", "start_month", default=None,
              help="First month to consider as YYYY-MM. Defaults to all cached weeklies.")
@click.option("--end", "end_month", default=None,
              help="Last month to consider as YYYY-MM. Defaults to all cached weeklies.")
@click.option("--force/--no-force", default=False,
              help="Regenerate all matching monthly summaries, even if they exist.")
@_summary_options
@click.pass_obj
def summarize_months_cmd(obj: dict, start_month: Optional[str], end_month: Optional[str],
                         force: bool, model: str, fetch_links: bool, max_links_daily: int,
                         concurrency: int, resume: bool, max_link_chars: int, css_file: str,
                         upload: bool, remote_user: str, remote_host: str,
                         remote_path: str) -> None:
    """Consolidate cached weekly summaries into monthly digests."""
    def parse_month(value: Optional[str], option: str) -> Optional[_dt.date]:
        if not value:
            return None
        try:
            year_i, month_i = (int(x) for x in value.split("-", 1))
            return _dt.date(year_i, month_i, 1)
        except ValueError:
            raise click.UsageError(f"{option} must be formatted as YYYY-MM, e.g. 2026-04.")

    start = parse_month(start_month, "--start")
    end = parse_month(end_month, "--end")
    force = force or not resume
    summarize.summarize_months(
        obj["data_dir"], start=start, end=end, concurrency=concurrency,
        model=model, force=force, upload=upload, css_file=css_file,
        remote_user=remote_user, remote_host=remote_host, remote_path=remote_path,
    )


@cli.command("generate-summary-index")
@click.option("--upload/--no-upload", default=True,
              help="Upload index.html to the remote summaries directory via scp.")
@click.option("--remote-user", default="bryan",
              help="Remote SSH user for upload.")
@click.option("--remote-host", default="gnusha.org",
              help="Remote SSH host for upload.")
@click.option("--remote-path", default=summarize.DEFAULT_REMOTE_PATH,
              help="Remote path for upload (default: chatsummaries/ directory).")
@click.pass_obj
def generate_summary_index_cmd(obj: dict, upload: bool, remote_user: str,
                               remote_host: str, remote_path: str) -> None:
    """Generate a static HTML index of cached daily / weekly / monthly summaries."""
    path = summary_index.generate(
        obj["data_dir"], upload=upload, remote_user=remote_user,
        remote_host=remote_host, remote_path=remote_path,
    )
    click.echo(f"Wrote summary index: {path}")
    if upload:
        click.echo(f"Uploaded index.html to: {remote_user}@{remote_host}:{remote_path}")


def main() -> None:
    """Invoke the CLI when run as a script."""
    cli()


if __name__ == "__main__":
    main()
