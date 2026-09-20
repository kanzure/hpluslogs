"""Paper commands registered on the existing flat Click interface."""
from __future__ import annotations

import datetime
import json
from pathlib import Path
import subprocess

import click

from hpluslogs.services import papers, papers_markdown


def register(cli):
    from hpluslogs.papers_remote_cli import register as register_remote
    register_remote(cli)
    from hpluslogs.papers_chroma_cli import register as register_chroma
    register_chroma(cli)
    from hpluslogs.papers_query_cli import register as register_queries
    register_queries(cli)

    @cli.command('papers-restore')
    @click.option('--repository', default=papers.RESTIC_REPO, required=not bool(papers.RESTIC_REPO),
                  help='Restic repository URL; defaults to PAPERS_RESTIC_REPOSITORY if set.')
    @click.option('--snapshot', required=True, help='Exact snapshot ID from restic snapshots (or latest).')
    @click.option('--snapshot-path', default=papers.SNAPSHOT_PATH, show_default=True)
    @click.option('--password-file', type=click.Path(exists=True, dir_okay=False, path_type=Path))
    @click.option('--dry-run', is_flag=True)
    @click.pass_obj
    def restore_cmd(obj, **kwargs):
        """Restore only papers2 from restic; verify restored contents."""
        try:
            papers.restore(obj['data_dir'], **kwargs)
        except (subprocess.CalledProcessError, FileNotFoundError) as error:
            raise click.ClickException(str(error)) from error

    @cli.command('papers-sync')
    @click.option('--source', default='bryan@gnusha.org:public_html/papers2/', show_default=True)
    @click.option('--dry-run', is_flag=True)
    @click.pass_obj
    def sync_cmd(obj, source, dry_run):
        """Fetch new/changed papers via rsync (preserves local extras)."""
        try:
            papers.sync(obj['data_dir'], source, dry_run)
        except (subprocess.CalledProcessError, FileNotFoundError) as error:
            raise click.ClickException(str(error)) from error

    @cli.command('papers-add')
    @click.argument('source')
    @click.option('--path', 'relative_path', help='Destination relative to papers2, e.g. biology/paper.pdf.')
    @click.option('--replace', is_flag=True, help='Replace an existing local PDF.')
    @click.pass_obj
    def add_cmd(obj, source, relative_path, replace):
        """Add a PDF from a local path or HTTP(S) URL; then run papers-markdown."""
        path = papers.add(obj['data_dir'], source, relative_path, replace)
        click.echo(f'Added {path}. Run papers-markdown, then papers-cost. Upload remains a separate approval step.')

    @cli.command('papers-markdown')
    @click.option('--workers', type=click.IntRange(min=1), default=2, show_default=True)
    @click.option('--timeout', type=click.IntRange(min=1), default=600, help='Conversion deadline per document in seconds.')
    @click.option('--limit', type=click.IntRange(min=1), default=None, help='Convert only the first N local PDFs for a sample.')
    @click.option('--skip-failed', is_flag=True, help='Leave unchanged previously failed PDFs deferred.')
    @click.pass_obj
    def markdown_cmd(obj, workers, timeout, limit, skip_failed):
        """Convert PDFs locally with PyMuPDF4LLM; never call xAI."""
        papers_markdown.convert(obj['data_dir'], workers, timeout, limit, skip_failed=skip_failed)

    @cli.command('papers-repair')
    @click.option('--workers', type=click.IntRange(min=1), default=8)
    @click.option('--timeout', type=click.IntRange(min=1), default=1800)
    @click.option('--limit', type=click.IntRange(min=1))
    @click.option('--apply', is_flag=True, help='Repair selected outputs; otherwise report candidates only.')
    @click.pass_obj
    def repair_cmd(obj, **kwargs):
        """Recover garbled Markdown and readable failed PDFs using local OCR."""
        from hpluslogs.services.papers_repair import repair
        repair(obj['data_dir'], **kwargs)

    @cli.command('papers-prepare')
    @click.option('--workers', type=click.IntRange(min=1), default=2, show_default=True)
    @click.option('--timeout', type=click.IntRange(min=1), default=600)
    @click.option('--limit', type=click.IntRange(min=1), default=None)
    @click.pass_obj
    def prepare_cmd(obj, workers, timeout, limit):
        """Convert locally, write measured costs, and stop without uploading."""
        papers_markdown.prepare(obj['data_dir'], workers, timeout, limit)

    @cli.command('papers-progress')
    @click.option('--days', type=click.IntRange(min=1), default=30)
    @click.option('--searches', type=click.IntRange(min=0), default=0)
    @click.option('--index-multiplier', type=click.FloatRange(min=0), default=1.0)
    @click.pass_obj
    def progress_cmd(obj, **kwargs):
        """Show live conversion progress, ETA and projected monthly costs; no API calls."""
        from hpluslogs.services import papers_progress
        click.echo(json.dumps(papers_progress.report(obj['data_dir'], **kwargs), indent=2))

    @cli.command('papers-cost')
    @click.option('--index-gib', type=click.FloatRange(min=0), help='Estimated billable index GiB; defaults to measured Markdown size.')
    @click.option('--searches', type=click.IntRange(min=0), default=0, help='Monthly collection search tool calls.')
    @click.option('--days', type=click.IntRange(min=1), default=30)
    @click.pass_obj
    def cost_cmd(obj, index_gib, searches, days):
        """Estimate monthly storage/search costs without calling xAI."""
        report = papers_markdown.cost_report(obj['data_dir'], index_gib, searches, days)
        click.echo(json.dumps(report, indent=2))

    @cli.command('papers-status')
    @click.option('--remote', is_flag=True, help='Also read collection metadata from xAI.')
    @click.pass_obj
    def status_cmd(obj, remote):
        """Show local PDF inventory and persisted upload/indexing results."""
        click.echo(json.dumps(papers.status(obj['data_dir'], remote), indent=2))

    @cli.command('papers-reconcile')
    @click.pass_obj
    def reconcile_cmd(obj):
        """Recover remote file IDs after an interrupted upload (read-only API)."""
        result = papers.reconcile(obj['data_dir'])
        click.echo(json.dumps(result, indent=2))
        if result['unresolved']:
            raise click.ClickException('Some upload outcomes remain uncertain; inspect their remote names in the xAI console before retrying.')

    @cli.command('papers-upload')
    @click.option('--collection-name', default='papers2-markdown', show_default=True)
    @click.option('--concurrency', type=click.IntRange(1, 32), default=4, show_default=True)
    @click.option('--wait/--no-wait', default=True, help='Wait for indexing, with a deadline per document.')
    @click.option('--timeout', type=click.IntRange(min=1), default=300, help='Indexing deadline per document in seconds.')
    @click.option('--monthly-budget', type=click.FloatRange(min=0, min_open=True), default=25., show_default=True,
                  help='Refuse API writes at/above this estimated monthly cost; tokens/downloads are extra.')
    @click.option('--index-gib', type=click.FloatRange(min=0), help='Assumed total billable index GiB; default: Markdown size.')
    @click.option('--searches', type=click.IntRange(min=0), default=0, help='Budget for monthly collection search tool calls.')
    @click.option('--dry-run', is_flag=True, help='Show inventory and costs without xAI API calls.')
    @click.option('--approve-upload', is_flag=True, help='Explicitly authorize upload AFTER reviewing the Markdown cost report.')
    @click.option('--retry-failed', is_flag=True, help='Request reindexing for unchanged failed PDFs.')
    @click.option('--chunk-size', type=click.IntRange(min=1), default=800)
    @click.option('--chunk-overlap', type=click.IntRange(min=0), default=100)
    @click.option('--base-url', default=papers.BASE_URL, show_default=True)
    @click.pass_obj
    def upload_cmd(obj, **kwargs):
        """Upload prepared Markdown only after cost review and explicit approval."""
        papers.upload(obj['data_dir'], **kwargs)

    cli.add_command(upload_cmd, 'papers-update')
    cli.add_command(upload_cmd, 'papers-collect')

    @cli.command('papers-query')
    @click.argument('query', required=False, default='')
    @click.option('--collection-id', default=None)
    @click.option('--model', default='openrouter/x-ai/grok-4.3', show_default=True)
    @click.option('--top-k', type=click.IntRange(1, 100), default=20, show_default=True)
    @click.option('--search-mode', type=click.Choice(['hybrid', 'semantic', 'keyword']), default='hybrid')
    @click.option('--filter', 'filter_str', help='xAI metadata filter expression.')
    @click.option('--nollm', is_flag=True, help='Retrieve passages and citations only.')
    @click.option('--prompt-fragment', default='')
    @click.option('--output-name', default=None)
    @click.option('--css-file', default='wrap.css')
    @click.option('--upload/--no-upload', default=False, help='Publish the answer via the existing scp workflow.')
    @click.option('--remote-user', default='bryan')
    @click.option('--remote-host', default='gnusha.org')
    @click.option('--remote-path', default='~/public_html/irc/chatgpt/papers2/')
    @click.pass_obj
    def query_cmd(obj, query, collection_id, model, top_k, search_mode, filter_str,
                  nollm, prompt_fragment, output_name, css_file, upload,
                  remote_user, remote_host, remote_path):
        """Search xAI's paper Markdown embeddings and answer with linked paper citations."""
        from hpluslogs.services import generation, publishing
        if not query and not prompt_fragment:
            raise click.UsageError('Supply a query or --prompt-fragment.')
        # Validate collection before spending tokens to generate search terms.
        collection_id = collection_id or papers.load_config(obj['data_dir'])['collection_id']
        if not query:
            query = generation.generate_search_query(prompt_fragment, model)
        results = papers.retrieve(obj['data_dir'], query, top_k, search_mode, filter_str, collection_id)
        context = papers.format_context(results)
        output_dir = obj['data_dir'] / 'papers2_queries'
        output_dir.mkdir(parents=True, exist_ok=True)
        name = output_name or datetime.datetime.now(datetime.timezone.utc).strftime('papers_%Y%m%dT%H%M%S_%f')
        if Path(name).name != name or name in ('.', '..'):
            raise click.BadParameter('--output-name must be a filename, not a path.')
        papers.write_json(output_dir / f'{name}.json', {'query': query, 'collection_id': collection_id,
                          'search_mode': search_mode, 'results': results})
        (output_dir / f'{name}.context.md').write_text(context)
        click.echo(f'Retrieved {len(results)} passages. Saved sources to {output_dir / (name + ".json")}.')
        if not results:
            click.echo('No matching passages; no answer generated.')
            return
        if nollm:
            click.echo(context)
            return
        answer = generation.generate_answer(query, results, model, prompt_fragment,
                                             papers.SYSTEM_PROMPT, context_override=context)
        (output_dir / f'{name}.md').write_text(answer)
        click.echo(answer)
        if upload:
            publishing.output(obj['data_dir'], answer, name, css_file, True,
                              remote_user, remote_host, remote_path, prefix='papers_query')
