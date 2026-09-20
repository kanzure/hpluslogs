"""Small, conversion-only entry point for the offline Docker image."""
import json
from pathlib import Path

import click
from hpluslogs.services import papers_markdown, papers_progress


@click.group()
@click.option('--data-dir', type=click.Path(path_type=Path), default='/data')
@click.pass_context
def cli(ctx, data_dir):
    ctx.obj = data_dir


@cli.command()
@click.option('--workers', type=click.IntRange(min=1), default=4)
@click.option('--timeout', type=click.IntRange(min=1), default=600)
@click.pass_obj
def convert(data_dir, workers, timeout):
    """Resume PDF conversion, without uploading or indexing."""
    papers_markdown.convert(data_dir, workers, timeout)


@cli.command()
@click.option('--days', type=click.IntRange(min=1), default=30)
@click.option('--searches', type=click.IntRange(min=0), default=0)
@click.option('--index-multiplier', type=click.FloatRange(min=0), default=1.0)
@click.pass_obj
def progress(data_dir, **kwargs):
    """Read current sizes, ETA and monthly cost projections."""
    click.echo(json.dumps(papers_progress.report(data_dir, **kwargs), indent=2))


@cli.command()
@click.pass_obj
def failures(data_dir):
    """Classify failed PDFs; distinguish damaged inputs from extraction failures."""
    from hpluslogs.services import papers_failures
    result = papers_failures.report(data_dir)
    (data_dir/'papers2_failure_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    click.echo(json.dumps({'failed_papers':result['failed_papers'], 'classifications':result['classifications']}, indent=2))


if __name__ == '__main__':
    cli()
