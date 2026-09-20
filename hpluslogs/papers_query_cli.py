"""Local rendering/publishing frontend for the remote paper RAG worker."""
import datetime
import json
from pathlib import Path
import subprocess

import click

from hpluslogs.papers_remote_cli import connection_options
from hpluslogs.services import paper_results
from hpluslogs.services.papers_remote import Remote


def publishing_options(fn):
    for option, kwargs in [
        ('--output-name', {'default': None, 'help': 'Base filename without extension.'}),
        ('--css-file', {'default': 'wrap.css'}),
        ('--upload/--no-upload', {'default': True, 'show_default': True}),
        ('--remote-user', {'default': 'bryan', 'show_default': True}),
        ('--remote-host', {'default': 'gnusha.org', 'show_default': True}),
        ('--remote-path', {'default': paper_results.REMOTE_PATH, 'show_default': True}),
    ]:
        fn = click.option(option, **kwargs)(fn)
    return fn


def show_outputs(result, report):
    click.echo(result.get('answer') or f'Retrieved {len(result["passages"])} passages.')
    click.echo(json.dumps(report, indent=2))


def register(cli):
    @cli.command('papers-remote-query')
    @connection_options
    @click.argument('question')
    @click.option('--top-k', type=click.IntRange(1, 100), default=8)
    @click.option('--nollm', is_flag=True, help='Render and optionally publish retrieved passages only.')
    @click.option('--model', default=None, envvar='PAPERS_LLM_MODEL')
    @click.option('--llm-url', default='http://127.0.0.1:8080/v1', envvar='PAPERS_LLM_URL')
    @click.option('--chroma-port', type=click.IntRange(1, 65535), default=18081)
    @publishing_options
    @click.pass_obj
    def query(obj, host, user, path, question, top_k, nollm, model, llm_url,
              chroma_port, output_name, **options):
        """Query remote Chroma/LLM, then save Markdown/HTML and publish from here."""
        if not nollm and not model:
            raise click.UsageError('Supply --model or use --nollm.')
        name = paper_results.output_name(output_name or datetime.datetime.now(datetime.timezone.utc).strftime('papers_%Y%m%dT%H%M%S_%f'))
        paper_results.preflight(**options)
        remote = Remote(host, user, path)
        command = ['docker', 'exec', remote.name+'-chroma-index', 'python', '-m',
                   'hpluslogs.papers_chroma_cli', 'papers-chroma-query', '--top-k', str(top_k),
                   '--chroma-port', str(chroma_port), '--llm-url', llm_url, '--output-name', name]
        command += ['--nollm'] if nollm else ['--model', model]
        try:
            # Remote.command quotes each argument; the question is never shell code.
            completed = remote.command(*command, '--', question, capture=True)
            result = paper_results.validate_result(json.loads(completed.stdout))
        except (subprocess.CalledProcessError, OSError, ValueError) as error:
            raise click.ClickException(f'Remote paper query failed: {error}') from error
        report = paper_results.publish_result(obj['data_dir'], result, name, **options)
        show_outputs(result, report)

    @cli.command('papers-publish')
    @click.argument('result_file', type=click.Path(exists=True, dir_okay=False, path_type=Path))
    @publishing_options
    @click.pass_obj
    def publish(obj, result_file, output_name, **options):
        """Render/publish saved paper-query JSON; no retrieval or model calls."""
        result = paper_results.read_result(result_file)
        report = paper_results.publish_result(obj['data_dir'], result,
            output_name or result_file.stem, **options)
        show_outputs(result, report)
