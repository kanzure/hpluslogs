"""Flat paper Chroma commands, also usable in the small Docker image."""
import datetime
import json
from pathlib import Path
import click
from hpluslogs.services import papers_chroma


def connection_options(fn):
    fn = click.option('--chroma-host', default='127.0.0.1', envvar='PAPERS_CHROMA_HOST')(fn)
    return click.option('--chroma-port', type=click.IntRange(1,65535), default=18081, envvar='PAPERS_CHROMA_PORT')(fn)


def register(cli):
    @cli.command('papers-chroma-index')
    @connection_options
    @click.option('--concurrency', '--workers', type=click.IntRange(1,256), default=80, show_default=True)
    @click.option('--batch-size', type=click.IntRange(1,1000), default=1000, show_default=True)
    @click.option('--cost-limit', type=click.FloatRange(min=0,min_open=True), default=5.0, show_default=True)
    @click.option('--limit', type=click.IntRange(1), default=None, help='Limit pending papers for a smoke test.')
    @click.option('--watch', is_flag=True, help='Continuously index newly completed Markdown.')
    @click.option('--interval', type=click.IntRange(10), default=120)
    @click.pass_obj
    def index_cmd(obj, chroma_host, chroma_port, **kwargs):
        papers_chroma.index(obj['data_dir'], chroma_host, chroma_port, **kwargs)

    @cli.command('papers-chroma-status')
    @connection_options
    @click.pass_obj
    def status_cmd(obj, chroma_host, chroma_port):
        click.echo(json.dumps(papers_chroma.status(obj['data_dir'], chroma_host, chroma_port), indent=2))

    @cli.command('papers-chroma-audit')
    @connection_options
    @click.option('--allow-skipped', is_flag=True, help='Permit explicitly reported unreadable Markdown exclusions.')
    @click.pass_obj
    def audit_cmd(obj, chroma_host, chroma_port, allow_skipped):
        """Check every stored passage against current ready Markdown."""
        result = papers_chroma.audit(obj['data_dir'], chroma_host, chroma_port)
        path = obj['data_dir']/'papers2_chroma_audit.json'
        path.write_text(json.dumps(result, indent=2), encoding='utf-8')
        click.echo(json.dumps(result, indent=2))
        if not result['all_eligible_markdown_verified' if allow_skipped else 'all_ready_markdown_verified']:
            raise click.ClickException('Archive audit is incomplete; inspect the saved report.')

    @cli.command('papers-chroma-query')
    @connection_options
    @click.argument('question')
    @click.option('--top-k', type=click.IntRange(1,100), default=8)
    @click.option('--nollm', is_flag=True, help='Return source passages without generating an answer.')
    @click.option('--llm-url', default='http://127.0.0.1:8080/v1', envvar='PAPERS_LLM_URL')
    @click.option('--model', default=None, envvar='PAPERS_LLM_MODEL', help='Served model name; required unless --nollm.')
    @click.option('--output-name', default=None)
    @click.option('--brief', is_flag=True, help='Generate a concise answer instead of a technical report.')
    @click.option('--prompt-fragment', default='', help='Additional report instructions; does not change retrieval.')
    @click.option('--max-answer-tokens', type=click.IntRange(1, 32768), default=None,
                  help='Output token limit: 8192 for reports, 1800 for brief answers.')
    @click.pass_obj
    def query_cmd(obj, chroma_host, chroma_port, question, top_k, nollm, llm_url, model, output_name,
                  brief, prompt_fragment, max_answer_tokens):
        if not nollm and not model:
            raise click.UsageError('Supply --model or use --nollm.')
        name = output_name or datetime.datetime.now(datetime.timezone.utc).strftime('local_%Y%m%dT%H%M%S_%f')
        if Path(name).name != name or name in ('.','..'):
            raise click.BadParameter('--output-name must be a filename.')
        passages = papers_chroma.retrieve(obj['data_dir'], chroma_host, chroma_port, question, top_k)
        result = {'question': question, 'collection': papers_chroma.COLLECTION, 'passages': passages}
        dest = obj['data_dir']/'papers2_local_queries'
        dest.mkdir(parents=True, exist_ok=True)
        saved = dest/(name+'.json')
        saved.write_text(json.dumps(result, indent=2), encoding='utf-8')
        if passages and not nollm:
            result['answer'] = papers_chroma.answer(question, passages, llm_url, model,
                brief=brief, prompt_fragment=prompt_fragment, max_answer_tokens=max_answer_tokens)
            result['generation'] = {'mode': 'brief' if brief else 'report', 'model': model,
                'prompt_fragment': prompt_fragment, 'max_answer_tokens': max_answer_tokens or (1800 if brief else 8192)}
            saved.write_text(json.dumps(result, indent=2), encoding='utf-8')
        click.echo(json.dumps(result, indent=2))


@click.group()
@click.option('--data-dir', type=click.Path(path_type=Path), default='/data')
@click.pass_context
def cli(ctx, data_dir):
    ctx.obj = {'data_dir': data_dir}


register(cli)
if __name__ == '__main__':
    cli()
