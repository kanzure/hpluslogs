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
    @click.option('--workers', type=click.IntRange(1,64), default=4)
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

    @cli.command('papers-chroma-query')
    @connection_options
    @click.argument('question')
    @click.option('--top-k', type=click.IntRange(1,100), default=8)
    @click.option('--nollm', is_flag=True, help='Return source passages without generating an answer.')
    @click.option('--llm-url', default='http://127.0.0.1:8080/v1', envvar='PAPERS_LLM_URL')
    @click.option('--model', default=None, envvar='PAPERS_LLM_MODEL', help='Served model name; required unless --nollm.')
    @click.option('--output-name', default=None)
    @click.pass_obj
    def query_cmd(obj, chroma_host, chroma_port, question, top_k, nollm, llm_url, model, output_name):
        if not nollm and not model:
            raise click.UsageError('Supply --model or use --nollm.')
        name = output_name or datetime.datetime.now(datetime.timezone.utc).strftime('local_%Y%m%dT%H%M%S_%f')
        if Path(name).name != name or name in ('.','..'):
            raise click.BadParameter('--output-name must be a filename.')
        passages = papers_chroma.retrieve(obj['data_dir'], chroma_host, chroma_port, question, top_k)
        result = {'question': question, 'collection': papers_chroma.COLLECTION, 'passages': passages}
        if passages and not nollm:
            result['answer'] = papers_chroma.answer(question, passages, llm_url, model)
        dest = obj['data_dir']/'papers2_local_queries'
        dest.mkdir(parents=True, exist_ok=True)
        (dest/(name+'.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
        click.echo(json.dumps(result, indent=2))


@click.group()
@click.option('--data-dir', type=click.Path(path_type=Path), default='/data')
@click.pass_context
def cli(ctx, data_dir):
    ctx.obj = {'data_dir': data_dir}


register(cli)
if __name__ == '__main__':
    cli()
