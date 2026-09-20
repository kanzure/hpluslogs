"""Render paper-query evidence and publish through the existing Pandoc/SCP adapters."""
import json
from pathlib import Path
import re
import shutil
from urllib.parse import unquote, urlparse

import click

from hpluslogs.integrations import scp
from hpluslogs.services import publishing

REMOTE_PATH = '~/public_html/irc/chatgpt/papers2/'
STYLESHEET_NAME = 'papers.css'


def output_name(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', value):
        raise click.BadParameter('Output name must start with a letter or number and contain only letters, numbers, dots, underscores or hyphens.')
    return value


def preflight(css_file, remote_user, remote_host, remote_path, upload):
    if not shutil.which('pandoc'):
        raise click.ClickException('Install pandoc on this machine to render paper results.')
    css = Path(css_file)
    if not css.is_file() and css_file == 'wrap.css':
        css = Path(__file__).resolve().parents[1]/'wrap.css'
    if not css.is_file():
        raise click.BadParameter(f'CSS file does not exist: {css_file}')
    if upload:
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_-]*', remote_user):
            raise click.BadParameter('Invalid publishing SSH username.')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]*', remote_host):
            raise click.BadParameter('Invalid publishing SSH hostname.')
        if (not remote_path.startswith(('/', '~/')) or
                not re.fullmatch(r'[A-Za-z0-9_./~+-]+', remote_path)):
            raise click.BadParameter('Use an absolute or ~/ publishing directory without spaces or shell metacharacters.')
    return css


def validate_result(result):
    if not isinstance(result, dict) or not isinstance(result.get('question'), str) or not isinstance(result.get('passages'), list):
        raise click.ClickException('Expected a papers-chroma-query JSON result with question and passages.')
    if 'answer' in result and not isinstance(result['answer'], str):
        raise click.ClickException('The saved answer must be Markdown text.')
    generation = result.get('generation', {})
    if not isinstance(generation, dict) or ('complete' in generation and not isinstance(generation['complete'], bool)):
        raise click.ClickException('Invalid saved generation status.')
    for passage in result['passages']:
        if not isinstance(passage, dict) or not isinstance(passage.get('content'), str):
            raise click.ClickException('Invalid saved paper passage.')
        meta = passage.get('metadata', {})
        if not isinstance(meta, dict) or not isinstance(meta.get('pdf_path'), str):
            raise click.ClickException('Missing paper source metadata.')
        url = meta.get('source_url')
        if (not isinstance(url, str) or urlparse(url).scheme not in ('http', 'https')
                or not urlparse(url).netloc or any(c in url for c in '\r\n<>')):
            raise click.ClickException('Paper sources must have HTTP(S) citation URLs.')
    return result


def read_result(path):
    try:
        return validate_result(json.loads(Path(path).read_text(encoding='utf-8')))
    except (OSError, ValueError) as error:
        raise click.ClickException(f'Cannot read paper result: {error}') from error


def label(text):
    return text.replace('\\', '\\\\').replace('[', '\\[').replace(']', '\\]').replace('\n', ' ').replace('\r', ' ')


def answer_markdown(answer, passages):
    """Make LLM lists and numeric citations unambiguous to Markdown readers.

    In particular, [1][2] means a reference link in Markdown, not two
    citations. Emit explicit links with visible brackets instead. Preserve
    code and existing links rather than rewriting their contents.
    """
    urls = {str(i): p['metadata']['source_url'] for i, p in enumerate(passages, 1)}
    number_range = r'\d+(?:\s*[-–]\s*\d+)?'
    citation_numbers = number_range+r'(?:\s*[,;]\s*'+number_range+r')*'
    tokens = re.compile(
        r'(?P<code>`+)(?:(?!(?P=code)).)*?(?P=code)'
        r'|!?\[(?:\\.|[^\]\\])*\]\([^\n]*?\)'
        r'|\[(?!'+citation_numbers+r'\])(?:\\.|[^\]\\])*\]\[[^\]\n]*\]'
        r'|<[^>\n]+>|\\.'
        r'|(?P<citation>\[(?P<numbers>'+citation_numbers+r')\])')

    def cite(match):
        if not match.group('numbers'):
            return match.group(0)
        numbers = []
        for part in re.split(r'[,;]', match.group('numbers')):
            bounds = [int(n.strip()) for n in re.split(r'[-–]', part)]
            start, end = bounds[0], bounds[-1]
            # Do not invent sources or expand arbitrary/out-of-range numbers.
            if not 1 <= start <= end <= len(passages):
                return match.group(0)
            numbers.extend(str(n) for n in range(start, end+1))
        return ' '.join(f'[\\[{n}\\]](<{urls[n]}>)' for n in numbers)

    lines, fence = [], None
    list_item = re.compile(r'^ {0,3}(?:[-+*]|\d+[.)])\s+')
    for line in answer.splitlines():
        marker = re.match(r'^ {0,3}(`{3,}|~{3,})', line)
        if fence:
            lines.append(line)
            if re.fullmatch(r' {0,3}'+re.escape(fence[0])+r'{'+str(len(fence))+r',}\s*', line):
                fence = None
            continue
        if marker:
            fence = marker.group(1)
            lines.append(line)
            continue
        # Indented code and reference definitions are not citation prose.
        if line.startswith(('    ', '\t')) or re.match(r'^ {0,3}\[[^\]]+\]:', line):
            lines.append(line)
            continue
        section = re.fullmatch(r'\*\*([^*]+)\*\*\s*', line)
        if section:
            if lines and lines[-1].strip():
                lines.append('')
            lines.extend(['## '+tokens.sub(cite, section.group(1)), ''])
            continue
        if list_item.match(line) and lines and lines[-1].strip() and not list_item.match(lines[-1]):
            lines.append('')
        lines.append(tokens.sub(cite, line))
    return '\n'.join(lines)


def publish_result(data_dir, result, name, css_file='wrap.css', upload=True,
                   remote_user='bryan', remote_host='gnusha.org', remote_path=REMOTE_PATH):
    """Save JSON, answer and context; render all files before any remote writes."""
    name = output_name(name)
    result = validate_result(result)
    incomplete = result.get('generation', {}).get('complete') is False
    if incomplete:
        upload = False
        if not name.endswith('.partial'):
            name += '.partial'
    css = preflight(css_file, remote_user, remote_host, remote_path, upload)
    data_dir = Path(data_dir)
    saved = data_dir/'papers2_local_queries'/(name+'.json')
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_text(json.dumps(result, indent=2), encoding='utf-8')
    outputs = data_dir/'outputs'
    outputs.mkdir(parents=True, exist_ok=True)
    stylesheet = outputs/STYLESHEET_NAME
    stylesheet.write_bytes(css.read_bytes())
    heading = '# '+label(result['question'])+'\n\n'
    sources, excerpts = [], []
    for i, passage in enumerate(result['passages'], 1):
        meta = passage['metadata']
        title = label(unquote(meta['pdf_path']))
        url = meta['source_url']
        sources.append(f'{i}. [{title}](<{url}>)')
        excerpts.append(f'## Source {i}: {title}\n\n<{url}>\n\n{passage["content"]}')
    context = heading+'\n\n'.join(excerpts) if excerpts else heading+'No matching passages found.\n'
    answer = result.get('answer')
    if incomplete and answer:
        answer = '> **Incomplete report: output token limit reached.**\n\n'+answer
    markdown = (heading+answer_markdown(answer, result['passages'])+'\n\n## Sources\n\n'+'\n'.join(sources)+'\n') if answer else context
    for suffix, content in (('.context', context), ('', markdown)):
        try:
            publishing.output(data_dir, content, name+suffix, stylesheet.name,
                              upload=False, prefix='papers_query', strict=True)
        except RuntimeError as error:
            raise click.ClickException(str(error)) from error
    files = [stylesheet, outputs/(name+'.context.md'), outputs/(name+'.context.html'),
             outputs/(name+'.md'), outputs/(name+'.html')]
    destination = None
    if upload:
        remote_path = remote_path.rstrip('/')+'/'
        if not scp.ensure_remote_directory(remote_user, remote_host, remote_path):
            raise click.ClickException('Cannot create the remote papers directory; local results remain saved.')
        for path in files:
            if not scp.upload_file(path, remote_user, remote_host, remote_path, path.name):
                raise click.ClickException(f'Upload failed for {path.name}; rerun papers-publish with {saved} to retry without model calls.')
        destination = f'{remote_user}@{remote_host}:{remote_path}'
    return {'result_json': str(saved), 'files': [str(p) for p in files], 'uploaded_to': destination,
            'complete': not incomplete}
