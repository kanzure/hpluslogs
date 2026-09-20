"""One isolated PyMuPDF conversion; called by papers-markdown, never uploads."""
from pathlib import Path
import hashlib
import json
import os
import sys
import tempfile

USED_PARSERS = set()


def render_markdown(source, **kwargs):
    """Recover existing hidden OCR text when layout extraction is empty/broken."""
    import pymupdf4llm
    primary_error = None
    if os.environ.get('PAPERS_PARSER') != 'legacy':
        try:
            text = pymupdf4llm.to_markdown(source, **kwargs)
            if isinstance(text,str) and text.strip():
                render_markdown.last_parser = 'layout'
                USED_PARSERS.add('layout')
                return text
        except Exception as error:
            primary_error = error
    # A visible licensing footer can prevent automatic recognition of hidden OCR.
    # Keep the existing text layer; this mode does not perform new OCR.
    legacy = {k:v for k,v in kwargs.items() if k not in ('header','footer')}
    try:
        pymupdf4llm.use_layout(False)
        text = pymupdf4llm.to_markdown(source, ignore_alpha=True, ignore_images=True,
                                     ignore_graphics=True, **legacy)
        render_markdown.last_parser = 'legacy-hidden-text'
        USED_PARSERS.add('legacy-hidden-text')
        return text
    except Exception:
        if primary_error is not None:
            raise primary_error
        raise
    finally:
        pymupdf4llm.use_layout(True)


def configure_inference_threads(threads=1):
    """Bound each process's ONNX pools before PyMuPDF creates any sessions."""
    if threads < 1:
        raise ValueError('PAPERS_INFERENCE_THREADS must be positive.')
    import onnxruntime as ort
    original = ort.InferenceSession

    class BoundedInferenceSession(original):
        def __init__(self, path_or_bytes, sess_options=None, *args, **kwargs):
            if sess_options is None:
                sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = threads
            sess_options.inter_op_num_threads = 1
            sess_options.add_session_config_entry('session.intra_op.allow_spinning', '0')
            sess_options.add_session_config_entry('session.inter_op.allow_spinning', '0')
            super().__init__(path_or_bytes, sess_options, *args, **kwargs)

    # PyMuPDF creates sessions internally without a public threading option.
    # This process-local adapter preserves its other SessionOptions/providers.
    ort.InferenceSession = BoundedInferenceSession


def cached_markdown(document, cache_dir, render, page_size=20):
    """Commit complete page batches; interrupted batches are safe to repeat."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = []
    for first in range(0, len(document), page_size):
        end = min(first+page_size, len(document))
        path = cache_dir/f'{first:08d}-{end:08d}.json'
        record = None
        try:
            candidate = json.loads(path.read_text(encoding='utf-8'))
            if (candidate['pages'] == [first,end] and isinstance(candidate['markdown'],str)
                    and hashlib.sha256(candidate['markdown'].encode()).hexdigest() == candidate['sha256']
                    and (candidate['markdown'].strip() or candidate.get('empty_policy') == 'legacy-fallback-v1')):
                record = candidate
        except (OSError, ValueError, KeyError, TypeError):
            pass
        if record is None:
            text = render(document, pages=list(range(first,end)), header=False, footer=False)
            if not isinstance(text,str):
                raise ValueError('Page batch did not return Markdown text.')
            record = {'pages':[first,end], 'markdown':text,
                      'sha256':hashlib.sha256(text.encode()).hexdigest(),
                      'parser':getattr(render,'last_parser','layout'), 'empty_policy':'legacy-fallback-v1'}
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=cache_dir,
                                             suffix='.tmp', delete=False) as handle:
                json.dump(record,handle)
                temp = Path(handle.name)
            temp.replace(path)
        output.append(record['markdown'])
        USED_PARSERS.add(record.get('parser','layout'))
    return '\n\n'.join(output)


def convert_with_page_cache(source, source_path, cache_root, expected_sha, settings, render):
    import pymupdf
    probe = pymupdf.open(source) if isinstance(source,str) else source
    long_document = len(probe) >= 200
    if isinstance(source,str):
        probe.close()
    if not long_document:
        return render(source, header=False, footer=False)
    # Parse an immutable byte snapshot so source edits cannot contaminate saved batches.
    raw = source_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('PDF changed before page-batch conversion.')
    try:
        tessdata = Path(pymupdf.get_tessdata())/'eng.traineddata'
        ocr = hashlib.sha256(tessdata.read_bytes()).hexdigest()
    except (OSError, RuntimeError, TypeError, ValueError):
        ocr = 'unavailable'
    cache_key = hashlib.sha256((expected_sha+settings+ocr+':pages20-v1').encode()).hexdigest()
    with pymupdf.open(stream=raw,filetype='pdf') as document:
        return cached_markdown(document, cache_root/cache_key, render)


def main():
    USED_PARSERS.clear()
    configure_inference_threads(int(os.environ.get('PAPERS_INFERENCE_THREADS', '1')))
    import pymupdf4llm
    # Primary parser/settings match ~/papers/physical-intelligence/run.py.
    source = sys.argv[1]
    if any(0xdc80 <= ord(char) <= 0xdcff for char in source):
        # MuPDF's filename interface cannot encode legacy Unix filename bytes.
        # Python can open them losslessly; keep the same document parser.
        import pymupdf
        source = pymupdf.open(stream=Path(source).read_bytes(), filetype='pdf')
    if len(sys.argv) > 3:
        markdown = convert_with_page_cache(source, Path(sys.argv[1]), Path(sys.argv[3]), sys.argv[4], sys.argv[5], render_markdown)
    else:
        markdown = render_markdown(source, header=False, footer=False)
    if not isinstance(markdown, str) or not markdown.strip():
        raise ValueError('No Markdown text extracted; inspect the PDF/OCR result.')
    Path(sys.argv[2]).write_text(markdown, encoding='utf-8')
    Path(sys.argv[2]+'.extraction.json').write_text(json.dumps({'parsers':sorted(USED_PARSERS)}),encoding='utf-8')


if __name__ == '__main__':
    main()
