"""One isolated PyMuPDF conversion; called by papers-markdown, never uploads."""
from pathlib import Path
import os
import sys


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


def main():
    configure_inference_threads(int(os.environ.get('PAPERS_INFERENCE_THREADS', '1')))
    import pymupdf4llm
    # Match ~/papers/physical-intelligence/run.py exactly.
    source = sys.argv[1]
    if any(0xdc80 <= ord(char) <= 0xdcff for char in source):
        # MuPDF's filename interface cannot encode legacy Unix filename bytes.
        # Python can open them losslessly; keep the same document parser.
        import pymupdf
        source = pymupdf.open(stream=Path(source).read_bytes(), filetype='pdf')
    markdown = pymupdf4llm.to_markdown(source, header=False, footer=False)
    if not isinstance(markdown, str) or not markdown.strip():
        raise ValueError('No Markdown text extracted; inspect the PDF/OCR result.')
    Path(sys.argv[2]).write_text(markdown, encoding='utf-8')


if __name__ == '__main__':
    main()
