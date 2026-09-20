"""One isolated PyMuPDF conversion; called by papers-markdown, never uploads."""
from pathlib import Path
import sys


def main():
    import pymupdf4llm
    # Match ~/papers/physical-intelligence/run.py exactly.
    markdown = pymupdf4llm.to_markdown(sys.argv[1], header=False, footer=False)
    if not isinstance(markdown, str) or not markdown.strip():
        raise ValueError('No Markdown text extracted; inspect the PDF/OCR result.')
    Path(sys.argv[2]).write_text(markdown, encoding='utf-8')


if __name__ == '__main__':
    main()
