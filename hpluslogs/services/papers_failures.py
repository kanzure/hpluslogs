"""Inspect failed source PDFs without changing files or conversion checkpoints."""
from collections import Counter
import hashlib
import os
from pathlib import Path
import sqlite3
from urllib.parse import unquote_to_bytes


def classify(path):
    import pymupdf
    raw = path.read_bytes()
    result = {'bytes':len(raw), 'sha256':hashlib.sha256(raw).hexdigest()}
    if not raw:
        return {**result, 'classification':'empty_file'}
    try:
        with pymupdf.open(stream=raw,filetype='pdf') as doc:
            result['pages'] = len(doc)
            if not doc.is_pdf:
                result['classification'] = 'not_pdf'
            elif doc.needs_pass:
                result['classification'] = 'password_required'
            elif not len(doc):
                result['classification'] = 'no_readable_pages'
            else:
                result['classification'] = 'readable_pdf_extraction_failed'
    except Exception as error:
        result.update(classification='unreadable_pdf', inspection_error=str(error))
    return result


def report(data_dir):
    manifest = data_dir/'papers2_markdown.sqlite3'
    with sqlite3.connect(f'{manifest.resolve().as_uri()}?mode=ro',uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(r) for r in db.execute("SELECT * FROM conversions WHERE status='failed' ORDER BY pdf_path")]
    results = []
    for row in rows:
        path = data_dir/'papers2'/os.fsdecode(unquote_to_bytes(row['pdf_path']))
        try:
            item = classify(path)
            if item['sha256'] != row['pdf_sha256']:
                item['classification'] = 'source_changed_since_failure'
        except OSError as error:
            item = {'classification':'source_unavailable','inspection_error':str(error)}
        item.update(pdf_path=row['pdf_path'], conversion_error=row['error'])
        results.append(item)
    return {'failed_papers':len(results), 'classifications':dict(Counter(r['classification'] for r in results)),
            'papers':results, 'scope':'Snapshot of failed conversions only; no sources or checkpoints modified.'}
