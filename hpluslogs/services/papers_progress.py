"""Read-only conversion progress, throughput and provisional xAI cost estimates."""
import datetime
import json
import sqlite3
import time

from hpluslogs.services import papers


def report(data_dir, days=30, searches=0, index_multiplier=1.0, now=None):
    now = time.time() if now is None else now
    sources = {papers.path_key(p.relative_to(data_dir / 'papers2').as_posix()): p.stat().st_size
               for p in papers.pdf_files(data_dir / 'papers2')}
    manifest = data_dir / 'papers2_markdown.sqlite3'
    rows = []
    if manifest.exists():
        with sqlite3.connect(f'{manifest.resolve().as_uri()}?mode=ro', uri=True) as db:
            db.row_factory = sqlite3.Row
            rows = [dict(r) for r in db.execute('SELECT * FROM conversions') if r['pdf_path'] in sources]
    # This is a quick estimate, not the hash-verified pre-upload cost gate.
    ready = []
    for row in rows:
        path = data_dir / 'papers2_markdown' / row['markdown_path']
        if (row['status'] == 'ready' and sources[row['pdf_path']] == row['pdf_bytes']
                and path.is_file() and not path.is_symlink()
                and path.stat().st_size == row['markdown_bytes']):
            ready.append(row)
    failed = sum(r['status'] == 'failed' for r in rows)
    converting = sum(r['status'] == 'converting' for r in rows)
    size = sum(r['markdown_bytes'] for r in ready)
    source_size = sum(r['pdf_bytes'] for r in ready)
    run_file = data_dir / 'papers2_conversion_run.json'
    run = json.loads(run_file.read_text()) if run_file.exists() else {}
    start = run.get('started_at')
    rates = []
    failed_this_pass = sum(r['status'] == 'failed' and r['updated'] >= (start or 0) for r in rows)
    remaining = len(sources) - len(ready) - failed_this_pass
    for minutes in (15, 60, 120):
        cutoff = max(now - minutes * 60, start or 0)
        elapsed = now - cutoff
        completed = sum(r['status'] in ('ready', 'failed') and r['updated'] > cutoff for r in rows)
        rate = completed / elapsed * 3600 if elapsed >= 60 else 0
        rates.append({'window_minutes': minutes, 'attempts': completed, 'papers_per_hour': rate,
                      'first_pass_remaining_hours': remaining / rate if rate and remaining else None})
    projections = {}
    if ready:
        projected = {'per_paper': size / len(ready) * len(sources),
                     'pdf_byte_ratio': size / source_size * sum(sources.values()) if source_size else 0}
        for name, estimate in projected.items():
            projections[name] = papers.estimate_cost(estimate, estimate / 2**30 * index_multiplier, searches, days)
    return {'measured_at': datetime.datetime.fromtimestamp(now, datetime.timezone.utc).isoformat(),
            'pdf_count': len(sources), 'ready': len(ready), 'failed': failed,
            'converting_or_interrupted': converting, 'first_pass_remaining': remaining,
            'markdown_bytes': size, 'markdown_mib': size / 2**20, 'markdown_gib': size / 2**30,
            'run': run, 'throughput': rates,
            'current_storage': papers.estimate_cost(size, size / 2**30 * index_multiplier, searches, days),
            'full_archive_projections': projections,
            'caveats': ['ETA covers the first pass, excluding repair/retries; document sizes vary.',
                        'Converting rows may be interrupted; inspect the process/container state.',
                        'Size checks only; use papers-cost for hash-verified pre-upload estimates.',
                        'Projections extrapolate a nonrandom successful subset; index size is assumed.',
                        'Model tokens and downloads are extra. No API calls or uploads.']}
