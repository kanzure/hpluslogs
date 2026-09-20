"""Local PDF-to-Markdown preparation and measured pre-upload cost reports."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import signal
import threading
import sqlite3
import subprocess
import sys
import tempfile
import time

import click

from hpluslogs.services import papers


def root(data_dir):
    return data_dir / 'papers2_markdown'


def connect(data_dir):
    db = sqlite3.connect(data_dir / 'papers2_markdown.sqlite3', timeout=60)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('''CREATE TABLE IF NOT EXISTS conversions (
        pdf_path TEXT PRIMARY KEY, pdf_sha256 TEXT NOT NULL, pdf_bytes INTEGER NOT NULL,
        markdown_path TEXT NOT NULL, markdown_sha256 TEXT, markdown_bytes INTEGER,
        recipe TEXT NOT NULL, status TEXT NOT NULL, error TEXT, updated REAL NOT NULL)''')
    return db


def recipe():
    try:
        return json.dumps({'converter': 'pymupdf4llm', 'version': version('pymupdf4llm'),
                           'pymupdf_version': version('pymupdf'), 'header': False, 'footer': False}, sort_keys=True)
    except Exception as error:
        raise click.ClickException('Install pymupdf4llm==1.28.2 from hpluslogs/requirements.txt.') from error


def output_path(relative):
    # ASCII names support legacy source bytes and long names, with collision-free
    # full path hashes. Original relative paths remain in the manifest and citations.
    import hashlib
    digest = hashlib.sha256(os.fsencode(relative)).hexdigest()
    stem = re.sub(r'[^a-zA-Z0-9._-]', '_', Path(relative).stem)[:80]
    return f'{digest[:2]}/{stem}-{digest}.md'


def convert_one(data_dir, path, settings, timeout):
    pdf_root = papers.paths(data_dir)[0]
    relative = path.relative_to(pdf_root).as_posix()
    key = papers.path_key(relative)
    output = output_path(relative)
    target = root(data_dir) / output
    db = connect(data_dir)
    digest, size = '', 0
    try:
        size = path.stat().st_size
        digest = papers.sha256(path)
        row = db.execute('SELECT * FROM conversions WHERE pdf_path=?', (key,)).fetchone()
        if (row and row['status'] == 'ready' and row['pdf_sha256'] == digest
                and row['recipe'] == settings and target.is_file()
                and papers.sha256(target) == row['markdown_sha256']):
            return key, 'unchanged'
        target.parent.mkdir(parents=True, exist_ok=True)
        with db:
            db.execute('INSERT OR REPLACE INTO conversions VALUES (?,?,?,?,?,?,?,?,?,?)',
                       (key, digest, size, output, None, None, settings, 'converting', None, time.time()))
        with tempfile.TemporaryDirectory(prefix='.convert-', dir=root(data_dir)) as temp:
            result = Path(temp) / 'paper.md'
            # PyMuPDF is not thread-safe; each document runs in its own process,
            # with a deadline so one bad PDF cannot stall the entire collection.
            worker = Path(__file__).with_name('paper_conversion_worker.py')
            command = [sys.executable, str(worker), str(path.resolve()), str(result.resolve()),
                       str((data_dir/'papers2_page_cache').resolve()), digest, settings]
            started = time.monotonic()
            completed = subprocess.run(command, capture_output=True, timeout=timeout)
            remaining = timeout-(time.monotonic()-started)
            if completed.returncode == -11 and remaining > 0:
                completed = subprocess.run(command, capture_output=True, timeout=remaining,
                                           env={**os.environ, 'PAPERS_PARSER':'legacy'})
            if completed.returncode:
                detail = completed.stderr.decode('utf-8', errors='replace')[-2000:]
                raise ValueError(f'Converter exited {completed.returncode}: {detail}')
            text = result.read_text(encoding='utf-8')
            if not text.strip():
                raise ValueError('No Markdown text extracted.')
            if papers.sha256(path) != digest:
                raise ValueError('PDF changed during conversion; rerun papers-markdown after the restore finishes.')
            md_digest, md_size = papers.sha256(result), result.stat().st_size
            result.replace(target)
            metadata = Path(str(result)+'.extraction.json')
            if metadata.exists():
                destination = data_dir/'papers2_extraction_metadata'/(output+'.json')
                destination.parent.mkdir(parents=True,exist_ok=True)
                metadata.replace(destination)
        with db:
            db.execute('INSERT OR REPLACE INTO conversions VALUES (?,?,?,?,?,?,?,?,?,?)',
                       (key, digest, size, output, md_digest, md_size, settings, 'ready', None, time.time()))
        return key, 'converted'
    except Exception as error:
        with db:
            db.execute('INSERT OR REPLACE INTO conversions VALUES (?,?,?,?,?,?,?,?,?,?)',
                       (key, digest, size, output, None, None, settings, 'failed', str(error), time.time()))
        return key, 'error: ' + str(error)
    finally:
        db.close()


def convert(data_dir, workers=2, timeout=600, limit=None):
    settings = recipe()
    with papers.exclusive(data_dir):
        root(data_dir).mkdir(parents=True, exist_ok=True)
        files = list(papers.pdf_files(papers.paths(data_dir)[0]))
        if limit is not None:
            files = files[:limit]
        if not files:
            raise click.ClickException('No local PDFs found. Restore papers2 first.')
        counts = {}
        run = {'started_at': time.time(), 'workers': workers, 'timeout': timeout,
               'pid': os.getpid(), 'state': 'running',
               'inference_threads_per_worker': int(os.environ.get('PAPERS_INFERENCE_THREADS', '1'))}
        run_file = data_dir / 'papers2_conversion_run.json'
        papers.write_json(run_file, run)
        previous_handler = None
        if threading.current_thread() is threading.main_thread():
            previous_handler = signal.getsignal(signal.SIGTERM)
            def stop(signum, frame):
                raise KeyboardInterrupt()
            signal.signal(signal.SIGTERM, stop)
        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            def work(path):
                return convert_one(data_dir, path, settings, timeout)
            futures = [pool.submit(work, path) for path in files]
            for done, future in enumerate(as_completed(futures), 1):
                key, result = future.result()
                kind = 'failed' if result.startswith('error:') else result
                counts[kind] = counts.get(kind, 0) + 1
                click.echo(f'[{done}/{len(files)}] {key}: {result}')
        except BaseException:
            run['state'] = 'interrupted'
            raise
        finally:
            # Do not process the entire queued corpus after an interrupted run.
            # Finish active PDFs and checkpoint them; cancel work not yet started.
            pool.shutdown(wait=True, cancel_futures=True)
            if previous_handler is not None:
                signal.signal(signal.SIGTERM, previous_handler)
            if run['state'] == 'running':
                run['state'] = 'finished_with_failures' if counts.get('failed') else 'finished'
            run['finished_at'] = time.time()
            papers.write_json(run_file, run)
        # Conversion only: cost reporting is deliberately a separate command.
        click.echo(json.dumps(counts, indent=2))
        if counts.get('failed'):
            raise click.ClickException('Some conversions failed. Inspect papers-status; rerun to retry.')
        return counts


def prepare(data_dir, workers=2, timeout=600, limit=None):
    """Complete the local pipeline, always reporting costs after conversion failures."""
    failure = None
    try:
        convert(data_dir, workers, timeout, limit)
    except click.ClickException as error:
        failure = error
    report = cost_report(data_dir)
    click.echo(json.dumps({key: value for key, value in report.items() if key != 'problems'}, indent=2))
    click.echo(f"Full report: {data_dir / 'papers2_markdown_cost.json'}")
    if failure:
        raise failure
    return report


def prepared(data_dir, verify=True):
    """Return only current, verified Markdown. Never silently accept stale outputs."""
    pdf_root = papers.paths(data_dir)[0]
    local = {papers.path_key(p.relative_to(pdf_root).as_posix()): p for p in papers.pdf_files(pdf_root)}
    db = connect(data_dir)
    try:
        rows = {r['pdf_path']: dict(r) for r in db.execute('SELECT * FROM conversions')}
    finally:
        db.close()
    items, problems = [], []
    for key, path in local.items():
        row = rows.get(key)
        if not row or row['status'] != 'ready':
            problems.append({'path': key, 'reason': row['error'] or row['status'] if row else 'not converted'})
            continue
        md = root(data_dir) / row['markdown_path']
        if not md.is_file() or md.is_symlink() or not md.resolve().is_relative_to(root(data_dir).resolve()):
            problems.append({'path': key, 'reason': 'Markdown missing or outside output directory'})
            continue
        if verify and (papers.sha256(path) != row['pdf_sha256'] or papers.sha256(md) != row['markdown_sha256']):
            problems.append({'path': key, 'reason': 'PDF or Markdown changed; run papers-markdown'})
            continue
        if md.stat().st_size > papers.MAX_FILE_BYTES:
            problems.append({'path': key, 'reason': 'Markdown exceeds xAI 100 MB limit; split before upload'})
            continue
        items.append(row)
    return items, problems, len(local)


def cost_report(data_dir, index_gib=None, searches=0, days=30):
    with papers.exclusive(data_dir):
        items, problems, count = prepared(data_dir)
        size = sum(r['markdown_bytes'] for r in items)
        source_bytes = sum(r['pdf_bytes'] for r in items)
        # Include retained remote Markdown for PDFs removed locally: updates never prune.
        db = papers.connect(data_dir)
        current = {r['pdf_path'] for r in items}
        retained = sum(r['size_bytes'] for r in db.execute('SELECT path, size_bytes FROM documents WHERE file_id IS NOT NULL')
                       if r['path'] not in current)
        db.close()
        cost = papers.estimate_cost(size + retained, index_gib, searches, days)
        report = {'format': 'markdown', 'local_pdf_count': count, 'ready_markdown_count': len(items),
                  'markdown_bytes': size, 'markdown_gib': size / 2**30,
                  'converted_source_pdf_bytes': source_bytes, 'retained_remote_markdown_bytes': retained,
                  'local_conversion_complete': bool(count) and not problems,
                  'problems': problems, 'cost': cost,
                  'upload_status': 'HOLD: review measured Markdown costs before explicitly authorizing any xAI upload.'}
        inventory_file = data_dir / 'papers2-remote-inventory.json'
        if inventory_file.exists():
            remote = json.loads(inventory_file.read_text())
            pdf_entries = [e for e in remote if e['path'].lower().endswith('.pdf')]
            expected = [e for e in pdf_entries
                        if not any(part.startswith('.') for part in Path(e['path']).parts)]
            present = {papers.path_key(p.relative_to(papers.paths(data_dir)[0]).as_posix()): p
                       for p in papers.pdf_files(papers.paths(data_dir)[0])}
            missing = [e for e in expected if papers.path_key(e['path']) not in present
                       or present[papers.path_key(e['path'])].stat().st_size != e['size']]
            report['source_inventory_pdf_count'] = len(expected)
            report['source_inventory_excluded_hidden_pdf_entries'] = len(pdf_entries) - len(expected)
            report['source_inventory_missing_or_different_size'] = len(missing)
            report['scope'] = 'full source inventory' if not missing and report['local_conversion_complete'] else 'partial collection only'
        else:
            report['scope'] = 'local PDFs only; restore completeness not independently verified'
        papers.write_json(data_dir / 'papers2_markdown_cost.json', report)
        return report
