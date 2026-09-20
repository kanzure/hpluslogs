"""Targeted, resumable extraction repair without embedding or uploading."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.parse import unquote_to_bytes

import click
from hpluslogs.services import papers, papers_markdown as markdown


def suspect_text(text):
    count = sum((ord(c) < 32 and c not in '\n\r\t') or 127 <= ord(c) <= 159
                or c == '\ufffd' for c in text)
    return count >= 32 and count / max(1, len(text)) > .01


def candidates(data_dir):
    from hpluslogs.services.papers_failures import classify
    with markdown.connect(data_dir) as db:
        rows = [dict(r) for r in db.execute('SELECT * FROM conversions ORDER BY pdf_path')]
    result = []
    for row in rows:
        if row['status'] == 'ready':
            path = markdown.root(data_dir)/row['markdown_path']
            if suspect_text(path.read_text(encoding='utf-8')):
                result.append({**row, 'repair_mode': 'raster-ocr'})
        elif row['status'] == 'failed':
            path = data_dir/'papers2'/os.fsdecode(unquote_to_bytes(row['pdf_path']))
            if classify(path)['classification'] == 'readable_pdf_extraction_failed':
                result.append({**row, 'repair_mode': 'legacy'})
    return result


def repair_one(data_dir, row, timeout):
    source = data_dir/'papers2'/os.fsdecode(unquote_to_bytes(row['pdf_path']))
    target = markdown.root(data_dir)/row['markdown_path']
    started = time.monotonic()
    try:
        if papers.sha256(source) != row['pdf_sha256']:
            raise ValueError('Source changed; run normal conversion first.')
        with tempfile.TemporaryDirectory(prefix='.repair-', dir=markdown.root(data_dir)) as tmp:
            output = Path(tmp)/'paper.md'
            modes = ['legacy', 'raster-ocr'] if row['repair_mode'] == 'legacy' else ['raster-ocr']
            for mode in modes:
                remaining = timeout-(time.monotonic()-started)
                if remaining <= 0:
                    raise TimeoutError('Repair deadline exceeded.')
                command = [sys.executable, str(Path(markdown.__file__).with_name('paper_conversion_worker.py')),
                           str(source.resolve()), str(output.resolve()),
                           str((data_dir/'papers2_page_cache').resolve()), row['pdf_sha256'], row['recipe']]
                completed = subprocess.run(command, capture_output=True, timeout=remaining,
                                           env={**os.environ, 'PAPERS_PARSER': mode})
                if completed.returncode:
                    if mode != modes[-1]:
                        continue
                    raise ValueError(completed.stderr.decode(errors='replace')[-1500:])
                text = output.read_text(encoding='utf-8')
                if text.strip() and not suspect_text(text):
                    break
                if mode == modes[-1]:
                    raise ValueError('Recovery still has empty or garbled text.')
            if papers.sha256(source) != row['pdf_sha256']:
                raise ValueError('Source changed during repair.')
            sha, size = papers.sha256(output), output.stat().st_size
            with markdown.connect(data_dir) as db:
                db.execute('BEGIN IMMEDIATE')
                current = db.execute('SELECT * FROM conversions WHERE pdf_path=?', (row['pdf_path'],)).fetchone()
                if dict(current) != {k:v for k,v in row.items() if k != 'repair_mode'}:
                    raise ValueError('Conversion checkpoint changed during repair.')
                if target.exists():
                    backup = data_dir/'papers2_repair_backups'/papers.sha256(target)
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    if not backup.exists():
                        shutil.copyfile(target, backup)
                target.parent.mkdir(parents=True, exist_ok=True)
                output.replace(target)
                metadata = data_dir/'papers2_extraction_metadata'/(row['markdown_path']+'.json')
                metadata.parent.mkdir(parents=True, exist_ok=True)
                Path(str(output)+'.extraction.json').replace(metadata)
                db.execute("UPDATE conversions SET markdown_sha256=?, markdown_bytes=?, status='ready', error=NULL, updated=? WHERE pdf_path=?",
                           (sha, size, time.time(), row['pdf_path']))
        return {'paper': row['pdf_path'], 'result': 'repaired', 'mode': mode, 'markdown_bytes': size}
    except Exception as error:
        return {'paper': row['pdf_path'], 'result': 'failed', 'error': str(error)}


def repair(data_dir, workers=8, timeout=1800, limit=None, apply=False):
    with papers.exclusive(data_dir):
        rows = candidates(data_dir)
        if limit is not None:
            rows = rows[:limit]
        report = {'started_at': time.time(), 'apply': apply, 'candidates': len(rows),
                  'workers': workers, 'results': []}
        report_path = data_dir/'papers2_repair_report.json'
        if not apply:
            report['results'] = [{'paper':r['pdf_path'], 'mode':r['repair_mode']} for r in rows]
            papers.write_json(report_path, report)
            click.echo(json.dumps(report, indent=2))
            return report
        papers.write_json(report_path, report)
        click.echo(json.dumps({'repair_candidates': len(rows), 'workers': workers}))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(repair_one, data_dir, row, timeout) for row in rows]
            for future in as_completed(futures):
                item = future.result()
                report['results'].append(item)
                papers.write_json(report_path, report)
                click.echo(json.dumps(item))
        report['finished_at'] = time.time()
        papers.write_json(report_path, report)
        if any(r['result'] == 'failed' for r in report['results']):
            raise click.ClickException('Some repairs failed; inspect papers2_repair_report.json. Page checkpoints are preserved.')
        return report
