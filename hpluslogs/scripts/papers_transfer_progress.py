#!/usr/bin/env python3
"""Sample remote PDF transfer progress/ETA over SSH; never modify or upload data."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hpluslogs.services import papers
from hpluslogs.services.papers_remote import Remote


def summarize(expected, actual):
    copied = complete = 0
    for key, (size, mtime) in expected.items():
        received = actual.get(key)
        if received is None:
            continue
        if received == [size, mtime]:
            copied += size; complete += 1
        elif received[0] < size:
            copied += received[0]
    return copied, complete


def measure(remote, expected):
    code = ("import os,json,sys; from pathlib import Path; from urllib.parse import quote_from_bytes; "
            "r=Path(sys.argv[1]); print(json.dumps({quote_from_bytes(os.fsencode(p.relative_to(r).as_posix()),safe='/'): "
            "[p.stat().st_size,p.stat().st_mtime_ns] for p in r.rglob('*') "
            "if p.is_file() and not p.is_symlink() and p.suffix.lower()=='.pdf'}))")
    actual = json.loads(remote.command('python3', '-c', code, remote.path+'/data/papers2', capture=True).stdout)
    return (*summarize(expected, actual), time.monotonic())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('host', 'user', 'path'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('hpluslogs/data'))
    parser.add_argument('--interval', type=float, default=20, help='Seconds between samples (1–60).')
    args = parser.parse_args()
    if not 1 <= args.interval <= 60:
        parser.error('--interval must be between 1 and 60 seconds.')
    remote = Remote(args.host, args.user, args.path)
    root = args.data_dir/'papers2'
    expected = {papers.path_key(p.relative_to(root).as_posix()): [p.stat().st_size, p.stat().st_mtime_ns]
                for p in papers.pdf_files(root)}
    total = sum(row[0] for row in expected.values())
    first, _, t0 = measure(remote, expected)
    time.sleep(args.interval)
    copied, complete, t1 = measure(remote, expected)
    rate = max(0, copied - first) / (t1 - t0)
    remaining = max(0, total - copied)
    print(json.dumps({'measured_at': datetime.now(timezone.utc).isoformat(),
                      'pdf_count': len(expected), 'complete_files': complete,
                      'pdf_gib': total / 2**30, 'copied_pdf_gib': copied / 2**30,
                      'percent': copied / total * 100 if total else 0,
                      'sample_seconds': t1-t0, 'pdf_mib_per_second': rate / 2**20,
                      'pdf_transfer_eta_hours': remaining / rate / 3600 if rate else (0 if not remaining else None),
                      'conversion_container_state': remote.state() or 'absent',
                      'caveat': 'PDF transfer only; Markdown/checkpoint transfer and conversion follow. Null ETA means no measured progress.'}, indent=2))


if __name__ == '__main__':
    main()
