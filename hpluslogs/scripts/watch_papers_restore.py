#!/usr/bin/env python3
"""Host-side restic completion watcher. Standard library only; never calls xAI."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time
from urllib.parse import unquote_to_bytes


def restore_processes(staging, proc_root=Path('/proc')):
    matches = []
    for process in proc_root.iterdir():
        if not process.name.isdigit():
            continue
        try:
            args = (process/'cmdline').read_bytes().split(b'\0')
            if (Path(os.fsdecode(args[0])).name == 'restic' and b'restore' in args
                    and b'--target' in args and args[args.index(b'--target')+1] == os.fsencode(staging)):
                state = (process/'stat').read_text().rsplit(') ', 1)[1].split()[0]
                if state != 'Z':
                    matches.append(int(process.name))
        except (OSError, IndexError, ValueError):
            continue
    return matches


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_sources(root, entries):
    def verify(entry):
        relative = Path(os.fsdecode(unquote_to_bytes(entry['path'])))
        if relative.is_absolute() or '..' in relative.parts:
            return entry['path'] + ': unsafe path'
        path = root/relative
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            return entry['path'] + ': unsafe link'
        if not path.is_file() or path.stat().st_size != entry['size']:
            return entry['path'] + ': missing or size differs'
        if digest(path) != entry['sha256']:
            return entry['path'] + ': checksum differs'
        return None
    with ThreadPoolExecutor(max_workers=4) as pool:
        errors = [error for error in pool.map(verify, entries) if error]
    if errors:
        raise ValueError(f'{len(errors)} restored PDF(s) differ: ' + '; '.join(errors[:10]))


def verify_checkpoint(data_dir):
    manifest = data_dir/'papers2_markdown.sqlite3'
    with sqlite3.connect(f'{manifest.resolve().as_uri()}?mode=ro', uri=True) as db:
        rows = db.execute("SELECT markdown_path, markdown_sha256 FROM conversions WHERE status='ready'").fetchall()
    for relative, expected in rows:
        path = data_dir/'papers2_markdown'/relative
        if not path.is_file() or path.is_symlink() or digest(path) != expected:
            raise ValueError(f'Saved Markdown is missing or changed: {relative}')
    return len(rows)


def promote(staging, destination):
    if staging.exists():
        if destination.exists():
            backup = destination.with_name(f'papers2-transfer-partial-{time.time_ns()}')
            destination.rename(backup)
        staging.rename(destination)
    if not destination.is_dir():
        raise ValueError('Neither staging nor destination PDF directory exists.')


def watch(config, update, poll_seconds=10):
    data_dir = Path(config['data_dir'])
    staging = Path(config['staging'])
    while True:
        pids = restore_processes(staging)
        seed_ready = Path(config['seed_ready']).is_file()
        if not pids and seed_ready:
            break
        update('waiting_for_restore' if pids else 'waiting_for_checkpoint', restore_pids=pids)
        time.sleep(poll_seconds)
    entries = json.loads(Path(config['source_manifest']).read_text())
    if not entries:
        raise ValueError('Empty source inventory; refusing to start conversion.')
    state = subprocess.run(['docker', 'container', 'ls', '-a', '--filter', f'name=^/{config["container"]}$',
                            '--format', '{{.State}}'], check=True, capture_output=True, text=True).stdout.strip()
    if state == 'running':
        update('conversion_running')
        return
    if state and state not in ('exited', 'created', 'dead'):
        raise ValueError(f'Unexpected container state: {state}')
    update('verifying_restored_pdfs', expected_pdfs=len(entries))
    verify_sources(staging if staging.exists() else data_dir/'papers2', entries)
    if restore_processes(staging):
        raise ValueError('A new restore started during verification; retry after it finishes.')
    preserved = verify_checkpoint(data_dir)
    update('promoting_restored_directory', preserved_markdown=preserved)
    promote(staging, data_dir/'papers2')
    if state:
        subprocess.run(['docker', 'rm', config['container']], check=True)
    update('starting_conversion', workers=config['workers'], preserved_markdown=preserved)
    result = subprocess.run(config['docker_command'], check=True, capture_output=True, text=True)
    container_id = result.stdout.strip()
    time.sleep(3)
    detail = json.loads(subprocess.run(['docker', 'inspect', '--format', '{{json .State}}', config['container']],
                                      check=True, capture_output=True, text=True).stdout)
    if not detail['Running'] and detail.get('ExitCode') != 0:
        raise ValueError(f'Converter failed to start: {detail}')
    update('conversion_started' if detail['Running'] else 'conversion_finished',
           container_id=container_id, workers=config['workers'], preserved_markdown=preserved,
           verified_pdfs=len(entries))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    data_dir = Path(config['data_dir'])
    state_path = data_dir/'papers2_restore_watch.json'
    def update(state, **extra):
        value = {'state': state, 'updated_at': time.time(), **extra}
        temp = state_path.with_suffix('.tmp')
        temp.write_text(json.dumps(value, indent=2)+'\n'); temp.replace(state_path)
        print(json.dumps(value), flush=True)
    with (data_dir/'papers2_restore_watch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            watch(config, update)
        except Exception as error:
            update('error_will_retry', error=str(error))
            raise


if __name__ == '__main__':
    main()
