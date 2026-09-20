"""Parameterised SSH/Docker deployment for conversion only; no API credentials."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import sqlite3
import subprocess
import tempfile
import time

import click
from hpluslogs.services import papers


class Remote:
    def __init__(self, host, user, path):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]*', host):
            raise click.BadParameter('Use an SSH hostname or IPv4 address.')
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_-]*', user):
            raise click.BadParameter('Invalid SSH username.')
        p = PurePosixPath(path)
        if not p.is_absolute() or len(p.parts) < 3 or '..' in p.parts or any(c in path for c in '\n\r,:'):
            raise click.BadParameter('Use a dedicated absolute remote directory (no .., commas, colons or newlines).')
        self.target = f'{user}@{host}'
        self.path = str(p)
        self.name = 'hpluslogs-papers-' + hashlib.sha256(self.path.encode()).hexdigest()[:12]
        self.image = self.name + ':latest'

    def shell(self, script, capture=False):
        return subprocess.run(['ssh', '-o', 'BatchMode=yes', self.target, script],
                              check=True, text=True, capture_output=capture)

    def command(self, *args, capture=False):
        return self.shell(shlex.join(str(a) for a in args), capture=capture)

    def state(self):
        result = self.command('docker', 'container', 'ls', '-a', '--filter', f'name=^/{self.name}$',
                              '--format', '{{.State}}', capture=True)
        return result.stdout.strip()

    def require_stopped(self):
        state = self.state()
        if state and state not in ('exited', 'created', 'dead'):
            raise click.ClickException(f'Conversion container is {state}; use papers-remote-stop before deploy/pull.')

    def rsync(self, source, destination, *options):
        subprocess.run(['rsync', '-a', '--protect-args', '--no-owner', '--no-group',
                        '--stats', '-e', 'ssh -o BatchMode=yes', *options, source, destination], check=True)

    def location(self, relative):
        return f'{self.target}:{self.path}/{relative}'

    def mount_args(self):
        return ['--mount', f'type=bind,src={self.path}/data,dst=/data',
                '--mount', f'type=bind,src={self.path}/data/papers2,dst=/data/papers2,readonly']

    def runtime_args(self):
        uid = self.command('id', '-u', capture=True).stdout.strip()
        gid = self.command('id', '-g', capture=True).stdout.strip()
        return ['--network', 'none', '--user', f'{uid}:{gid}', '--read-only',
                '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
                '--tmpfs', '/tmp:rw,nosuid,nodev,size=1g', *self.mount_args()]


def snapshot(source, destination):
    """SQLite backup includes committed WAL entries; never copy a live DB alone."""
    with sqlite3.connect(str(source)) as src, sqlite3.connect(str(destination)) as dst:
        src.backup(dst)


def build_context(destination):
    project = Path(__file__).resolve().parents[1]
    for relative in ['conversion_cli.py', 'services/papers.py', 'services/papers_markdown.py',
                     'services/papers_progress.py', 'services/paper_conversion_worker.py', 'integrations/xai.py']:
        target = destination / 'hpluslogs' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / relative, target)
    for name in ('Dockerfile', 'requirements.txt', '.dockerignore'):
        shutil.copy2(project / 'docker/papers' / name, destination / name)


def transfer_pdfs_tar(remote, source_root, sources, source_list):
    """Stream changed PDFs over SSH; preserve nanosecond mtimes for future resumes."""
    import os
    code = ("import os,json,sys; from pathlib import Path; from urllib.parse import quote_from_bytes; "
            "r=Path(sys.argv[1]); "
            "print(json.dumps({quote_from_bytes(os.fsencode(p.relative_to(r).as_posix()),safe='/'): "
            "[p.stat().st_size,p.stat().st_mtime_ns] for p in r.rglob('*') "
            "if p.is_file() and not p.is_symlink() and p.suffix.lower()=='.pdf'}))")
    existing = json.loads(remote.command('python3', '-c', code, remote.path + '/data/papers2', capture=True).stdout)
    changed = [p for p in sources if existing.get(papers.path_key(p.relative_to(source_root).as_posix()))
               != [p.stat().st_size, p.stat().st_mtime_ns]]
    source_list.write_bytes(b''.join(os.fsencode(p.relative_to(source_root)) + b'\0' for p in changed))
    click.echo(f'Tar transfer: {len(changed)} PDFs, {sum(p.stat().st_size for p in changed) / 2**30:.2f} GiB; '
               f'{len(sources) - len(changed)} unchanged files skipped.')
    if not changed:
        return
    commands = [
        ['tar', '--format=pax', '-C', str(source_root), '--null', '--verbatim-files-from',
         '-T', str(source_list), '-cf', '-'],
        ['zstd', '-T4', '-3', '-c'],
        ['ssh', '-o', 'BatchMode=yes', remote.target,
         shlex.join(['tar', '--zstd', '-xf', '-', '--no-same-owner', '-C', remote.path + '/data/papers2'])],
    ]
    processes = []
    try:
        for i, command in enumerate(commands):
            process = subprocess.Popen(command, stdin=processes[-1].stdout if processes else None,
                                       stdout=subprocess.PIPE if i < len(commands) - 1 else None)
            if processes:
                processes[-1].stdout.close()
            processes.append(process)
        codes = [process.wait() for process in reversed(processes)][::-1]
        for command, code in zip(commands, codes):
            if code:
                raise subprocess.CalledProcessError(code, command)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
    click.echo('Tar transfer complete.')


def deploy(data_dir, remote, workers=48, timeout=600, transfer='tar'):
    # Lock prevents changing the local checkpoint/source while it is transferred.
    with papers.exclusive(data_dir), tempfile.TemporaryDirectory(prefix='papers-deploy-') as tmp:
        remote.require_stopped()
        q = shlex.quote
        path = q(remote.path)
        marker = q(remote.path + '/.hpluslogs-papers-conversion')
        remote.shell(f'set -eu; if [ ! -f {marker} ]; then '
                     f'if [ -d {path} ] && [ -n "$(ls -A {path})" ]; then '
                     'echo "Refusing nonempty unmarked deployment directory" >&2; exit 1; fi; '
                     f'mkdir -p {path}; touch {marker}; fi; '
                     f'mkdir -p {q(remote.path + "/data/papers2")} {q(remote.path + "/build")}')
        tmp = Path(tmp)
        context = tmp / 'build'; context.mkdir()
        build_context(context)
        remote.rsync(str(context) + '/', remote.location('build/'), '--delete')
        remote.command('docker', 'build', '-t', remote.image, remote.path + '/build')
        # Only eligible PDFs, including legacy non-UTF8 names. No credentials or unrelated data.
        source_root = data_dir / 'papers2'
        source_list = tmp / 'pdfs.list'
        sources = list(papers.pdf_files(source_root))
        import os
        source_list.write_bytes(b''.join(os.fsencode(p.relative_to(source_root)) + b'\0'
                                        for p in sources))
        if transfer == 'tar':
            transfer_pdfs_tar(remote, source_root, sources, source_list)
        else:
            remote.rsync(str(source_root) + '/', remote.location('data/papers2/'),
                         '--from0', '--files-from=' + str(source_list), '--partial')
        output = data_dir / 'papers2_markdown'
        if output.exists():
            remote.rsync(str(output) + '/', remote.location('data/papers2_markdown/'),
                         '--exclude=.*', '--ignore-existing')
        manifest = data_dir / 'papers2_markdown.sqlite3'
        if manifest.exists():
            # Remote manifest stays authoritative on later deployments.
            backups = data_dir / 'papers2_checkpoints'; backups.mkdir(exist_ok=True)
            checkpoint = backups / f'conversion-{time.time_ns()}.sqlite3'
            snapshot(manifest, checkpoint)
            remote.rsync(str(checkpoint), remote.location('data/conversion-seed.sqlite3'))
            remote.shell(f'set -eu; if [ ! -f {q(remote.path + "/data/papers2_markdown.sqlite3")} ]; then '
                         f'mv {q(remote.path + "/data/conversion-seed.sqlite3")} '
                         f'{q(remote.path + "/data/papers2_markdown.sqlite3")}; fi')
        if remote.state():
            remote.command('docker', 'rm', remote.name)
        remote.command('docker', 'run', '-d', '--init', '--name', remote.name,
                       '--label', 'hpluslogs.role=paper-conversion', '--restart', 'no',
                       '--stop-timeout', str(timeout + 30), '--cpus', str(workers),
                       *remote.runtime_args(), remote.image, 'convert', '--workers', str(workers),
                       '--timeout', str(timeout))
        click.echo(f'Conversion started: {remote.target}:{remote.path} ({workers} workers).')


def progress(remote, days=30, searches=0, index_multiplier=1.0):
    state = remote.state()
    click.echo(f'Container state: {state or "absent"}', err=True)
    remote.command('docker', 'run', '--rm', *remote.runtime_args(), remote.image,
                   'progress', '--days', str(days), '--searches', str(searches),
                   '--index-multiplier', str(index_multiplier))


def stop(remote):
    remote.command('docker', 'stop', remote.name)
    click.echo('Stopped; completed Markdown and the conversion manifest remain in the remote data directory.')


def pull(data_dir, remote):
    remote.require_stopped()
    with papers.exclusive(data_dir), tempfile.TemporaryDirectory(prefix='papers-pull-', dir=data_dir) as tmp:
        tmp = Path(tmp)
        code = ('import sqlite3,sys; '
                's=sqlite3.connect(sys.argv[1]); d=sqlite3.connect(sys.argv[2]); '
                's.backup(d); d.close(); s.close()')
        remote.command('python3', '-c', code, remote.path + '/data/papers2_markdown.sqlite3',
                       remote.path + '/data/conversion-export.sqlite3')
        remote.rsync(remote.location('data/conversion-export.sqlite3'), str(tmp / 'manifest.sqlite3'))
        remote.rsync(remote.location('data/papers2_markdown/'), str(data_dir / 'papers2_markdown') + '/', '--exclude=.*')
        target = data_dir / 'papers2_markdown.sqlite3'
        if target.exists():
            backups = data_dir / 'papers2_checkpoints'; backups.mkdir(exist_ok=True)
            snapshot(target, backups / f'before-pull-{time.time_ns()}.sqlite3')
            with sqlite3.connect(str(target)) as db:
                db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        (tmp / 'manifest.sqlite3').replace(target)
        remote.rsync(remote.location('data/papers2_conversion_run.json'), str(data_dir) + '/')
        click.echo('Pulled Markdown and checkpoint. Local PDFs are unchanged; no xAI operations performed.')


def watch_restore(data_dir, remote, staging_name='papers2-restic', workers=48, timeout=600):
    """Arm a durable host-side watcher, then seed checkpoints and PDF checksums."""
    import uuid
    from concurrent.futures import ThreadPoolExecutor
    if Path(staging_name).name != staging_name or staging_name in ('.', '..', 'papers2'):
        raise click.BadParameter('Use a staging directory name distinct from papers2.')
    remote.require_stopped()
    service = remote.name + '-restore-watch'
    active = remote.command('systemctl', '--user', 'show', service+'.service',
                            '--property=ActiveState', '--value', capture=True).stdout.strip()
    if active in ('active', 'activating'):
        raise click.ClickException('The restore watcher is already active; inspect its state/logs.')
    with papers.exclusive(data_dir), tempfile.TemporaryDirectory(prefix='papers-watch-') as tmp:
        tmp = Path(tmp)
        remote.command('mkdir', '-p', remote.path+'/data', remote.path+'/watcher')
        script = Path(__file__).resolve().parents[1]/'scripts/watch_papers_restore.py'
        remote.rsync(str(script), remote.location('watcher/watch_papers_restore.py'))
        ready_path = remote.path+'/data/restore-seed-'+uuid.uuid4().hex+'.ready'
        config = {'data_dir': remote.path+'/data', 'staging': remote.path+'/data/'+staging_name,
                  'seed_ready': ready_path, 'source_manifest': remote.path+'/data/papers2_source_manifest.json',
                  'container': remote.name, 'workers': workers,
                  'docker_command': ['docker', 'run', '-d', '--init', '--name', remote.name,
                                     '--label', 'hpluslogs.role=paper-conversion', '--restart', 'no',
                                     '--stop-timeout', str(timeout+30), '--cpus', str(workers),
                                     *remote.runtime_args(), remote.image, 'convert',
                                     '--workers', str(workers), '--timeout', str(timeout)]}
        (tmp/'restore-watch.json').write_text(json.dumps(config, indent=2))
        remote.rsync(str(tmp/'restore-watch.json'), remote.location('watcher/config.json'))
        # systemd runs on the destination; no local process/SSH connection must survive.
        remote.command('systemd-run', '--user', '--collect', '--unit', service,
                       '--property=Restart=on-failure', '--property=RestartSec=60',
                       '--property=StartLimitIntervalSec=0', '/usr/bin/python3', '-u',
                       remote.path+'/watcher/watch_papers_restore.py', '--config', remote.path+'/watcher/config.json')
        click.echo('Remote watcher armed. Preparing checksum inventory and saved Markdown.', err=True)
        sources = list(papers.pdf_files(data_dir/'papers2'))
        def entry(path):
            return {'path': papers.path_key(path.relative_to(data_dir/'papers2').as_posix()),
                    'size': path.stat().st_size, 'sha256': papers.sha256(path)}
        with ThreadPoolExecutor(max_workers=4) as pool:
            inventory = list(pool.map(entry, sources))
        (tmp/'source-manifest.json').write_text(json.dumps(inventory))
        remote.rsync(str(tmp/'source-manifest.json'), remote.location('data/papers2_source_manifest.json'))
        backups = data_dir/'papers2_checkpoints'; backups.mkdir(exist_ok=True)
        checkpoint = backups/f'watch-restore-{time.time_ns()}.sqlite3'
        snapshot(data_dir/'papers2_markdown.sqlite3', checkpoint)
        remote.rsync(str(data_dir/'papers2_markdown')+'/', remote.location('data/papers2_markdown/'),
                     '-z', '--exclude=.*', '--ignore-existing')
        remote.rsync(str(checkpoint), remote.location('data/conversion-seed.sqlite3'))
        q = shlex.quote
        remote.shell(f'set -eu; if [ ! -f {q(remote.path+"/data/papers2_markdown.sqlite3")} ]; then '
                     f'mv {q(remote.path+"/data/conversion-seed.sqlite3")} '
                     f'{q(remote.path+"/data/papers2_markdown.sqlite3")}; fi; touch {q(ready_path)}')
        click.echo(f'Watcher ready: {service}. It will verify the restored PDFs and start {workers} workers automatically.')


def monitor(remote, interval=300):
    """Record periodic progress on the host, independently of the chat/session."""
    unit = remote.name+'-progress'
    active = remote.command('systemctl', '--user', 'show', unit+'.service',
                            '--property=ActiveState', '--value', capture=True).stdout.strip()
    if active in ('active', 'activating'):
        click.echo(f'Progress reporter already running: {unit}')
        return
    remote.command('mkdir', '-p', remote.path+'/watcher')
    script = Path(__file__).resolve().parents[1]/'scripts/report_papers_progress.py'
    remote.rsync(str(script), remote.location('watcher/report_papers_progress.py'))
    remote.command('systemd-run', '--user', '--collect', '--unit', unit,
                   '--property=Restart=on-failure', '--property=RestartSec=60',
                   '--property=StartLimitIntervalSec=0', '/usr/bin/python3', '-u',
                   remote.path+'/watcher/report_papers_progress.py', '--container', remote.name,
                   '--data-dir', remote.path+'/data', '--interval', str(interval))
    click.echo(f'Recording progress every {interval} seconds until the converter exits: {unit}')
