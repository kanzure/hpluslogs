import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import click
from hpluslogs.services import papers_markdown as md, papers_progress, papers_remote


class RemoteConversionTest(unittest.TestCase):
    def test_deploy_cli_forwards_deferred_failure_policy(self):
        from click.testing import CliRunner
        from hpluslogs.papers_remote_cli import register
        @click.group()
        def cli():
            pass
        register(cli)
        with patch.object(papers_remote, 'deploy') as deploy:
            result = CliRunner().invoke(cli, ['papers-remote-deploy', '--host', 'server.local',
                '--user', 'user', '--path', '/srv/papers', '--skip-failed'], obj={'data_dir':self.data})
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(deploy.call_args.kwargs['skip_failed'])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name)
        (self.data / 'papers2').mkdir()

    def test_checkpoint_includes_wal_and_resumes_in_new_directory(self):
        source = self.data / 'papers2/a.pdf'; source.write_bytes(b'pdf')
        output = md.output_path('a.pdf')
        target = self.data / 'papers2_markdown' / output
        target.parent.mkdir(parents=True); target.write_text('markdown')
        from hpluslogs.services import papers
        db = md.connect(self.data)
        db.execute('INSERT INTO conversions VALUES (?,?,?,?,?,?,?,?,?,?)',
                   ('a.pdf', papers.sha256(source), 3, output, papers.sha256(target), 8, 'recipe', 'ready', None, 100))
        db.commit()  # Keep the WAL connection alive while snapshotting.
        backup = self.data / 'backup.sqlite3'
        papers_remote.snapshot(self.data / 'papers2_markdown.sqlite3', backup)
        with sqlite3.connect(backup) as copied:
            self.assertEqual(copied.execute('select count(*) from conversions').fetchone()[0], 1)
        db.close()
        import shutil
        moved = self.data / 'moved'; moved.mkdir()
        shutil.copytree(self.data / 'papers2', moved / 'papers2')
        shutil.copytree(self.data / 'papers2_markdown', moved / 'papers2_markdown')
        shutil.copy(backup, moved / 'papers2_markdown.sqlite3')
        with patch.object(md.subprocess, 'run') as convert:
            self.assertEqual(md.convert_one(moved, moved / 'papers2/a.pdf', 'recipe', 10)[1], 'unchanged')
            convert.assert_not_called()

    def test_progress_rates_exclude_old_host_and_project_cost(self):
        for name in ('a', 'b', 'c', 'd'):
            (self.data / f'papers2/{name}.pdf').write_bytes(b'1234')
        root = self.data / 'papers2_markdown'; root.mkdir()
        (root / 'a.md').write_text('abcdefgh')
        (root / 'b.md').write_text('abcdefghijkl')
        db = md.connect(self.data)
        with db:
            for name, size, status, updated in [('a', 8, 'ready', 100), ('b', 12, 'ready', 1950), ('c', None, 'failed', 1980)]:
                db.execute('INSERT INTO conversions VALUES (?,?,?,?,?,?,?,?,?,?)',
                           (name+'.pdf', 'hash', 4, name+'.md', 'hash', size, 'r', status, None, updated))
        db.close()
        (self.data / 'papers2_conversion_run.json').write_text(json.dumps({'started_at': 1900, 'state':'running'}))
        result = papers_progress.report(self.data, now=2000, searches=1000)
        self.assertEqual(result['ready'], 2)
        self.assertEqual(result['markdown_bytes'], 20)
        self.assertEqual(result['first_pass_remaining'], 1)
        self.assertEqual(result['throughput'][0]['attempts'], 2)
        self.assertEqual(result['throughput'][0]['papers_per_hour'], 72)
        cost = result['full_archive_projections']['per_paper']
        self.assertAlmostEqual(cost['file_gib'], 40 / 2**30)
        self.assertAlmostEqual(cost['estimated_usd_excluding_tokens_downloads'], 2.5 + 40/2**30*3.75)
        (root / 'a.md').unlink()
        self.assertEqual(papers_progress.report(self.data, now=2000)['ready'], 1)

    def test_empty_progress_is_valid(self):
        result = papers_progress.report(self.data)
        self.assertEqual(result['ready'], 0)
        self.assertEqual(result['full_archive_projections'], {})
        self.assertIsNone(result['throughput'][0]['first_pass_remaining_hours'])

    def test_build_context_is_explicit_allowlist(self):
        context = self.data / 'context'; context.mkdir()
        papers_remote.build_context(context)
        files = [str(p.relative_to(context)) for p in context.rglob('*') if p.is_file()]
        self.assertEqual(len(files), 11)
        self.assertIn('hpluslogs/conversion_cli.py', files)
        self.assertFalse(any('.env' in p or '/data/' in p or p.endswith('.pdf') for p in files))

    def test_remote_parameters_are_quoted_and_option_injection_rejected(self):
        for host, user, path in [('-oProxyCommand=x', 'user', '/srv/papers'),
                                 ('host', '-user', '/srv/papers'), ('host','user','/'),
                                 ('host','user','/srv/../etc'), ('host','user','/srv/a,b')]:
            with self.assertRaises(click.BadParameter):
                papers_remote.Remote(host, user, path)
        remote = papers_remote.Remote('server.local', 'user', "/srv/paper's archive")
        with patch.object(papers_remote.subprocess, 'run') as run:
            remote.command('printf', '%s', '$(false)')
            command = run.call_args.args[0]
            self.assertEqual(command[-1], "printf %s '$(false)'")
        with patch.object(remote, 'state', return_value='running'):
            with self.assertRaises(click.ClickException): remote.require_stopped()

    def test_runtime_has_no_network_and_readonly_sources(self):
        from types import SimpleNamespace
        remote = papers_remote.Remote('server.local', 'user', '/srv/papers')
        with patch.object(remote, 'command', return_value=SimpleNamespace(stdout='1000\n')):
            args = remote.runtime_args()
        self.assertEqual(args[args.index('--network') + 1], 'none')
        self.assertIn('--read-only', args)
        self.assertIn('type=bind,src=/srv/papers/data/papers2,dst=/data/papers2,readonly', args)

    def test_tar_stream_preserves_legacy_names_and_skips_unchanged(self):
        import os
        import subprocess
        from types import SimpleNamespace
        from hpluslogs.services import papers
        source = self.data / 'papers2'
        names = ['space paper.pdf', '-leading.pdf', 'line\nbreak.pdf', 'legacy-\udcf4.PDF']
        files = []
        for name in names:
            p = source / name; p.write_bytes(b'%PDF-test'); files.append(p)
        destination = self.data / 'remote'; (destination / 'data/papers2').mkdir(parents=True)
        remote = papers_remote.Remote('test.local', 'user', str(destination))
        real_popen = subprocess.Popen
        def local_ssh(command, **kwargs):
            if command[0] == 'ssh':
                command = ['bash', '-c', command[-1]]
            return real_popen(command, **kwargs)
        with patch.object(remote, 'command', return_value=SimpleNamespace(stdout='{}')), \
             patch.object(papers_remote.subprocess, 'Popen', side_effect=local_ssh):
            papers_remote.transfer_pdfs_tar(remote, source, files, self.data / 'list')
        for name in names:
            copied = destination / 'data/papers2' / name
            self.assertEqual(copied.read_bytes(), b'%PDF-test')
            self.assertEqual(copied.stat().st_mtime_ns, (source/name).stat().st_mtime_ns)
        inventory = {papers.path_key(p.name): [p.stat().st_size, p.stat().st_mtime_ns] for p in files}
        with patch.object(remote, 'command', return_value=SimpleNamespace(stdout=json.dumps(inventory))), \
             patch.object(papers_remote.subprocess, 'Popen') as process:
            papers_remote.transfer_pdfs_tar(remote, source, files, self.data / 'list')
            process.assert_not_called()

    def test_tar_extraction_failure_is_reported(self):
        import subprocess
        from types import SimpleNamespace
        source = self.data / 'papers2'
        pdf = source / 'paper.pdf'; pdf.write_bytes(b'%PDF-test')
        remote = papers_remote.Remote('test.local', 'user', str(self.data / 'remote'))
        real_popen = subprocess.Popen
        def failed_ssh(command, **kwargs):
            if command[0] == 'ssh':
                command = ['bash', '-c', 'cat >/dev/null; exit 23']
            return real_popen(command, **kwargs)
        with patch.object(remote, 'command', return_value=SimpleNamespace(stdout='{}')), \
             patch.object(papers_remote.subprocess, 'Popen', side_effect=failed_ssh):
            with self.assertRaises(subprocess.CalledProcessError) as error:
                papers_remote.transfer_pdfs_tar(remote, source, [pdf], self.data / 'list')
            self.assertEqual(error.exception.returncode, 23)
