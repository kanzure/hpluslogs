"""Offline regression tests; no API calls, credentials, or paid storage needed."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, patch

import click
from click.testing import CliRunner
from xai_sdk.proto import collections_pb2 as pb, documents_pb2, files_pb2

from hpluslogs.services import papers, papers_markdown
from hpluslogs.papers_cli import register


class PapersTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name)
        self.root = self.data / 'papers2'
        self.root.mkdir()
        self.client = MagicMock()
        self.client.files.upload.side_effect = [NS(id='file-1'), NS(id='file-2')]
        self.client.collections.get_document.return_value = pb.DocumentMetadata(status=pb.DOCUMENT_STATUS_PROCESSED)

    def pdf(self, relative='a.pdf', content=b'%PDF-1.7\nfirst'):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def ingest(self, path, **kw):
        relative = path.relative_to(self.root).as_posix()
        md = self.data / 'test-markdown' / (relative + '.md')
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_bytes(path.read_bytes())
        return papers._ingest_one(self.data, md, self.client, 'collection-test', papers.BASE_URL,
                                  kw.get('wait', False), kw.get('timeout', 1), kw.get('retry_failed', False), relative)

    def row(self, relative='a.pdf'):
        db = papers.connect(self.data)
        try:
            return dict(db.execute('SELECT * FROM documents WHERE path=?', (relative,)).fetchone())
        finally:
            db.close()

    def test_duplicate_basenames_and_resume(self):
        self.assertEqual(self.ingest(self.pdf('a/paper.pdf'))[1], 'ready')
        self.assertEqual(self.ingest(self.pdf('b/paper.pdf'))[1], 'ready')
        self.assertEqual(self.ingest(self.root/'a/paper.pdf')[1], 'unchanged')
        self.assertEqual(self.client.files.upload.call_count, 2)
        names = [c.kwargs['filename'] for c in self.client.files.upload.call_args_list]
        self.assertNotEqual(*names)
        self.assertEqual(self.row('b/paper.pdf')['file_id'], 'file-2')

    def test_changed_pdf_updates_existing_file(self):
        path = self.pdf()
        self.ingest(path)
        path.write_bytes(b'%PDF-1.7\nchanged')
        self.ingest(path)
        self.assertEqual(self.client.files.upload.call_count, 1)
        self.client.collections.update_document.assert_called_once()
        self.assertEqual(self.client.collections.update_document.call_args.kwargs['file_id'], 'file-1')
        self.assertEqual(self.row()['sha256'], papers.sha256(path))

    def test_failed_update_is_retried_without_new_file(self):
        path = self.pdf()
        self.ingest(path)
        path.write_bytes(b'%PDF-1.7\nchanged')
        self.client.collections.update_document.side_effect = RuntimeError('network')
        self.assertTrue(self.ingest(path)[1].startswith('error:'))
        self.assertEqual(self.row()['phase'], 'updating')
        self.client.collections.update_document.side_effect = None
        self.assertEqual(self.ingest(path)[1], 'ready')
        self.assertEqual(self.client.collections.update_document.call_count, 2)
        self.assertEqual(self.client.files.upload.call_count, 1)

    def test_invalid_replacement_does_not_mark_old_index_current(self):
        path = self.pdf()
        self.ingest(path)
        old_hash = self.row()['sha256']
        path.write_bytes(b'\xffinvalid UTF-8')
        self.assertTrue(self.ingest(path)[1].startswith('error:'))
        self.assertEqual(self.row()['sha256'], old_hash)
        path.write_bytes(b'%PDF-1.7\nreplacement')
        self.ingest(path)
        self.client.collections.update_document.assert_called_once()

    def test_attachment_failure_keeps_file_id_for_resume(self):
        path = self.pdf()
        self.client.collections.add_existing_document.side_effect = RuntimeError('network')
        self.ingest(path)
        self.assertEqual(self.row()['phase'], 'uploaded')
        self.assertEqual(self.row()['file_id'], 'file-1')
        self.client.collections.add_existing_document.side_effect = None
        self.assertEqual(self.ingest(path)[1], 'ready')
        self.assertEqual(self.client.files.upload.call_count, 1)

    def test_unknown_upload_outcome_is_not_blindly_retried(self):
        path = self.pdf()
        self.client.files.upload.side_effect = RuntimeError('connection lost')
        self.ingest(path)
        self.assertIn('uncertain', self.ingest(path)[1])
        self.assertEqual(self.client.files.upload.call_count, 1)

    def test_reconcile_recovers_id_without_uploading(self):
        path = self.pdf()
        self.client.files.upload.side_effect = RuntimeError('connection lost')
        self.ingest(path)
        name = self.client.files.upload.call_args.kwargs['filename']
        self.client.__enter__.return_value = self.client
        self.client.files.list.side_effect = [files_pb2.ListFilesResponse(pagination_token='next'),
            files_pb2.ListFilesResponse(data=[files_pb2.File(id='recovered', filename=name)])]
        with patch.object(papers.xai, 'get_sync_client', return_value=self.client):
            self.assertEqual(papers.reconcile(self.data), {'recovered': 1, 'unresolved': []})
        self.assertEqual(self.row()['file_id'], 'recovered')
        self.assertEqual(self.ingest(path)[1], 'ready')
        self.assertEqual(self.client.files.upload.call_count, 1)

    def test_legacy_filename_roundtrips_in_manifest_and_url(self):
        relative = os.fsdecode(b'legacy-\xf4.pdf')
        self.assertEqual(self.ingest(self.pdf(relative))[1], 'ready')
        key = papers.path_key(relative)
        self.assertEqual(key, 'legacy-%F4.pdf')
        self.assertEqual(self.row(key)['phase'], 'ready')
        fields = self.client.collections.add_existing_document.call_args.kwargs['fields']
        self.assertEqual(fields['source_url'], papers.BASE_URL + 'legacy-%F4.pdf')

    def test_processing_then_ready_without_reupload(self):
        path = self.pdf()
        self.client.collections.get_document.return_value = pb.DocumentMetadata(status=pb.DOCUMENT_STATUS_PROCESSING)
        self.assertEqual(self.ingest(path)[1], 'pending')
        self.client.collections.get_document.return_value = pb.DocumentMetadata(status=pb.DOCUMENT_STATUS_PROCESSED)
        self.assertEqual(self.ingest(path)[1], 'ready')
        self.assertEqual(self.client.files.upload.call_count, 1)

    def test_failed_index_can_be_retried(self):
        path = self.pdf()
        self.client.collections.get_document.return_value = pb.DocumentMetadata(status=pb.DOCUMENT_STATUS_FAILED, error_message='conversion')
        self.ingest(path)
        self.assertEqual(self.row()['phase'], 'failed')
        self.client.collections.get_document.return_value = pb.DocumentMetadata(status=pb.DOCUMENT_STATUS_PROCESSED)
        self.ingest(path, retry_failed=True)
        self.client.collections.reindex_document.assert_called_once()
        self.assertEqual(self.client.files.upload.call_count, 1)

    def test_budget_blocks_before_client_creation(self):
        path = self.pdf()
        def extract(command, **kwargs):
            Path(command[3]).write_text('# Paper\nScientific content')
            return NS(returncode=0)
        with patch.object(papers_markdown.subprocess, 'run', side_effect=extract):
            papers_markdown.convert_one(self.data, path, 'test', 10)
        with patch.object(papers.xai, 'get_sync_client') as client:
            with self.assertRaisesRegex(click.ClickException, 'monthly cost'):
                papers.upload(self.data, approve_upload=True, monthly_budget=1e-12)
            client.assert_not_called()
        self.assertAlmostEqual(papers.estimate_cost(20 * 2**30)['estimated_usd_excluding_tokens_downloads'], 75)

    def test_upload_requires_explicit_approval_even_below_budget(self):
        self.pdf()
        with patch.object(papers.xai, 'get_sync_client') as client:
            with self.assertRaisesRegex(click.ClickException, 'Upload held'):
                papers.upload(self.data)
            client.assert_not_called()

    def test_raw_pdf_is_never_uploaded(self):
        path = self.pdf()
        result = papers._ingest_one(self.data, path, self.client, 'c', papers.BASE_URL, False, 10, False, 'a.pdf')
        self.assertIn('Only prepared Markdown', result[1])
        self.client.files.upload.assert_not_called()

    def test_dry_run_has_no_api_calls(self):
        self.pdf()
        with patch.object(papers.xai, 'get_sync_client') as client:
            papers.upload(self.data, dry_run=True)
            client.assert_not_called()

    def test_symlinks_and_partial_downloads_are_excluded(self):
        path = self.pdf('real.PDF')
        self.pdf('.rsync-partial/partial.pdf')
        (self.root / 'alias.pdf').symlink_to(path)
        (self.root / 'loop').symlink_to(self.root)
        self.assertEqual(list(papers.pdf_files(self.root)), [path])

    def test_add_rejects_html_traversal_and_overwrite(self):
        source = self.data / 'source.pdf'
        source.write_bytes(b'%PDF-1.7\npaper')
        papers.add(self.data, str(source), 'bio/paper.pdf')
        with self.assertRaises(click.ClickException):
            papers.add(self.data, str(source), 'bio/paper.pdf')
        with self.assertRaises(click.BadParameter):
            papers.add(self.data, str(source), '../paper.pdf')
        source.write_bytes(b'<html>error</html>')
        with self.assertRaisesRegex(click.ClickException, 'not a PDF'):
            papers.add(self.data, str(source), 'other.pdf')
        self.assertFalse((self.root / 'other.pdf').exists())

    def test_restore_command_is_subtree_only(self):
        with patch.object(papers.subprocess, 'run') as run:
            papers.restore(self.data, 'abc123', papers.SNAPSHOT_PATH, 'sftp:user@backup-host:/backups/papers', None, True)
        cmd = run.call_args.args[0]
        self.assertIn('abc123:/mnt/diyhplus/public_html/papers2', cmd)
        self.assertIn('--dry-run', cmd)
        self.assertIn('--verify', cmd)
        self.assertEqual(cmd[cmd.index('--target') + 1], str(self.root))
        with self.assertRaises(click.BadParameter):
            papers.restore(self.data, 'latest', '/mnt/diyhplus', 'sftp:user@backup-host:/backups/papers', None, True)

    def test_search_joins_metadata_and_preserves_source_identity(self):
        self.client.__enter__.return_value = self.client
        self.client.collections.search.return_value = documents_pb2.SearchResponse(matches=[
            documents_pb2.SearchMatch(file_id='file-1', chunk_id='chunk-1', chunk_content='Evidence', score=.8),
            documents_pb2.SearchMatch(file_id='file-1', chunk_id='chunk-2', chunk_content='More evidence', score=.7)])
        self.client.collections.get_document.return_value = pb.DocumentMetadata(fields={
            'relative_path': 'bio/a paper.pdf', 'source_url': papers.BASE_URL + 'bio/a%20paper.pdf'})
        with patch.object(papers.xai, 'get_sync_client', return_value=self.client):
            result = papers.retrieve(self.data, 'question', collection_id='c')
        self.assertEqual(len(result), 2)
        self.client.collections.get_document.assert_called_once()
        self.assertIn('[1] bio/a paper.pdf', papers.format_context(result))
        self.assertIn('a%20paper.pdf', papers.format_context(result))
        json.dumps(result)

    def test_cli_nollm_saves_results_without_generation(self):
        @click.group()
        @click.pass_context
        def cli(ctx):
            ctx.obj = {'data_dir': self.data}
        register(cli)
        result = {'content': 'Evidence', 'score': .8, 'file_id': 'f', 'chunk_id': 'c',
                  'metadata': {'source_url': papers.BASE_URL+'a.pdf', 'relative_path': 'a.pdf'}}
        with patch.object(papers, 'retrieve', return_value=[result]), \
             patch('hpluslogs.services.generation.generate_answer') as answer:
            output = CliRunner().invoke(cli, ['papers-query', '--collection-id', 'test', '--nollm', 'question'])
            self.assertEqual(output.exit_code, 0, output.output)
            self.assertIn('Evidence', output.output)
            answer.assert_not_called()
        self.assertEqual(len(list((self.data/'papers2_queries').glob('*.json'))), 1)


if __name__ == '__main__':
    unittest.main()
