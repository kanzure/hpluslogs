import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import click
from click.testing import CliRunner

from hpluslogs.papers_query_cli import register
from hpluslogs.services import paper_results as pr
from hpluslogs.services.papers_remote import Remote


class PaperResultsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.result = {'question': 'Genetic barcoding?', 'answer': 'Barcodes identify lineages [1].',
            'passages': [{'content': 'A uniquely identifiable genetic lineage.', 'metadata': {
                'pdf_path': 'bio/Genetic%20barcodes.pdf',
                'source_url': 'https://example.org/papers2/bio/Genetic%20barcodes.pdf'}}]}
        @click.group()
        def cli():
            pass
        register(cli)
        self.cli = cli

    @unittest.skipUnless(shutil.which('pandoc'), 'Pandoc not installed')
    def test_real_pandoc_outputs_answer_context_links_and_styles(self):
        with patch.object(pr.scp, 'upload_file') as upload:
            report = pr.publish_result(self.root, self.result, 'barcodes', upload=False)
        upload.assert_not_called()
        self.assertEqual(len(report['files']), 5)
        self.assertTrue(all(Path(p).is_file() for p in report['files']))
        html = (self.root/'outputs/barcodes.html').read_text()
        self.assertIn('href="https://example.org/papers2/bio/Genetic%20barcodes.pdf"', html)
        self.assertIn('href="barcodes.css"', html)
        self.assertIn('Genetic barcodes.pdf', html)
        self.assertIn('uniquely identifiable', (self.root/'outputs/barcodes.context.html').read_text())
        self.assertEqual(json.loads(Path(report['result_json']).read_text()), self.result)

    @unittest.skipUnless(shutil.which('pandoc'), 'Pandoc not installed')
    def test_upload_uses_papers_directory_and_includes_both_formats_and_css(self):
        with patch.object(pr.scp, 'ensure_remote_directory', return_value=True) as mkdir, \
             patch.object(pr.scp, 'upload_file', return_value=True) as upload:
            report = pr.publish_result(self.root, self.result, 'barcodes')
        mkdir.assert_called_once_with('bryan', 'gnusha.org', pr.REMOTE_PATH)
        self.assertEqual({c.args[-1] for c in upload.call_args_list}, {
            'barcodes.css', 'barcodes.context.md', 'barcodes.context.html', 'barcodes.md', 'barcodes.html'})
        self.assertEqual(report['uploaded_to'], 'bryan@gnusha.org:'+pr.REMOTE_PATH)

    @unittest.skipUnless(shutil.which('pandoc'), 'Pandoc not installed')
    def test_render_failure_cannot_upload_stale_html_and_keeps_result(self):
        (self.root/'outputs').mkdir()
        (self.root/'outputs/barcodes.html').write_text('stale HTML')
        with patch.object(pr.publishing.pandoc, 'generate_html', return_value=False), \
             patch.object(pr.scp, 'ensure_remote_directory') as upload:
            with self.assertRaisesRegex(click.ClickException, 'Pandoc failed'):
                pr.publish_result(self.root, self.result, 'barcodes')
        upload.assert_not_called()
        self.assertEqual(pr.read_result(self.root/'papers2_local_queries/barcodes.json'), self.result)

    @unittest.skipUnless(shutil.which('pandoc'), 'Pandoc not installed')
    def test_failed_upload_can_be_retried_from_saved_json_without_query(self):
        with patch.object(pr.scp, 'ensure_remote_directory', return_value=True), \
             patch.object(pr.scp, 'upload_file', return_value=False):
            with self.assertRaisesRegex(click.ClickException, 'rerun papers-publish'):
                pr.publish_result(self.root, self.result, 'barcodes')
        saved = self.root/'papers2_local_queries/barcodes.json'
        with patch.object(Remote, 'command') as remote:
            result = CliRunner().invoke(self.cli, ['papers-publish', str(saved), '--no-upload'],
                                        obj={'data_dir': self.root})
        self.assertEqual(result.exit_code, 0, result.output)
        remote.assert_not_called()

    @unittest.skipUnless(shutil.which('pandoc'), 'Pandoc not installed')
    def test_remote_query_preserves_question_and_renders_locally(self):
        question = "barcodes; $(do-not-execute) 'quotes'"
        with patch.object(Remote, 'command', return_value=SimpleNamespace(stdout=json.dumps(self.result))) as remote:
            result = CliRunner().invoke(self.cli, ['papers-remote-query', '--host', 'worker.local',
                '--user', 'user', '--path', '/srv/papers', '--model', 'local-model', '--top-k', '100',
                '--output-name', 'barcodes', '--no-upload', question], obj={'data_dir':self.root})
        self.assertEqual(result.exit_code, 0, result.output)
        args = remote.call_args.args
        self.assertEqual(args[-2:], ('--', question))
        self.assertEqual(args[args.index('--model')+1], 'local-model')
        self.assertEqual(args[args.index('--top-k')+1], '100')
        self.assertTrue((self.root/'outputs/barcodes.html').is_file())

    def test_missing_pandoc_fails_before_remote_model_call(self):
        with patch.object(pr.shutil, 'which', return_value=None), patch.object(Remote, 'command') as remote:
            result = CliRunner().invoke(self.cli, ['papers-remote-query', '--host', 'worker.local',
                '--user', 'user', '--path', '/srv/papers', '--nollm', 'barcodes'], obj={'data_dir':self.root})
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('Install pandoc', result.output)
        remote.assert_not_called()

    @unittest.skipUnless(shutil.which('pandoc'), 'Pandoc not installed')
    def test_context_only_and_empty_results_render(self):
        self.result.pop('answer')
        report = pr.publish_result(self.root, self.result, 'context', upload=False)
        self.assertIn('uniquely identifiable', (self.root/'outputs/context.md').read_text())
        self.result['passages'] = []
        pr.publish_result(self.root, self.result, 'empty', upload=False)
        self.assertIn('No matching passages', (self.root/'outputs/empty.html').read_text())

    def test_unsafe_output_and_remote_paths_are_rejected(self):
        for name in ('../escape', '.hidden', 'a;touch other', 'name/path'):
            with self.assertRaises(click.BadParameter):
                pr.output_name(name)
        with patch.object(pr.shutil, 'which', return_value='/usr/bin/pandoc'):
            with self.assertRaises(click.BadParameter):
                pr.preflight('wrap.css', 'bryan', 'gnusha.org', '~/papers; touch /tmp/bad', True)

    def test_invalid_result_is_rejected(self):
        with self.assertRaises(click.ClickException):
            pr.validate_result({'question':'q', 'passages':[{'content':'x','metadata':{}}]})


if __name__ == '__main__':
    unittest.main()
