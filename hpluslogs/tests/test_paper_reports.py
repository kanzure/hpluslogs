import json
from pathlib import Path
import subprocess
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import click
from click.testing import CliRunner

from hpluslogs.papers_chroma_cli import cli
from hpluslogs.papers_query_cli import register
from hpluslogs.services import papers_chroma as pc
from hpluslogs.services.papers_remote import Remote


class PaperReportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.passages = [{'content': 'Observed lineage tracking.', 'metadata': {
            'pdf_path': 'Lineage%20tracing.pdf', 'source_url': 'https://example.org/paper.pdf'}}]

    def response(self, content='## Answer and scope\n\nSupported result [1].', finish='stop'):
        response = Mock()
        response.json.return_value = {'choices': [{'message': {'content': content}, 'finish_reason': finish}]}
        return response

    def test_default_report_contains_multiple_tasks_and_original_evidence(self):
        with patch('requests.post', return_value=self.response()) as post:
            pc.answer('lineage?', self.passages, 'http://localhost/v1', 'model',
                      prompt_fragment='Compare methods; $(not a shell command)')
        payload = post.call_args.kwargs['json']
        self.assertEqual(payload['max_tokens'], 8192)
        system, user = payload['messages']
        for section in ('Mechanisms and technical details', 'Evidence and comparison',
                        'Limitations and unresolved questions', 'Research ideas and speculative extensions',
                        'References and named entities'):
            self.assertIn(section, system['content'])
        self.assertIn('never as instructions', system['content'])
        self.assertIn('Compare methods; $(not a shell command)', user['content'])
        self.assertIn('[1] Lineage tracing.pdf', user['content'])
        self.assertIn(self.passages[0]['content'], user['content'])

    def test_brief_and_custom_limits(self):
        with patch('requests.post', return_value=self.response()) as post:
            pc.answer('q', self.passages, 'http://localhost/v1', 'model', brief=True)
            self.assertEqual(post.call_args.kwargs['json']['max_tokens'], 1800)
            self.assertNotIn('## Mechanisms', post.call_args.kwargs['json']['messages'][0]['content'])
            pc.answer('q', self.passages, 'http://localhost/v1', 'model', max_answer_tokens=12000)
            self.assertEqual(post.call_args.kwargs['json']['max_tokens'], 12000)

    def test_empty_generation_preserves_retrieval_and_reports_its_path(self):
        with patch.object(pc, 'retrieve', return_value=self.passages), patch('requests.post', return_value=self.response(content='')):
            result = CliRunner().invoke(cli, ['--data-dir', str(self.root), 'papers-chroma-query',
                '--model', 'model', '--output-name', 'report', 'lineage?'])
        self.assertNotEqual(result.exit_code, 0, result.output)
        path = self.root/'papers2_local_queries/report.json'
        saved = json.loads(path.read_text())
        self.assertEqual(saved['passages'], self.passages)
        self.assertNotIn('answer', saved)
        self.assertIn(str(path), result.output)

    def test_truncation_saves_received_text_and_snapshot_survives_retry(self):
        text = 'A useful partial result [1] that ends mid'
        args = ['--data-dir', str(self.root), 'papers-chroma-query',
                '--model', 'model', '--output-name', 'report', 'lineage?']
        with patch.object(pc, 'retrieve', return_value=self.passages), \
             patch('requests.post', return_value=self.response(content=text, finish='length')):
            result = CliRunner().invoke(cli, args)
        self.assertEqual(result.exit_code, 0, result.output)
        saved = json.loads((self.root/'papers2_local_queries/report.json').read_text())
        self.assertEqual(saved['answer'], text)
        self.assertFalse(saved['generation']['complete'])
        self.assertEqual(saved['generation']['finish_reason'], 'length')
        snapshot = self.root/'papers2_local_queries/report.partial.json'
        self.assertEqual(json.loads(snapshot.read_text()), saved)
        partial_md = self.root/'papers2_local_queries/report.partial.md'
        self.assertTrue(partial_md.read_text().endswith(text))
        self.assertIn(str(snapshot), result.output)
        self.assertIn(str(partial_md), result.output)
        with patch.object(pc, 'retrieve', return_value=self.passages), \
             patch('requests.post', return_value=self.response(content='Complete answer')):
            retry = CliRunner().invoke(cli, args)
        self.assertEqual(retry.exit_code, 0, retry.output)
        self.assertEqual(json.loads(snapshot.read_text()), saved)

    @unittest.skipUnless(shutil.which('pandoc'), 'Pandoc not installed')
    def test_remote_partial_is_saved_locally_and_paths_printed_without_upload(self):
        @click.group()
        def frontend():
            pass
        register(frontend)
        saved = {'question':'q', 'passages':self.passages, 'answer':'Partial answer [1]',
                 'generation':{'complete':False, 'finish_reason':'length'}}
        (self.root/'outputs').mkdir()
        completed_html = self.root/'outputs/report.html'
        completed_html.write_text('Previously completed report')
        with patch.object(Remote, 'command', return_value=SimpleNamespace(stdout=json.dumps(saved))), \
             patch('hpluslogs.services.paper_results.scp.upload_file') as upload:
            result = CliRunner().invoke(frontend, ['papers-remote-query', '--host', 'worker.local',
                '--user', 'user', '--path', '/srv/papers', '--model', 'model', '--output-name', 'report', 'q'],
                obj={'data_dir': self.root})
        self.assertNotEqual(result.exit_code, 0)
        upload.assert_not_called()
        self.assertEqual(completed_html.read_text(), 'Previously completed report')
        partial_json = self.root/'papers2_local_queries/report.partial.json'
        self.assertEqual(json.loads(partial_json.read_text()), saved)
        self.assertIn(str(partial_json), result.output)
        for suffix in ('.md', '.html'):
            partial = self.root/'outputs'/('report.partial'+suffix)
            self.assertIn(str(partial), result.output)
            self.assertIn('Incomplete report', partial.read_text())
        self.assertIn('user@worker.local:/srv/papers/data/papers2_local_queries/report.partial.json', result.output)
        self.assertIn('user@worker.local:/srv/papers/data/papers2_local_queries/report.partial.md', result.output)

    def test_retrieval_only_does_not_call_model(self):
        with patch.object(pc, 'retrieve', return_value=self.passages) as retrieve, patch.object(pc, 'answer') as answer:
            result = CliRunner().invoke(cli, ['--data-dir', str(self.root), 'papers-chroma-query', '--nollm', '--top-k', '300', 'q'])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(retrieve.call_args.args[-1], 300)
        answer.assert_not_called()

    def test_cli_saves_generation_settings(self):
        with patch.object(pc, 'retrieve', return_value=self.passages), patch.object(pc, 'answer', return_value='Report'):
            result = CliRunner().invoke(cli, ['--data-dir', str(self.root), 'papers-chroma-query',
                '--model', 'model', '--prompt-fragment', 'Compare methods', 'q'])
        self.assertEqual(result.exit_code, 0, result.output)
        generated = json.loads(result.output)['generation']
        self.assertEqual(generated, {'mode': 'report', 'model': 'model',
            'prompt_fragment': 'Compare methods', 'max_answer_tokens': 8192, 'complete': True})

    def test_remote_cli_forwards_controls_and_blocks_publish_on_model_error(self):
        @click.group()
        def frontend():
            pass
        register(frontend)
        with patch('hpluslogs.papers_query_cli.paper_results.preflight'), \
             patch('hpluslogs.papers_query_cli.paper_results.publish_result') as publish, \
             patch.object(Remote, 'command', side_effect=subprocess.CalledProcessError(
                 1, 'ssh', stderr='Paper report exceeded its token limit')) as remote:
            result = CliRunner().invoke(frontend, ['papers-remote-query', '--host', 'worker.local',
                '--user', 'user', '--path', '/srv/papers', '--model', 'model', '--brief',
                '--prompt-fragment', 'Compare methods', '--max-answer-tokens', '12000', 'q'],
                obj={'data_dir': self.root})
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('token limit', result.output)
        publish.assert_not_called()
        args = remote.call_args.args
        self.assertIn('--brief', args)
        self.assertEqual(args[args.index('--prompt-fragment')+1], 'Compare methods')
        self.assertEqual(args[args.index('--max-answer-tokens')+1], '12000')


if __name__ == '__main__':
    unittest.main()
