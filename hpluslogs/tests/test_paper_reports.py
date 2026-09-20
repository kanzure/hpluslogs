import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

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

    def test_failed_generation_preserves_retrieval_and_returns_failure(self):
        for response in (self.response(finish='length'), self.response(content='')):
            with patch.object(pc, 'retrieve', return_value=self.passages), patch('requests.post', return_value=response):
                result = CliRunner().invoke(cli, ['--data-dir', str(self.root), 'papers-chroma-query',
                    '--model', 'model', '--output-name', 'report', 'lineage?'])
            self.assertNotEqual(result.exit_code, 0, result.output)
            saved = json.loads((self.root/'papers2_local_queries/report.json').read_text())
            self.assertEqual(saved['passages'], self.passages)
            self.assertNotIn('answer', saved)

    def test_retrieval_only_does_not_call_model(self):
        with patch.object(pc, 'retrieve', return_value=self.passages), patch.object(pc, 'answer') as answer:
            result = CliRunner().invoke(cli, ['--data-dir', str(self.root), 'papers-chroma-query', '--nollm', 'q'])
        self.assertEqual(result.exit_code, 0, result.output)
        answer.assert_not_called()

    def test_cli_saves_generation_settings(self):
        with patch.object(pc, 'retrieve', return_value=self.passages), patch.object(pc, 'answer', return_value='Report'):
            result = CliRunner().invoke(cli, ['--data-dir', str(self.root), 'papers-chroma-query',
                '--model', 'model', '--prompt-fragment', 'Compare methods', 'q'])
        self.assertEqual(result.exit_code, 0, result.output)
        generated = json.loads(result.output)['generation']
        self.assertEqual(generated, {'mode': 'report', 'model': 'model',
            'prompt_fragment': 'Compare methods', 'max_answer_tokens': 8192})

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
