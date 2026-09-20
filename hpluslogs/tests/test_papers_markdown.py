import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import click

from hpluslogs.services import papers, papers_markdown as md


class MarkdownTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name)
        self.root = self.data / 'papers2'
        self.root.mkdir()
        self.source = self.root / 'paper.pdf'
        self.source.write_bytes(b'%PDF-1.7\n' + b'x'*5000)
        self.runner = patch.object(md.subprocess, 'run', side_effect=self.fake_convert).start()
        self.addCleanup(patch.stopall)

    def fake_convert(self, command, **kwargs):
        Path(command[3]).write_text('# Scientific paper\n\nExtracted text.\n', encoding='utf-8')
        return NS(returncode=0)

    def test_resumes_and_reconverts_changed_pdf_or_output(self):
        self.assertEqual(md.convert_one(self.data, self.source, 'recipe1', 10)[1], 'converted')
        self.assertEqual(md.convert_one(self.data, self.source, 'recipe1', 10)[1], 'unchanged')
        self.source.write_bytes(b'%PDF-1.7\nchanged')
        self.assertEqual(md.convert_one(self.data, self.source, 'recipe1', 10)[1], 'converted')
        output = md.root(self.data) / md.output_path('paper.pdf')
        output.write_text('tampered')
        self.assertEqual(md.convert_one(self.data, self.source, 'recipe1', 10)[1], 'converted')
        self.assertEqual(md.convert_one(self.data, self.source, 'recipe2', 10)[1], 'converted')
        self.assertEqual(self.runner.call_count, 4)

    def test_cost_measures_markdown_not_pdf(self):
        md.convert_one(self.data, self.source, 'r', 10)
        with patch.object(papers.xai, 'get_sync_client') as client:
            report = md.cost_report(self.data)
            client.assert_not_called()
        self.assertTrue(report['local_conversion_complete'])
        self.assertEqual(report['markdown_bytes'], len('# Scientific paper\n\nExtracted text.\n'.encode()))
        self.assertGreater(report['converted_source_pdf_bytes'], report['markdown_bytes']*100)
        self.assertAlmostEqual(report['cost']['estimated_usd_excluding_tokens_downloads'], report['markdown_bytes']/2**30*3.75)
        self.assertTrue((self.data/'papers2_markdown_cost.json').exists())

    def test_stale_pdf_and_markdown_are_not_priced_as_ready(self):
        md.convert_one(self.data, self.source, 'r', 10)
        self.source.write_bytes(b'%PDF-1.7\nchanged')
        report = md.cost_report(self.data)
        self.assertEqual(report['ready_markdown_count'], 0)
        self.assertFalse(report['local_conversion_complete'])
        md.convert_one(self.data, self.source, 'r', 10)
        (md.root(self.data)/md.output_path('paper.pdf')).write_text('edited')
        self.assertEqual(md.cost_report(self.data)['ready_markdown_count'], 0)

    def test_failed_conversion_excludes_previous_markdown(self):
        md.convert_one(self.data, self.source, 'r', 10)
        self.source.write_bytes(b'%PDF-1.7\nchanged')
        self.runner.return_value = NS(returncode=1, stderr=b'Broken PDF')
        self.runner.side_effect = None
        self.assertIn('Broken PDF', md.convert_one(self.data, self.source, 'r', 10)[1])
        self.assertEqual(md.cost_report(self.data)['ready_markdown_count'], 0)

    def test_empty_conversion_fails(self):
        def empty(command, **kwargs):
            Path(command[3]).write_text('  \n')
            return NS(returncode=0)
        self.runner.side_effect = empty
        self.assertIn('No Markdown', md.convert_one(self.data, self.source, 'r', 10)[1])

    def test_native_crash_retries_alternate_parser_with_remaining_deadline(self):
        def retry(command, **kwargs):
            if self.runner.call_count == 1:
                return NS(returncode=-11, stderr=b'')
            self.assertEqual(kwargs['env']['PAPERS_PARSER'], 'legacy')
            self.assertGreater(kwargs['timeout'], 0)
            self.assertLessEqual(kwargs['timeout'], 10)
            Path(command[3]+'.extraction.json').write_text('{"parsers":["legacy-hidden-text"]}')
            return self.fake_convert(command, **kwargs)
        self.runner.side_effect = retry
        self.assertEqual(md.convert_one(self.data, self.source, 'r', 10)[1], 'converted')
        metadata = self.data/'papers2_extraction_metadata'/(md.output_path('paper.pdf')+'.json')
        self.assertEqual(json.loads(metadata.read_text())['parsers'], ['legacy-hidden-text'])
        self.assertEqual(self.runner.call_count, 2)

    def test_termination_does_not_trigger_retry(self):
        self.runner.side_effect = None
        self.runner.return_value = NS(returncode=-15, stderr=b'')
        self.assertIn('exited -15', md.convert_one(self.data, self.source, 'r', 10)[1])
        self.assertEqual(self.runner.call_count, 1)

    def test_pdf_changed_during_conversion_fails(self):
        def concurrent_change(command, **kwargs):
            result = self.fake_convert(command, **kwargs)
            self.source.write_bytes(b'changed while converting')
            return result
        self.runner.side_effect = concurrent_change
        self.assertIn('changed during conversion', md.convert_one(self.data, self.source, 'r', 10)[1])

    def test_names_are_unique_and_safe(self):
        self.assertNotEqual(md.output_path('a/paper.pdf'), md.output_path('b/paper.pdf'))
        self.assertNotEqual(md.output_path('paper.pdf'), md.output_path('paper.PDF'))
        self.assertLess(len(Path(md.output_path('a'*250+'.pdf')).name), 255)
        md.output_path('legacy-\udcf4.pdf').encode('ascii')

    def test_incomplete_restore_is_labeled_partial(self):
        md.convert_one(self.data, self.source, 'r', 10)
        (self.data/'papers2-remote-inventory.json').write_text(json.dumps([
            {'path':'paper.pdf', 'size':self.source.stat().st_size}, {'path':'missing.pdf','size':10}]))
        report = md.cost_report(self.data)
        self.assertEqual(report['scope'], 'partial collection only')
        self.assertEqual(report['source_inventory_missing_or_different_size'], 1)
        with patch.object(papers.xai, 'get_sync_client') as client:
            with self.assertRaisesRegex(click.ClickException, 'Source inventory is incomplete'):
                papers.upload(self.data, approve_upload=True)
            client.assert_not_called()

    def test_unconverted_pdf_blocks_approved_upload(self):
        with patch.object(papers.xai, 'get_sync_client') as client:
            with self.assertRaisesRegex(click.ClickException, 'Convert every local PDF'):
                papers.upload(self.data, approve_upload=True)
            client.assert_not_called()

    def test_hidden_macos_sidecars_do_not_make_restore_incomplete(self):
        md.convert_one(self.data, self.source, 'r', 10)
        (self.data/'papers2-remote-inventory.json').write_text(json.dumps([
            {'path':'paper.pdf', 'size':self.source.stat().st_size},
            {'path':'._paper.pdf', 'size':100}]))
        report = md.cost_report(self.data)
        self.assertEqual(report['scope'], 'full source inventory')
        self.assertEqual(report['source_inventory_pdf_count'], 1)
        self.assertEqual(report['source_inventory_excluded_hidden_pdf_entries'], 1)
        self.assertEqual(report['source_inventory_missing_or_different_size'], 0)

    def test_prepare_reports_costs_even_when_conversion_fails(self):
        with patch.object(md, 'convert', side_effect=click.ClickException('Conversion failed')), \
             patch.object(papers.xai, 'get_sync_client') as client:
            with self.assertRaisesRegex(click.ClickException, 'Conversion failed'):
                md.prepare(self.data)
            client.assert_not_called()
        report = json.loads((self.data/'papers2_markdown_cost.json').read_text())
        self.assertFalse(report['local_conversion_complete'])
        self.assertEqual(report['ready_markdown_count'], 0)

    def test_conversion_worker_matches_reference_settings(self):
        from hpluslogs.services import paper_conversion_worker as worker
        output = self.data/'out.md'
        with patch.object(worker, 'configure_inference_threads'), \
             patch('pymupdf4llm.to_markdown', return_value='# markdown') as convert, \
             patch.object(worker.sys, 'argv', ['worker', 'input.pdf', str(output)]):
            worker.main()
        convert.assert_called_once_with('input.pdf', header=False, footer=False)
        self.assertEqual(output.read_text(), '# markdown')

    def test_inference_thread_limits_preserve_session_options_and_providers(self):
        from hpluslogs.services import paper_conversion_worker as worker
        class Options:
            def __init__(self):
                self.entries = {}
                self.enable_cpu_mem_arena = False
            def add_session_config_entry(self, name, value):
                self.entries[name] = value
        class Session:
            def __init__(self, model, opts, *args, **kwargs):
                self.model, self.options, self.extra = model, opts, (args, kwargs)
        fake = NS(InferenceSession=Session, SessionOptions=Options)
        with patch.dict('sys.modules', {'onnxruntime': fake}):
            worker.configure_inference_threads(1)
            options = Options()
            session = fake.InferenceSession('model.onnx', options, providers=['CPUExecutionProvider'])
            self.assertIs(session.options, options)
            self.assertFalse(options.enable_cpu_mem_arena)
            self.assertEqual(options.intra_op_num_threads, 1)
            self.assertEqual(options.inter_op_num_threads, 1)
            self.assertEqual(options.entries['session.intra_op.allow_spinning'], '0')
            self.assertEqual(session.extra[1]['providers'], ['CPUExecutionProvider'])
            self.assertEqual(fake.InferenceSession('model.onnx').options.intra_op_num_threads, 1)
        with self.assertRaises(ValueError):
            worker.configure_inference_threads(0)

    def test_legacy_filename_uses_document_stream(self):
        from hpluslogs.services import paper_conversion_worker as worker
        source = self.data / 'legacy-\udcff.pdf'
        source.write_bytes(b'%PDF-test')
        output = self.data / 'legacy.md'
        with patch.object(worker, 'configure_inference_threads'), \
             patch('pymupdf.open') as open_pdf, \
             patch('pymupdf4llm.to_markdown', return_value='# legacy') as convert, \
             patch.object(worker.sys, 'argv', ['worker', str(source), str(output)]):
            worker.main()
        open_pdf.assert_called_once_with(stream=b'%PDF-test', filetype='pdf')
        convert.assert_called_once_with(open_pdf.return_value, header=False, footer=False)
