import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hpluslogs.services import paper_conversion_worker as worker


class TextFallbackTest(unittest.TestCase):
    def test_normal_conversion_disables_automatic_ocr(self):
        with patch('pymupdf4llm.to_markdown', return_value='# Existing text') as render, \
             patch.dict('os.environ', {'PAPERS_PARSER': ''}):
            self.assertEqual(worker.render_markdown('paper'), '# Existing text')
        self.assertFalse(render.call_args.kwargs['use_ocr'])

    def test_surrogate_pair_is_preserved_and_isolated_surrogate_is_marked(self):
        original = 'DNA \ud83e\uddec; invalid \udcff; ordinary café'
        result = worker.normalize_unicode(original)
        self.assertEqual(result, 'DNA 🧬; invalid \ufffd; ordinary café')
        self.assertEqual(result.encode().decode(), result)
        self.assertEqual(worker.render_markdown.last_unicode_repairs,
                         ['surrogate-pair-decoded','unpaired-surrogate-replaced'])
        self.assertEqual(worker.normalize_unicode('valid 🧬'), 'valid 🧬')
        self.assertEqual(worker.render_markdown.last_unicode_repairs, [])

    def test_empty_layout_recovers_hidden_text_and_restores_parser(self):
        with patch('pymupdf4llm.to_markdown', side_effect=['', '# Actual body']) as render, \
             patch('pymupdf4llm.use_layout') as mode, \
             patch.dict('os.environ', {'PAPERS_PARSER': ''}):
            self.assertEqual(worker.render_markdown('paper', header=False, footer=False, pages=[0]), '# Actual body')
        self.assertEqual([call.args for call in mode.call_args_list], [(False,), (True,)])
        self.assertEqual(render.call_args.kwargs, dict(ignore_alpha=True, ignore_images=True, ignore_graphics=True, pages=[0]))
        self.assertEqual(worker.render_markdown.last_parser, 'legacy-hidden-text')

    def test_exception_preserves_original_error_and_restores_parser(self):
        with patch('pymupdf4llm.to_markdown', side_effect=[ValueError('original'), RuntimeError('fallback')]), \
             patch('pymupdf4llm.use_layout') as mode, \
             patch.dict('os.environ', {'PAPERS_PARSER': ''}):
            with self.assertRaisesRegex(ValueError, 'original'):
                worker.render_markdown('paper')
        self.assertEqual(mode.call_args.args, (True,))

    def test_old_empty_batch_is_rechecked_but_new_empty_batch_is_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'00000000-00000001.json').write_text(json.dumps({
                'pages': [0,1], 'markdown': '', 'sha256': hashlib.sha256(b'').hexdigest()}))
            with patch('pymupdf4llm.to_markdown', return_value='') as render, \
                 patch('pymupdf4llm.use_layout'):
                worker.cached_markdown(range(1), root, worker.render_markdown)
                self.assertEqual(render.call_count, 2)
                worker.cached_markdown(range(1), root, worker.render_markdown)
                self.assertEqual(render.call_count, 2)
