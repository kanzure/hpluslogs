import unittest
import hashlib
from pathlib import Path
import tempfile
from hpluslogs.scripts.estimate_papers_chroma import chunk_count, estimated_input_tokens
from hpluslogs.scripts import estimate_papers_chroma as estimate
from hpluslogs.services import papers_markdown
from hpluslogs.scripts.papers_transfer_progress import summarize


class CapacityEstimateTest(unittest.TestCase):
    def test_usable_scope_excludes_garbled_text_without_extrapolating(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/'papers2').mkdir(); (root/'papers2_markdown').mkdir()
            for name in ('good','bad','failed'):
                (root/'papers2'/f'{name}.pdf').write_bytes(b'source')
            with papers_markdown.connect(root) as db:
                for name,text in [('good','Readable paper text. '*100),('bad','\ufffd'*100)]:
                    raw=text.encode(); (root/'papers2_markdown'/f'{name}.md').write_bytes(raw)
                    db.execute('INSERT INTO conversions VALUES (?,?,?,?,?,?,?,?,?,?)',
                               (name+'.pdf','source-hash',6,name+'.md',hashlib.sha256(raw).hexdigest(),len(raw),'recipe','ready',None,1))
            result=estimate.estimate(root,workers=1,skip_unreadable=True)
            self.assertEqual(result['pdf_count'],3)
            self.assertEqual(result['ready_markdown_count'],1)
            self.assertEqual(result['skipped_unreadable_markdown'],['bad.pdf'])
            for item in result['projections']:
                self.assertEqual(item['full_archive_vectors_range'],[item['measured_chunks']]*2)

    def test_estimator_rejects_changed_markdown(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'paper.md'; path.write_text('Changed')
            with self.assertRaisesRegex(ValueError,'changed since conversion'):
                estimate.measure_file(({'pdf_path':'paper.pdf','markdown_sha256':'old'},path))

    def test_chunk_boundaries_and_overlap(self):
        self.assertEqual(chunk_count(0, 800, 100), 0)
        self.assertEqual(chunk_count(800, 800, 100), 1)
        self.assertEqual(chunk_count(801, 800, 100), 2)
        self.assertEqual(chunk_count(1500, 800, 100), 2)
        self.assertEqual(chunk_count(1501, 800, 100), 3)
        with self.assertRaises(ValueError):
            chunk_count(100, 800, 800)

    def test_billing_includes_repeated_overlap_and_special_tokens(self):
        self.assertEqual(estimated_input_tokens(0,0,800,100),0)
        self.assertEqual(estimated_input_tokens(800,1000,800,100),1001)
        self.assertEqual(estimated_input_tokens(1500,1800,800,100),1922)

    def test_transfer_excludes_extras_and_outdated_complete_files(self):
        expected = {'ready': [100, 10], 'partial': [200, 20], 'old': [100, 30], 'missing': [50, 40]}
        actual = {'ready': [100, 10], 'partial': [70, 999], 'old': [100, 1], 'unrelated': [9999, 1]}
        self.assertEqual(summarize(expected, actual), (170, 1))
