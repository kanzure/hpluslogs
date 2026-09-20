from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from hpluslogs.services import papers, papers_markdown as md, papers_repair as repair


class RepairTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root/'papers2'/'broken.pdf'
        self.source.parent.mkdir()
        self.source.write_bytes(b'unchanged source')
        self.target = md.root(self.root)/md.output_path('broken.pdf')
        self.target.parent.mkdir(parents=True)
        self.target.write_text('\ufffd'*100+' original')
        self.old = self.target.read_bytes()
        with md.connect(self.root) as db:
            db.execute('INSERT INTO conversions VALUES (?,?,?,?,?,?,?,?,?,?)',
                       ('broken.pdf', papers.sha256(self.source), self.source.stat().st_size,
                        md.output_path('broken.pdf'), papers.sha256(self.target), len(self.old),
                        'recipe', 'ready', None, 1))
        self.row = repair.candidates(self.root)[0]

    def output(self, command, **kwargs):
        path = Path(command[3])
        path.write_text('Recovered scientific paper text.')
        Path(str(path)+'.extraction.json').write_text('{"parsers":["raster-ocr-200-v1"]}')
        return SimpleNamespace(returncode=0)

    def test_repair_updates_manifest_preserves_source_and_backup_and_is_idempotent(self):
        with patch.object(repair.subprocess, 'run', side_effect=self.output) as call:
            result = repair.repair_one(self.root, self.row, 60)
        self.assertEqual(result['result'], 'repaired')
        self.assertEqual(call.call_args.kwargs['env']['PAPERS_PARSER'], 'raster-ocr')
        self.assertEqual(self.source.read_bytes(), b'unchanged source')
        self.assertEqual(next((self.root/'papers2_repair_backups').iterdir()).read_bytes(), self.old)
        self.assertEqual(repair.candidates(self.root), [])
        with md.connect(self.root) as db:
            row = db.execute('SELECT * FROM conversions').fetchone()
            self.assertEqual(row['markdown_sha256'], papers.sha256(self.target))
            self.assertEqual(row['recipe'], 'recipe')

    def test_failed_repair_does_not_replace_existing_output_or_checkpoint(self):
        with patch.object(repair.subprocess, 'run', return_value=SimpleNamespace(returncode=1, stderr=b'failed')):
            result = repair.repair_one(self.root, self.row, 60)
        self.assertEqual(result['result'], 'failed')
        self.assertEqual(self.target.read_bytes(), self.old)
        self.assertEqual(repair.candidates(self.root), [self.row])

    def test_changed_source_is_not_repaired(self):
        self.source.write_bytes(b'new source')
        with patch.object(repair.subprocess, 'run') as call:
            self.assertEqual(repair.repair_one(self.root, self.row, 60)['result'], 'failed')
            call.assert_not_called()
        self.assertEqual(self.target.read_bytes(), self.old)
