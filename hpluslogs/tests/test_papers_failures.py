from pathlib import Path
import tempfile
import unittest
import pymupdf
from hpluslogs.services.papers_failures import classify


class FailuresTest(unittest.TestCase):
    def test_distinguishes_valid_empty_and_invalid_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            empty=root/'empty.pdf'; empty.write_bytes(b'')
            invalid=root/'invalid.pdf'; invalid.write_bytes(b'<html>Not a PDF</html>')
            valid=root/'legacy-\udcff.pdf'
            doc=pymupdf.open(); doc.new_page().insert_text((72,72),'Scientific paper')
            valid.write_bytes(doc.tobytes()); doc.close()
            self.assertEqual(classify(empty)['classification'],'empty_file')
            self.assertEqual(classify(invalid)['classification'],'not_pdf')
            result=classify(valid)
            self.assertEqual(result['classification'],'readable_pdf_extraction_failed')
            self.assertEqual(result['pages'],1)
