from pathlib import Path
import tempfile
import unittest
from hpluslogs.services.paper_conversion_worker import cached_markdown


class PageCacheTest(unittest.TestCase):
    def test_interrupted_book_resumes_without_repeating_finished_batches(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            calls=[]
            def render(doc,pages,**kwargs):
                calls.append(pages)
                if pages[0] == 20:
                    raise TimeoutError('Interrupted book')
                return '|'.join(str(p) for p in pages)
            with self.assertRaises(TimeoutError):
                cached_markdown(range(45),root,render)
            self.assertEqual(len(list(root.glob('*.json'))),1)
            calls.clear()
            def resumed(doc,pages,**kwargs):
                calls.append(pages)
                return '|'.join(str(p) for p in pages)
            text=cached_markdown(range(45),root,resumed)
            self.assertEqual(calls,[list(range(20,40)),list(range(40,45))])
            self.assertEqual([int(n) for n in text.replace('\n\n','|').split('|')],list(range(45)))
            calls.clear()
            self.assertEqual(cached_markdown(range(45),root,resumed),text)
            self.assertFalse(calls)
            (root/'00000000-00000020.json').write_text('{corrupt')
            self.assertEqual(cached_markdown(range(45),root,resumed),text)
            self.assertEqual(calls,[list(range(20))])


if __name__=='__main__':
    unittest.main()
