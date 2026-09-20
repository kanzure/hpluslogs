import hashlib
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from hpluslogs.services import papers_chroma as pc


class Tokenizer:
    def encode(self,text,**kwargs):
        return SimpleNamespace(offsets=[(i,i+1) for i in range(len(text))])


class Encoder:
    tokenizer = Tokenizer()
    def embed(self,texts):
        return [[1.,0.] for _ in texts]


class Collection:
    id = 'test-collection'
    def __init__(self):
        self.docs = {}
        self.calls = 0
        self.fail_at = None
    def delete(self,where):
        self.docs = {key:value for key,value in self.docs.items() if value[1]['paper_key'] != where['paper_key']}
    def upsert(self,ids,embeddings,documents,metadatas):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError('Interrupted request')
        self.docs.update(zip(ids, zip(documents,metadatas)))


class ChromaTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name)
        (self.data/'papers2_markdown').mkdir()
        self.coll = Collection()
        self.enc = Encoder()
    def row(self,text):
        (self.data/'papers2_markdown'/'paper.md').write_text(text)
        return {'pdf_path':'topic/paper.pdf','markdown_path':'paper.md',
                'markdown_sha256':hashlib.sha256(text.encode()).hexdigest()}
    def test_offsets_and_overlap_preserve_all_text(self):
        text=''.join(str(i%10) for i in range(900))
        pieces=list(pc.chunks(text,self.enc.tokenizer))
        self.assertEqual(pieces[0]['start'],0)
        self.assertEqual(pieces[-1]['end'],len(text))
        for a,b in zip(pieces,pieces[1:]):
            self.assertEqual(a['end']-b['start'],32)
        self.assertTrue(all(p['content']==text[p['start']:p['end']] for p in pieces))
    def test_resume_replacement_and_collection_recreation(self):
        row=self.row('abc '*600)
        self.coll.fail_at=2
        self.assertTrue(pc.index_document(self.data,self.coll,self.enc,row,batch_size=2).startswith('failed'))
        with pc.checkpoint(self.data) as db:
            self.assertEqual(db.execute('SELECT next_chunk FROM indexed').fetchone()[0],2)
        self.coll.fail_at=None
        self.assertEqual(pc.index_document(self.data,self.coll,self.enc,row,2),'indexed')
        before=self.coll.calls
        self.assertEqual(pc.index_document(self.data,self.coll,self.enc,row,2),'unchanged')
        self.assertEqual(self.coll.calls,before)
        row=self.row('Replacement document')
        self.assertEqual(pc.index_document(self.data,self.coll,self.enc,row,2),'indexed')
        self.assertEqual(len(self.coll.docs),1)
        self.assertEqual(next(iter(self.coll.docs.values()))[0],'Replacement document')
        new=Collection(); new.id='new-collection'
        self.assertEqual(pc.index_document(self.data,new,self.enc,row,2),'indexed')
        self.assertEqual(len(new.docs),1)
    def test_changed_markdown_is_not_indexed(self):
        row=self.row('Original')
        (self.data/'papers2_markdown'/'paper.md').write_text('Changed')
        self.assertTrue(pc.index_document(self.data,self.coll,self.enc,row).startswith('failed'))
        self.assertFalse(self.coll.docs)
        with pc.checkpoint(self.data) as db:
            self.assertEqual(db.execute('SELECT state FROM indexed').fetchone()[0], 'failed')


if __name__=='__main__':
    unittest.main()
