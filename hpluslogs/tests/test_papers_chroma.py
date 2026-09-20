import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from hpluslogs.services import papers_chroma as pc


class Tokenizer:
    def encode(self,text,**kwargs):
        return SimpleNamespace(offsets=[(i,i+1) for i in range(len(text))])


class Encoder:
    tokenizer = Tokenizer()
    def embed(self,texts,**kwargs):
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
    def count(self):
        return len(self.docs)
    def get(self,limit,offset,include):
        items=list(self.docs.items())[offset:offset+limit]
        return {'ids':[i for i,_ in items], 'documents':[v[0] for _,v in items],
                'metadatas':[v[1] for _,v in items]}


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
            self.assertEqual(a['end']-b['start'],20)
        self.assertTrue(all(p['content']==text[p['start']:p['end']] for p in pieces))

    def test_chunk_eta_accounts_for_size_and_rejects_stale_scope(self):
        report={'projection_scope':'usable ready Markdown only','ready_markdown_count':1,
                'skipped_unreadable_markdown':['bad.pdf'],'markdown_bytes':100,'measured_at':'now',
                'projections':[{'chunk_size':175,'overlap':20,'dimensions':4096,'measured_chunks':1000}]}
        (self.data/'papers2_usable_token_estimate.json').write_text(json.dumps(report))
        rows=[{'pdf_path':'good.pdf','markdown_bytes':100},{'pdf_path':'bad.pdf','markdown_bytes':50}]
        result=pc.chunk_eta(self.data,rows,{'initial_chunks':100},300,3600)
        self.assertEqual(result['estimated_remaining_chunks'],700)
        self.assertEqual(result['estimated_chunk_backlog_eta_hours'],3.5)
        rows[0]['markdown_bytes']=101
        self.assertNotIn('estimated_chunk_backlog_eta_hours',pc.chunk_eta(self.data,rows,{'initial_chunks':100},300,3600))
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

    def test_corrupted_encoding_is_rejected_before_paid_embedding(self):
        row=self.row('Unreadable '+ '\ufffd'*100)
        with patch.object(self.enc,'embed') as embed:
            result=pc.index_document(self.data,self.coll,self.enc,row)
        self.assertIn('encoding quality',result)
        embed.assert_not_called()

    def test_audit_detects_missing_and_corrupted_passages(self):
        row=self.row('Verified text '*100)
        pc.index_document(self.data,self.coll,self.enc,row)
        with patch.object(pc,'collection',return_value=self.coll), \
             patch.object(pc,'source_rows',return_value=[row]):
            report=pc.audit(self.data,'host',18081,page_size=2)
            self.assertTrue(report['all_ready_markdown_verified'],report)
            ident=next(iter(self.coll.docs))
            original=self.coll.docs[ident]
            self.coll.docs[ident]=('Corrupt text',original[1])
            report=pc.audit(self.data,'host',18081,page_size=2)
            self.assertFalse(report['all_ready_markdown_verified'])
            self.assertTrue(any('differs from original' in e for e in report['errors']))
            del self.coll.docs[ident]
            report=pc.audit(self.data,'host',18081,page_size=2)
            self.assertFalse(report['all_ready_markdown_verified'])
            self.assertTrue(any('chunk count differs' in e for e in report['errors']))

    def test_audit_detects_stale_markdown(self):
        row=self.row('Verified text')
        pc.index_document(self.data,self.coll,self.enc,row)
        (self.data/'papers2_markdown'/'paper.md').write_text('Tampered text')
        with patch.object(pc,'collection',return_value=self.coll), \
             patch.object(pc,'source_rows',return_value=[row]):
            report=pc.audit(self.data,'host',18081)
            self.assertFalse(report['all_ready_markdown_verified'])
            self.assertTrue(any('digest differs' in e for e in report['errors']))

    def test_audit_reports_explicit_unreadable_exclusions_without_ignoring_missing_good_papers(self):
        good = self.row('Verified text')
        pc.index_document(self.data,self.coll,self.enc,good)
        text = '\ufffd'*100
        (self.data/'papers2_markdown'/'bad.md').write_text(text)
        bad = {'pdf_path':'bad.pdf', 'markdown_path':'bad.md', 'markdown_sha256':pc.digest(text)}
        pc.index_document(self.data,self.coll,self.enc,bad)
        with patch.object(pc,'collection',return_value=self.coll), \
             patch.object(pc,'source_rows',return_value=[good,bad]):
            report=pc.audit(self.data,'host',18081)
            self.assertFalse(report['all_ready_markdown_verified'])
            self.assertTrue(report['all_eligible_markdown_verified'],report)
            self.assertEqual(report['skipped_unreadable_markdown'],['bad.pdf'])
            self.coll.docs.clear()
            self.assertFalse(pc.audit(self.data,'host',18081)['all_eligible_markdown_verified'])


if __name__=='__main__':
    unittest.main()
