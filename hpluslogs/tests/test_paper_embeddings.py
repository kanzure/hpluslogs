from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, patch

from hpluslogs.services import paper_embeddings as pe, papers_chroma as pc


class PaperEmbeddingsTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def response(self):
        return NS(data=[NS(index=1,embedding=[2.]*4096),NS(index=0,embedding=[1.]*4096)],
                  usage=NS(prompt_tokens=10,cost=0.0000001),id='request-test')

    def test_order_dimensions_provider_price_and_durable_cache(self):
        client=MagicMock(); client.with_options.return_value=client
        client.embeddings.create.return_value=self.response()
        encoder=pe.Encoder(self.root)
        with patch.object(pe.openrouter,'get_sync_client',return_value=client):
            vectors=encoder.embed(['one','two'],key='batch1')
            self.assertEqual([v[0] for v in vectors],[1.,2.])
            self.assertEqual(encoder.embed(['one','two'],key='batch1'),vectors)
            client.embeddings.create.assert_called_once()
            params=client.embeddings.create.call_args.kwargs
            self.assertEqual(params['model'],'qwen/qwen3-embedding-8b')
            self.assertEqual(params['dimensions'],4096)
            self.assertNotIn('encoding_format', params)  # SDK base64 + automatic float decoding
            self.assertEqual(params['extra_body']['provider']['max_price']['prompt'],0.01)
            self.assertEqual(params['extra_body']['provider']['only'],['nebius','deepinfra'])
            self.assertEqual(client.with_options.call_args.kwargs['max_retries'],0)
        self.assertAlmostEqual(pe.usage(self.root)['accounted_usd'],0.0000001)
        encoder.forget(['one','two'],'batch1')
        self.assertFalse(list(encoder.cache.glob('*.json')))

    def test_wrong_dimensions_fail_after_recording_actual_charge(self):
        client=MagicMock();client.with_options.return_value=client
        response=self.response();response.data[0].embedding=[2.]*384
        client.embeddings.create.return_value=response
        with patch.object(pe.openrouter,'get_sync_client',return_value=client):
            with self.assertRaisesRegex(ValueError,'4096'):
                pe.Encoder(self.root).embed(['one','two'])
        self.assertAlmostEqual(pe.usage(self.root)['accounted_usd'],0.0000001)

    def test_sdk_decodes_compact_transport_and_cache_preserves_values(self):
        import base64
        import json
        import struct
        import httpx
        import openai
        raw = base64.b64encode(struct.pack('<4096f', *([.125]*4096))).decode()
        def respond(request):
            self.assertEqual(json.loads(request.content)['encoding_format'], 'base64')
            return httpx.Response(200,json={'object':'list','model':pe.MODEL,
                'data':[{'object':'embedding','index':0,'embedding':raw}],
                'usage':{'prompt_tokens':5,'total_tokens':5,'cost':.00000005}})
        with openai.OpenAI(api_key='test',http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
            with patch.object(pe.openrouter,'get_sync_client',return_value=client):
                encoder=pe.Encoder(self.root)
                vectors=encoder.embed(['text'],key='compact')
                self.assertEqual(vectors,[[.125]*4096])
                with patch.object(client.embeddings,'create',side_effect=AssertionError('Cache must avoid API')):
                    self.assertEqual(encoder.embed(['text'],key='compact'),vectors)

    def test_budget_reservations_are_atomic_and_survive_restart(self):
        encoder=pe.Encoder(self.root,cost_limit=0.000001)
        def reserve(_):
            try: encoder.reserve(['a']); return True
            except pe.BudgetExceeded: return False
        with ThreadPoolExecutor(max_workers=8) as pool:
            self.assertEqual(sum(pool.map(reserve,range(8))),3)
        with self.assertRaises(pe.BudgetExceeded):
            pe.Encoder(self.root,cost_limit=0.000001).reserve(['a'])

    def test_unicode_chunk_offsets_keep_original_text(self):
        text=('A DNA 🧬 test: 中文 Ελληνικά café.\n'*100)
        pieces=list(pc.chunks(text,pe.Tokenizer()))
        self.assertEqual(pieces[0]['start'],0)
        self.assertEqual(pieces[-1]['end'],len(text))
        self.assertTrue(all(p['content']==text[p['start']:p['end']] for p in pieces))
        self.assertTrue(all(a['end']>=b['start'] for a,b in zip(pieces,pieces[1:])))
