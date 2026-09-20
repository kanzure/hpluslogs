"""Qwen paper embeddings through the project's OpenRouter client."""
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import tempfile
import time
from types import SimpleNamespace
import uuid
import orjson

import click
from hpluslogs.integrations import openrouter

MODEL = 'qwen/qwen3-embedding-8b'
DIMENSIONS = 4096
PRICE_PER_MILLION = 0.01


class BudgetExceeded(click.ClickException):
    pass


class Tokenizer:
    def __init__(self):
        import tiktoken
        self.encoding = tiktoken.get_encoding('o200k_base')

    def encode(self, text, **kwargs):
        ids = self.encoding.encode(text, disallowed_special=())
        decoded, starts = self.encoding.decode_with_offsets(ids)
        if decoded != text:
            raise ValueError('Tokenizer did not preserve Markdown text')
        ends = starts[1:] + [len(text)]
        return SimpleNamespace(offsets=[(a,max(a+1,b)) for a,b in zip(starts,ends)])


def usage_db(data_dir):
    db = sqlite3.connect(data_dir/'papers2_embedding_usage.sqlite3', timeout=60)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('''CREATE TABLE IF NOT EXISTS requests (
        id TEXT PRIMARY KEY, reserved_usd REAL, cost_usd REAL, prompt_tokens INTEGER,
        request_id TEXT, state TEXT, error TEXT, created REAL)''')
    return db


def usage(data_dir):
    with usage_db(data_dir) as db:
        row = db.execute('''SELECT count(*) requests, coalesce(sum(cost_usd),0) accounted_usd,
            coalesce(sum(CASE WHEN cost_usd IS NULL THEN reserved_usd ELSE 0 END),0) unresolved_reserved_usd,
            coalesce(sum(prompt_tokens),0) prompt_tokens FROM requests''').fetchone()
    return dict(row)


def validate_vectors(response, count):
    items = sorted(response.data, key=lambda item:item.index)
    if [item.index for item in items] != list(range(count)):
        raise ValueError('Embedding response has missing/duplicate input indices')
    vectors = [item.embedding for item in items]
    if any(len(v)!=DIMENSIONS or not all(math.isfinite(x) for x in v) or not any(v) for v in vectors):
        raise ValueError('Expected finite, nonzero 4096-dimensional Qwen embeddings')
    return vectors


class Encoder:
    def __init__(self, data_dir, cost_limit=5.0):
        if not math.isfinite(cost_limit) or cost_limit <= 0:
            raise ValueError('Embedding cost limit must be positive and finite')
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True,exist_ok=True)
        self.cost_limit = cost_limit
        self.tokenizer = Tokenizer()
        self.cache = self.data_dir/'papers2_embedding_pending'
        self.cache.mkdir(exist_ok=True)
        usage_db(self.data_dir).close()

    def cache_path(self, key, texts):
        value = json.dumps([MODEL,DIMENSIONS,key,texts],ensure_ascii=False).encode()
        return self.cache/(hashlib.sha256(value).hexdigest()+'.json')

    def reserve(self, texts):
        # Byte-level Qwen tokens cannot exceed UTF-8 bytes, plus special-token headroom.
        upper_tokens = sum(len(t.encode())+32 for t in texts)
        amount = upper_tokens/1e6*PRICE_PER_MILLION
        ident = uuid.uuid4().hex
        with usage_db(self.data_dir) as db:
            db.execute('BEGIN IMMEDIATE')
            spent = db.execute('SELECT coalesce(sum(coalesce(cost_usd,reserved_usd)),0) FROM requests').fetchone()[0]
            if spent+amount > self.cost_limit:
                raise BudgetExceeded(f'Embedding ledger would exceed ${self.cost_limit:.2f}; inspect papers-chroma-status or raise --cost-limit.')
            db.execute('INSERT INTO requests VALUES (?,?,NULL,NULL,NULL,?,NULL,?)',
                       (ident,amount,'reserved',time.time()))
        return ident

    def embed(self, texts, key=None):
        if not texts:
            return []
        if any(len(t.encode())+32 > 32000 for t in texts):
            raise ValueError('Input exceeds conservative Qwen context bound')
        path = self.cache_path(key,texts)
        if key is not None and path.exists():
            saved = orjson.loads(path.read_bytes())
            response = SimpleNamespace(data=[SimpleNamespace(index=i,embedding=v) for i,v in enumerate(saved)])
            return validate_vectors(response,len(texts))
        ident = self.reserve(texts)
        try:
            # Reuse project authentication/endpoint. Explicit retries stay accounted in the ledger.
            client = openrouter.get_sync_client().with_options(max_retries=0,timeout=180)
            # The SDK defaults to compact base64 transport and decodes to floats.
            response = client.embeddings.create(model=MODEL, input=texts, dimensions=DIMENSIONS,
                extra_body={'provider': {
                    'only':['nebius','deepinfra'], 'sort':'price',
                    'max_price':{'prompt':PRICE_PER_MILLION,'completion':0,'request':0}}})
            tokens = getattr(response.usage,'prompt_tokens',None)
            cost = getattr(response.usage,'cost',None)
            if cost is None and isinstance(tokens,int) and tokens >= 0:
                cost = tokens/1e6*PRICE_PER_MILLION
            if cost is not None and (not math.isfinite(cost) or cost < 0):
                raise ValueError('Invalid embedding usage cost')
            with usage_db(self.data_dir) as db:
                db.execute('UPDATE requests SET cost_usd=?,prompt_tokens=?,request_id=?,state=? WHERE id=?',
                           (cost,tokens,getattr(response,'id',None),'received',ident))
            vectors = validate_vectors(response,len(texts))
            if key is not None:
                with tempfile.NamedTemporaryFile(mode='wb',dir=self.cache,delete=False) as f:
                    f.write(orjson.dumps(vectors)); temporary = Path(f.name)
                temporary.replace(path)
            return vectors
        except Exception as error:
            # Unknown outcomes retain their reservation, including process interruption.
            with usage_db(self.data_dir) as db:
                db.execute('UPDATE requests SET state=?,error=? WHERE id=?',('failed',str(error)[:1000],ident))
                if getattr(error,'status_code',None) in (400,401,402,403,404,413,422,429):
                    db.execute('UPDATE requests SET cost_usd=0 WHERE id=? AND cost_usd IS NULL',(ident,))
            raise

    def forget(self, texts, key):
        self.cache_path(key,texts).unlink(missing_ok=True)
