#!/usr/bin/env python3
"""Read-only Chroma capacity estimate from converted papers; no embedding/API calls."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import hashlib
import math
from pathlib import Path
import re
import sqlite3
import sys

import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hpluslogs.services import papers


def initialize_tokenizers(encoding, billing_path, skip_unreadable=False):
    global _chunk_tokenizer, _billing_tokenizer, _skip_unreadable
    _skip_unreadable = skip_unreadable
    _chunk_tokenizer = tiktoken.get_encoding(encoding)
    _billing_tokenizer = None
    if billing_path:
        from tokenizers import Tokenizer
        _billing_tokenizer = Tokenizer.from_file(str(billing_path))
        _billing_tokenizer.no_padding(); _billing_tokenizer.no_truncation()


def measure_file(item):
    row, path = item
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != row['markdown_sha256']:
        raise ValueError(f'Markdown changed since conversion: {row["pdf_path"]}')
    text = raw.decode('utf-8')
    if _skip_unreadable:
        suspect = sum(1 for _ in re.finditer('[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ufffd]', text))
        if suspect >= 32 and suspect/max(1,len(text)) > .01:
            return row, None, None
    n = len(_chunk_tokenizer.encode(text, disallowed_special=()))
    billed = len(_billing_tokenizer.encode(text, add_special_tokens=False).ids) if _billing_tokenizer else n
    return row, n, billed


def chunk_count(tokens, size, overlap):
    if not 0 <= overlap < size:
        raise ValueError('Chunk overlap must be smaller than chunk size.')
    return 0 if not tokens else 1 + math.ceil(max(0, tokens - size) / (size - overlap))


def estimate(data_dir, encoding='o200k_base', dimensions=(1024, 4096), billing_tokenizer=None,
             prices=(0.01, 0.04), workers=4, skip_unreadable=False):
    sources = {papers.path_key(p.relative_to(data_dir/'papers2').as_posix()): p.stat().st_size
               for p in papers.pdf_files(data_dir/'papers2')}
    manifest = data_dir/'papers2_markdown.sqlite3'
    with sqlite3.connect(f'{manifest.resolve().as_uri()}?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(r) for r in db.execute("SELECT * FROM conversions WHERE status='ready'")]
    configs = [(800, 100), (175, 20)]
    counts = {c: 0 for c in configs}
    submitted = {c: 0 for c in configs}
    billing_total = 0
    ready = size = source_bytes = tokens = 0
    inputs = []
    skipped = []
    for row in rows:
        path = data_dir/'papers2_markdown'/row['markdown_path']
        if (sources.get(row['pdf_path']) != row['pdf_bytes'] or not path.is_file()
                or path.is_symlink() or path.stat().st_size != row['markdown_bytes']):
            continue
        inputs.append((row,path))
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_tokenizers,
                             initargs=(encoding,billing_tokenizer,skip_unreadable)) as pool:
        for row, n, billed in pool.map(measure_file, inputs, chunksize=8):
            if n is None:
                skipped.append(row['pdf_path'])
                continue
            tokens += n; billing_total += billed; ready += 1
            size += row['markdown_bytes']; source_bytes += row['pdf_bytes']
            for config in configs:
                count = chunk_count(n, *config)
                counts[config] += count
                submitted[config] += estimated_input_tokens(n, billed, *config)
            if ready % 1000 == 0:
                print(f'Measured {ready}/{len(inputs)} papers', file=sys.stderr, flush=True)
    if not ready:
        raise ValueError('No ready Markdown files to measure.')
    projections = []
    for (chunk_size, overlap), count in counts.items():
        low, high = sorted([math.ceil(count * len(sources) / ready),
                            math.ceil(count * sum(sources.values()) / source_bytes)])
        input_range = sorted([math.ceil(submitted[(chunk_size,overlap)] * len(sources) / ready),
                              math.ceil(submitted[(chunk_size,overlap)] * sum(sources.values()) / source_bytes)])
        if skip_unreadable:
            low = high = count
            input_range = [submitted[(chunk_size,overlap)]]*2
        for dim in dimensions:
            payload = [n * dim * 4 / 2**30 for n in (low, high)]
            projections.append({'chunk_size': chunk_size, 'overlap': overlap, 'dimensions': dim,
                                'measured_chunks': count, 'full_archive_vectors_range': [low, high],
                                'estimated_measured_input_tokens': submitted[(chunk_size,overlap)],
                                'full_archive_input_tokens_range': input_range,
                                'one_time_embedding_cost_usd': [
                                    {'price_per_million':price,'range':[n/1e6*price for n in input_range]}
                                    for price in prices],
                                'fp32_payload_gib_range': payload,
                                'planning_ram_gib_range': [2 + 2 * value for value in payload]})
    return {'measured_at': datetime.now(timezone.utc).isoformat(), 'pdf_count': len(sources),
            'projection_scope': 'usable ready Markdown only' if skip_unreadable else 'all source PDFs, extrapolated',
            'skipped_unreadable_markdown': skipped,
            'ready_markdown_count': ready, 'markdown_bytes': size,
            'converted_source_pdf_bytes': source_bytes, 'total_source_pdf_bytes': sum(sources.values()),
            'tokenizer': encoding, 'measured_tokens': tokens, 'bytes_per_token': size / tokens if tokens else None,
            'billing_tokenizer': str(billing_tokenizer) if billing_tokenizer else encoding,
            'measured_billing_tokens_before_overlap': billing_total,
            'projections': projections,
            'assumptions': ['Fixed token windows, one vector per chunk; no chunk crosses a paper boundary.',
                            'Range spans extrapolation by paper count and source PDF bytes, not a confidence interval.',
                            'Planning RAM = 2 GiB base allowance + 2x FP32 vector payload; a heuristic, not measured RSS.',
                            'Excludes embedding/generation model RAM or VRAM and large ingestion batches.',
                            'o200k_base is the project chunking tokenizer, not the Qwen tokenizer.',
                            'Billing-tokenizer counts are measured before overlap; overlap is estimated per paper using its tokenizer ratio, plus one special token per chunk.',
                            'Provider tokenization, retries, credit-purchase fees and future OCR repairs can change the invoice. These are one-time embedding costs, not monthly storage fees.',
                            'Markdown hashes verified; source PDF sizes checked, not source hashes. No embeddings generated.',
                            'With --skip-unreadable, ranges equal measured usable totals; no extrapolation to excluded/failed PDFs.',
                            'Token-window counts may slightly exceed actual vectors when a window contains only whitespace.']}


def estimated_input_tokens(tokens, billing_tokens, size, overlap):
    count = chunk_count(tokens, size, overlap)
    return math.ceil(billing_tokens * (tokens + max(0,count-1)*overlap) / tokens) + count if tokens else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('hpluslogs/data'))
    parser.add_argument('--encoding', default='o200k_base')
    parser.add_argument('--dimensions', nargs='+', type=int, default=[1024, 4096])
    parser.add_argument('--billing-tokenizer', type=Path, help='Local Qwen tokenizer.json; no model weights/API calls')
    parser.add_argument('--prices-per-million', nargs='+', type=float, default=[0.01,0.04])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--skip-unreadable', action='store_true', help='Measure only usable ready Markdown, without extrapolating to skipped PDFs.')
    args = parser.parse_args()
    if any(d < 1 for d in args.dimensions):
        parser.error('Dimensions must be positive.')
    if args.workers < 1 or any(p < 0 for p in args.prices_per_million):
        parser.error('Workers must be positive; prices must be nonnegative.')
    print(json.dumps(estimate(args.data_dir, args.encoding, args.dimensions,
                             args.billing_tokenizer, args.prices_per_million, args.workers,
                             args.skip_unreadable), indent=2))


if __name__ == '__main__':
    main()
