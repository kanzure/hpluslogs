#!/usr/bin/env python3
"""Read-only Chroma capacity estimate from converted papers; no embedding/API calls."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import sys

import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hpluslogs.services import papers


def chunk_count(tokens, size, overlap):
    if not 0 <= overlap < size:
        raise ValueError('Chunk overlap must be smaller than chunk size.')
    return 0 if not tokens else 1 + math.ceil(max(0, tokens - size) / (size - overlap))


def estimate(data_dir, encoding='o200k_base', dimensions=(1024, 4096)):
    sources = {papers.path_key(p.relative_to(data_dir/'papers2').as_posix()): p.stat().st_size
               for p in papers.pdf_files(data_dir/'papers2')}
    manifest = data_dir/'papers2_markdown.sqlite3'
    with sqlite3.connect(f'{manifest.resolve().as_uri()}?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(r) for r in db.execute("SELECT * FROM conversions WHERE status='ready'")]
    tokenizer = tiktoken.get_encoding(encoding)
    configs = [(800, 100), (175, 20)]
    counts = {c: 0 for c in configs}
    ready = size = source_bytes = tokens = 0
    for row in rows:
        path = data_dir/'papers2_markdown'/row['markdown_path']
        if (sources.get(row['pdf_path']) != row['pdf_bytes'] or not path.is_file()
                or path.is_symlink() or path.stat().st_size != row['markdown_bytes']):
            continue
        n = len(tokenizer.encode(path.read_text(encoding='utf-8'), disallowed_special=()))
        tokens += n; ready += 1; size += row['markdown_bytes']; source_bytes += row['pdf_bytes']
        for config in configs:
            counts[config] += chunk_count(n, *config)
    if not ready:
        raise ValueError('No ready Markdown files to measure.')
    projections = []
    for (chunk_size, overlap), count in counts.items():
        low, high = sorted([math.ceil(count * len(sources) / ready),
                            math.ceil(count * sum(sources.values()) / source_bytes)])
        for dim in dimensions:
            payload = [n * dim * 4 / 2**30 for n in (low, high)]
            projections.append({'chunk_size': chunk_size, 'overlap': overlap, 'dimensions': dim,
                                'measured_chunks': count, 'full_archive_vectors_range': [low, high],
                                'fp32_payload_gib_range': payload,
                                'planning_ram_gib_range': [2 + 2 * value for value in payload]})
    return {'measured_at': datetime.now(timezone.utc).isoformat(), 'pdf_count': len(sources),
            'ready_markdown_count': ready, 'markdown_bytes': size,
            'converted_source_pdf_bytes': source_bytes, 'total_source_pdf_bytes': sum(sources.values()),
            'tokenizer': encoding, 'measured_tokens': tokens, 'bytes_per_token': size / tokens if tokens else None,
            'projections': projections,
            'assumptions': ['Fixed token windows, one vector per chunk; no chunk crosses a paper boundary.',
                            'Range spans extrapolation by paper count and source PDF bytes, not a confidence interval.',
                            'Planning RAM = 2 GiB base allowance + 2x FP32 vector payload; a heuristic, not measured RSS.',
                            'Excludes embedding/generation model RAM or VRAM and large ingestion batches.',
                            'o200k_base is the project chunking tokenizer, not the Qwen tokenizer.',
                            'Counts size-matching ready outputs; no source hashes rechecked and no embeddings generated.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('hpluslogs/data'))
    parser.add_argument('--encoding', default='o200k_base')
    parser.add_argument('--dimensions', nargs='+', type=int, default=[1024, 4096])
    args = parser.parse_args()
    if any(d < 1 for d in args.dimensions):
        parser.error('Dimensions must be positive.')
    print(json.dumps(estimate(args.data_dir, args.encoding, args.dimensions), indent=2))


if __name__ == '__main__':
    main()
