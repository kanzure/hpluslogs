# papers2 — OpenRouter embedding cost estimate

## Observed full usable-archive ingestion — September 20, 2026

| Completed indexing | Value |
| --- | ---: |
| Usable papers / stored vectors | 11,324 / 1,876,298 |
| Recorded embedding input tokens | 347,441,606 |
| Recorded cumulative embedding cost | **$3.47441606 once** |
| Unresolved reservation from an earlier request | $0.00141643 |
| Cumulative ledger limit | $5 |

Measured after ingestion, before the final two answer checks. The ledger includes
earlier query embeddings. Later queries/new or changed papers add usage. The
reservation is an unknown billing outcome, not a confirmed charge. All provider
failures were resolved; 413 unreadable outputs and 46 unconverted PDFs remain
deferred. Local Chroma storage has no provider monthly subscription charge.


## Measured usable-corpus estimate — September 19, 2026, 23:45 CDT

**$3.47 total one-time embedding estimate**, including work already performed.
The previous size-scaled estimate was $3.49. The cumulative ledger limit stays $5.

| Full usable Markdown measurement | Value |
| --- | ---: |
| Files / size | 11,324 / 1,132,819,233 bytes (1.055 GiB) |
| Project tokenizer, before overlap | 290,191,980 tokens |
| Qwen tokenizer, before overlap | 306,200,108 tokens |
| Estimated submitted Qwen tokens, overlap included | 347,441,837 |
| 175/20 token windows / expected vectors | 1,876,417 |
| Embedding rate | $0.01 / million input tokens |
| One-time embedding estimate | **$3.4744** |
| FP32 vectors alone | 28.63 GiB |
| Heuristic Chroma RAM | 59.26 GiB; deployed limit 96 GiB |

All Markdown hashes were checked. The measurement excludes 413 unreadable outputs
and 46 unconverted PDFs. Billing still estimates overlap from tokenizer ratios
and one special token per chunk; provider tokenization, retries and query inputs
can change actual charges. Whitespace-only windows can slightly reduce vector
counts. [Measured report](papers2-openrouter-usable-estimate-2026-09-19.json).

## Previous size-based estimate — September 19, 2026

**About $3.49 total, once, for the current usable Markdown scope.** This includes
embedding spend already incurred; it is not an additional fee or a monthly bill.
The $5 cumulative ledger limit remains in place.

| Scope | Markdown files | Markdown size | Estimated input tokens, including overlap | Estimated vectors | One-time OpenRouter cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Usable Markdown being indexed | 11,324 | 1.055 GiB | 348.55 million | 1.88 million | **$3.49** |
| All generated Markdown, including 413 garbled outputs currently skipped | 11,737 | 1.111 GiB | 367.10 million | 1.98 million | **$3.67** |

Calculated by scaling the previous near-complete Qwen token measurement by the
final Markdown byte totals, using 175/20 chunks and $0.01/M tokens. This is an
approximation, not a fresh tokenizer pass or a provider quote; changed text mix,
retries and query embeddings can change the final charge. The 46 unconverted PDFs
are excluded. The ledger's roughly $0.32 at the time of this update covered all
requests so far, not only the large handbook finishing during the worker upgrade.

[Calculation snapshot](papers2-openrouter-size-estimate-2026-09-19.json).
Nebius and DeepInfra rates rechecked against the public endpoint API for this update.
Local Chroma has no hosted monthly storage charge. For the separate xAI alternative,
see [monthly storage estimates](papers2-xai-costs.md): about $3.96/month for usable
Markdown or $4.17/month for all generated Markdown **if index size equals Markdown**,
plus search/model charges. xAI uploads remain off.

```bash
python - <<'PYTHON'
import json
from pathlib import Path
p = Path('hpluslogs/docs')
b = json.loads((p/'papers2-openrouter-estimate-2026-09-19.json').read_text())
c = next(x for x in b['projections'] if x['chunk_size'] == 175)
for label, size in [('usable', 1132819233), ('all generated', 1193113963)]:
    tokens = c['estimated_measured_input_tokens'] * size / b['markdown_bytes']
    print(label, round(tokens), 'estimated input tokens;', round(tokens / 1e6 * .01, 2), 'USD once')
PYTHON
```

## Decision / current state

- Use the project's existing `qwen/qwen3-embedding-8b` OpenRouter adapter, with **4,096-dimensional vectors**.
- MiniLM ingestion stopped; automatic restart disabled; `papers2_minilm_v1` deleted (695,841 vectors).
- OpenRouter ingestion authorized and started September 19, 2026; the PDF conversion pass is finished and OCR recovery is deferred.
- First live batch: 187,811 input tokens, 1,000 vectors, $0.00187811. Use `papers-chroma-status` for current cumulative spend. xAI uploads remain off.
- Live settings: 80 workers, batch size 128, persistent $5 ledger limit; Nebius/DeepInfra routing capped at $0.01/M.
- [Deployment, status and query commands](papers2-local-rag.md).

## Verified input-token prices — September 19, 2026

| Provider | USD / million input tokens |
| --- | ---: |
| Nebius | $0.01 |
| DeepInfra | $0.01 |
| SiliconFlow | $0.04 |

[OpenRouter model pricing](https://openrouter.ai/qwen/qwen3-embedding-8b),
[public endpoint prices](https://openrouter.ai/api/v1/models/qwen/qwen3-embedding-8b/endpoints),
[Qwen model dimensions](https://huggingface.co/Qwen/Qwen3-Embedding-8B).

The existing adapter allows OpenRouter to route requests; the cheapest price is
not a guaranteed cap unless provider routing/pricing is constrained.

## Earlier full-archive measurement

**Estimate: about $3.70 once using Nebius/DeepInfra, or $14.80 once at
SiliconFlow's rate.** Suggested allowance: $5 with low-price routing enforced,
or $20 if higher-price fallback is allowed. This is a planning allowance, not
authorization to start paid ingestion.

Measured 2026-09-19 22:25 CDT / 2026-09-20 03:25 UTC:

| Input | Measured |
| --- | ---: |
| Full source inventory | 11,783 PDFs |
| Converted PDFs measured | 11,675 |
| Markdown | 1,180,702,107 bytes / 1.100 GiB |
| Project tokenizer (`o200k_base`), before overlap | 303,289,165 tokens |
| Qwen tokenizer, before overlap | 320,160,527 tokens |

| Chunk tokens / overlap | Full-archive embeddings | Estimated billed input tokens | Once at $0.01/M | Once at $0.04/M |
| --- | ---: | ---: | ---: | ---: |
| **175 / 20** — project's IRC-sized configuration | **1,979,168–1,984,323** | 366.64–367.60 million | **$3.67–$3.68** | **$14.67–$14.70** |
| 800 / 100 — larger paper passages | 441,513–442,663 | 368.93–369.89 million | $3.69–$3.70 | $14.76–$14.80 |

One embedding per chunk, not per paper. The two configurations send similar
total text because their overlap fractions are similar; vector counts differ
by about 4.5×. PDF papers use fixed token windows for this estimate; the IRC
implementation also respects message boundaries.

Ranges extrapolate by paper count and PDF bytes; they are **not confidence
intervals**. Conversion/recovery was unfinished at that measurement. The final
text-encoding check excluded 413 outputs from the current run. OCR recovery is
deferred; the updated size-based estimate above uses the completed Markdown totals.

Exact measured snapshot: [papers2-openrouter-estimate-2026-09-19.json](papers2-openrouter-estimate-2026-09-19.json).

## Chroma RAM / disk at 4,096 dimensions

| Chunk tokens / overlap | FP32 vectors alone | RAM heuristic | Suggested RAM allocation | Approximate disk planning, before extra headroom |
| --- | ---: | ---: | ---: | ---: |
| 175 / 20 | 30.20–30.28 GiB | 62.40–62.56 GiB | 96 GiB | 60–121 GiB plus text/metadata/WAL |
| 800 / 100 | 6.74–6.75 GiB | 15.47–15.51 GiB | 32 GiB | 14–27 GiB plus text/metadata/WAL |

**bigboy.local has enough RAM:** 372.9 GiB total, approximately 257 GiB available
at the latest check. The Chroma store is now on
`/srv/storage/disk01/hpluslogs-papers-chroma`, with about **5.1 TiB free** at the
September 19 deployment check. The conversion root remains on the system disk;
rust2 is not used.
Disk planning uses 2–4× vector payload; actual Chroma usage varies with index,
documents, metadata and WAL. [Chroma resource guidance](https://cookbook.chromadb.dev/core/resources/).

## Billing interpretation

- Embedding the existing archive is a **one-time input-token charge**, not a monthly subscription.
- Unchanged paper embeddings can be reused. Updates/new papers and query embeddings incur additional token charges.
- Local Chroma has no hosted vector-storage subscription. Machine storage/electricity and answer-generation costs are separate.
- Chunk overlap repeats input tokens. Provider retries and credit-purchase fees are not included.
- The conversion pass is finished; OCR recovery is deferred. New or repaired Markdown changes future totals.

```text
Embedding cost = submitted input tokens / 1,000,000 × provider price
Raw FP32 vector GiB = vector count × 4096 × 4 / 2^30
Planning Chroma RAM GiB = 2 + 2 × raw vector GiB (heuristic)
```

## Recalculate locally from a current Markdown checkpoint

```bash
source hpluslogs/venv/bin/activate
PAPERS_DATA=/path/to/current/data
PAPERS_TOKENIZER="$PAPERS_DATA/qwen3-embedding-8b-tokenizer.json"
curl -fL 'https://huggingface.co/Qwen/Qwen3-Embedding-8B/resolve/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af/tokenizer.json' -o "$PAPERS_TOKENIZER"
python hpluslogs/scripts/estimate_papers_chroma.py \
  --data-dir "$PAPERS_DATA" --dimensions 4096 \
  --billing-tokenizer "$PAPERS_TOKENIZER" --workers 8 --skip-unreadable \
  --prices-per-million 0.01 0.04 \
  > "$PAPERS_DATA/papers2_usable_token_estimate.json"
python -m json.tool "$PAPERS_DATA/papers2_usable_token_estimate.json"
```

Uses the project's `o200k_base` chunk-count convention and the pinned Qwen
tokenizer for pre-overlap input counts. Overlap billing uses each paper's observed
tokenizer ratio plus one special token per chunk. This is an estimate, not a
provider invoice. `--skip-unreadable` verifies Markdown hashes and measures the
current usable scope without extrapolating to deferred PDFs. Only tokenizer files
are downloaded; no embedding API calls.

## Recalculate on the current remote deployment

```bash
PAPERS_HOST=bigboy.local
PAPERS_USER=kanzure
PAPERS_PATH=/home/kanzure/hpluslogs-papers-conversion
PAPERS_CONTAINER="hpluslogs-papers-$(python -c 'import hashlib,sys; print(hashlib.sha256(sys.argv[1].encode()).hexdigest()[:12])' "$PAPERS_PATH")"
ssh "$PAPERS_USER@$PAPERS_HOST" "docker run --rm --network none --read-only --cpus 8 --memory 16g --user \$(id -u):\$(id -g) --mount type=bind,src=$PAPERS_PATH/data,dst=/data,readonly --entrypoint python ${PAPERS_CONTAINER}-estimate:latest /app/hpluslogs/scripts/estimate_papers_chroma.py --data-dir /data --dimensions 4096 --billing-tokenizer /app/qwen3-tokenizer.json --workers 8 --skip-unreadable --prices-per-million 0.01 0.04 > '$PAPERS_PATH/data/papers2_usable_token_estimate.json'"
ssh "$PAPERS_USER@$PAPERS_HOST" "cat '$PAPERS_PATH/data/papers2_usable_token_estimate.json'"
```

Separate xAI estimate: [papers2-xai-costs.md](papers2-xai-costs.md).
