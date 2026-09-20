# papers2 — OpenRouter embedding cost estimate

## Decision / current state

- Use the project's existing `qwen/qwen3-embedding-8b` OpenRouter adapter, with **4,096-dimensional vectors**.
- MiniLM ingestion stopped; automatic restart disabled; `papers2_minilm_v1` deleted (695,841 vectors).
- OpenRouter ingestion authorized and started September 19, 2026; PDF conversion continues.
- First live batch: 187,811 input tokens, 1,000 vectors, $0.00187811. Use `papers-chroma-status` for current cumulative spend. xAI uploads remain off.
- Live settings: 80 workers, batch size 1,000, persistent $5 ledger limit; Nebius/DeepInfra routing capped at $0.01/M.
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

## Full-archive measurement

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
intervals**. Conversion/recovery is unfinished. A text-encoding audit flagged
412 outputs for review (flags do not prove every output is unusable); repair can
change the final counts. The indexer skips outputs failing its encoding-quality
gate; recalculate after repairs.

Exact measured snapshot: [papers2-openrouter-estimate-2026-09-19.json](papers2-openrouter-estimate-2026-09-19.json).

## Chroma RAM / disk at 4,096 dimensions

| Chunk tokens / overlap | FP32 vectors alone | RAM heuristic | Suggested RAM allocation | Approximate disk planning, before extra headroom |
| --- | ---: | ---: | ---: | ---: |
| 175 / 20 | 30.20–30.28 GiB | 62.40–62.56 GiB | 96 GiB | 60–121 GiB plus text/metadata/WAL |
| 800 / 100 | 6.74–6.75 GiB | 15.47–15.51 GiB | 32 GiB | 14–27 GiB plus text/metadata/WAL |

**bigboy.local has enough RAM:** 372.9 GiB total, approximately 257 GiB available
at the latest check. The Chroma store is now on
`/srv/storage/rust1/hpluslogs-papers-chroma`, with about **14 TiB free** at the
September 19 deployment check. The conversion root remains on the system disk;
rust2 is not used.
Disk planning uses 2–4× vector payload; actual Chroma usage varies with index,
documents, metadata and WAL. [Chroma resource guidance](https://cookbook.chromadb.dev/core/resources/).

## Billing interpretation

- Embedding the existing archive is a **one-time input-token charge**, not a monthly subscription.
- Unchanged paper embeddings can be reused. Updates/new papers and query embeddings incur additional token charges.
- Local Chroma has no hosted vector-storage subscription. Machine storage/electricity and answer-generation costs are separate.
- Chunk overlap repeats input tokens. Provider retries and credit-purchase fees are not included.
- Conversion and text-quality repairs are still in progress; final Markdown/token totals can change.

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
  --billing-tokenizer "$PAPERS_TOKENIZER" --workers 8 \
  --prices-per-million 0.01 0.04 \
  > "$PAPERS_DATA/papers2_openrouter_estimate.json"
python -m json.tool "$PAPERS_DATA/papers2_openrouter_estimate.json"
```

Uses the project's `o200k_base` chunk-count convention and the pinned Qwen
tokenizer for pre-overlap input counts. Overlap billing uses each paper's observed
tokenizer ratio plus one special token per chunk. This is an estimate, not a
provider invoice. Only tokenizer files are downloaded; no embedding API calls.

## Recalculate on the current remote deployment

```bash
PAPERS_HOST=bigboy.local
PAPERS_USER=kanzure
PAPERS_PATH=/home/kanzure/hpluslogs-papers-conversion
PAPERS_CONTAINER="hpluslogs-papers-$(python -c 'import hashlib,sys; print(hashlib.sha256(sys.argv[1].encode()).hexdigest()[:12])' "$PAPERS_PATH")"
ssh "$PAPERS_USER@$PAPERS_HOST" "docker run --rm --network none --read-only --cpus 8 --memory 16g --user \$(id -u):\$(id -g) --mount type=bind,src=$PAPERS_PATH/data,dst=/data,readonly --entrypoint python ${PAPERS_CONTAINER}-estimate:latest /app/hpluslogs/scripts/estimate_papers_chroma.py --data-dir /data --dimensions 4096 --billing-tokenizer /app/qwen3-tokenizer.json --workers 8 --prices-per-million 0.01 0.04 > '$PAPERS_PATH/data/papers2_openrouter_estimate.json'"
ssh "$PAPERS_USER@$PAPERS_HOST" "cat '$PAPERS_PATH/data/papers2_openrouter_estimate.json'"
```

Separate xAI estimate: [papers2-xai-costs.md](papers2-xai-costs.md).
