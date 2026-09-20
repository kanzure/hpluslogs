# papers2 local Chroma RAM estimate

## Planning answer

For **800-token chunks with 100-token overlap** and **4,096-dimensional vectors**,
budget **16–32 GiB RAM for Chroma**, with **32 GiB providing more room for indexing
and queries**. This excludes the embedding model and answer-generation model.

Using the existing IRC-sized **175-token chunks with 20-token overlap** instead
raises the estimate substantially: budget **64–96 GiB** at 4,096 dimensions.

These are capacity estimates, not benchmarked peak resident memory. Local
Chroma ingestion/query support for the paper archive has not been implemented;
the current `papers-query` command uses xAI.

## Capacity of the current conversion host

**Yes: `bigboy.local` has enough RAM for the projected Chroma database.**
Measured on 2026-09-19 (America/Chicago):

| Host memory | Value |
| --- | ---: |
| Physical RAM | 400,362,168,320 bytes / 372.9 GiB |
| Available RAM at inspection | 302,392,160,256 bytes / 281.6 GiB |
| Chroma allocation, 800/100 chunks and 4,096 dimensions | 16–32 GiB |
| Chroma allocation, 175/20 chunks and 4,096 dimensions | 64–96 GiB |

Both configurations fit comfortably at the observed available-memory level.
Availability changes with other workloads, especially parallel PDF conversion.
This is database capacity, not a guarantee that arbitrary embedding/answering
models and concurrent jobs will all fit. Recheck before launching additional jobs:

```bash
ssh "$PAPERS_USER@$PAPERS_HOST" 'free -h'
```

## Measured input — 2026-09-20 01:20 UTC

| Measurement | Value |
| --- | ---: |
| Full source inventory | 11,783 PDFs |
| Successfully converted subset | 2,009 PDFs |
| Markdown measured | 141,324,532 bytes / 134.78 MiB |
| Tokens measured with `o200k_base` | 35,107,608 |
| Observed Markdown bytes/token | 4.025 |
| Measured 800/100 chunks | 50,904 |
| Measured 175/20 chunks | 227,258 |

No embeddings or database were created. Each paper is tokenized separately;
chunks never cross paper boundaries. This estimates the chunked corpus, not
one embedding per whole paper. `o200k_base` matches the project's chunking/counting
convention; it is not the Qwen tokenizer or an embedding-provider billing count.

Full-archive projections extrapolate by paper count and by source PDF bytes.
They cover all 11,783 papers in theory, including ones still awaiting conversion.
Successful PDFs form a biased sample, so the range is not a confidence interval.

## Full collection estimates

| Chunk / overlap | Dimensions | Projected vectors | FP32 vectors alone | Planning RAM* | Suggested allocation |
| --- | ---: | ---: | ---: | ---: | ---: |
| 800 / 100 | 1,024 | 298,558–392,504 | 1.14–1.50 GiB | 4.28–4.99 GiB | 8–16 GiB |
| 800 / 100 | 4,096 | 298,558–392,504 | 4.56–5.99 GiB | 11.11–13.98 GiB | 16–32 GiB |
| 175 / 20 | 1,024 | 1,332,893–1,752,312 | 5.08–6.68 GiB | 12.17–15.37 GiB | 24–32 GiB |
| 175 / 20 | 4,096 | 1,332,893–1,752,312 | 20.34–26.74 GiB | 42.68–55.48 GiB | 64–96 GiB |

```text
Per-paper chunks = 1 + ceil(max(0, tokens - chunk_size) / (chunk_size - overlap))
Vector payload GiB = vectors × dimensions × 4 / 2^30
* Planning RAM GiB = 2 + 2 × vector payload GiB
```

The planning multiplier/base allowance and suggested allocations are our
heuristics. They allow for graph links, process overhead, caches and working
memory; actual demand depends on Chroma/HNSW configuration and concurrency.
Chroma's [resource guidance](https://cookbook.chromadb.dev/core/resources/)
identifies FP32 payload as a lower bound, with additional index, buffer and query
memory. Its disk guidance is roughly 2–4× vector payload in many workloads,
with additional space potentially needed for documents, metadata and WAL.

## Embedding dimensions and model memory

The project's current embedding adapter defaults to `qwen/qwen3-embedding-8b`.
[Qwen's model card](https://huggingface.co/Qwen/Qwen3-Embedding-8B) specifies a
4,096-dimensional output and support for reduced dimensions. The 1,024-dimensional
scenario requires explicitly changing/validating embedding output; setting a
Chroma option alone will not shrink existing vectors.

Running an 8-billion-parameter model locally requires about **14.9 GiB for BF16
weights alone** (`8e9 × 2 / 2^30`), plus runtime/activation memory. Quantization
changes this. The answering LLM adds its own RAM/VRAM requirements. If embeddings
and answers remain API-hosted, those model weights do not occupy local RAM.

The existing IRC embedding service loads all pending chunks and permits high
concurrency. A future paper implementation should stream bounded batches;
otherwise client-side ingestion memory can exceed the database-only estimate.
Keep papers in a separate collection/index with paper citation metadata.

## Recompute after pulling more Markdown

```bash
python hpluslogs/scripts/estimate_papers_chroma.py \
  --data-dir "$PAPERS_DATA" --dimensions 1024 4096 \
  > "$PAPERS_DATA/papers2_chroma_estimate.json"
python -m json.tool "$PAPERS_DATA/papers2_chroma_estimate.json"
```

Runbook/setup: [papers2-commands.md](papers2-commands.md).
xAI alternative: [papers2-xai-costs.md](papers2-xai-costs.md).

## Local deployment implemented September 19, 2026

The running paper index uses CPU MiniLM-L6-v2, 384-dimensional vectors and
224-wordpiece chunks with 32 overlap. The Qwen scenarios above remain sizing
alternatives, not the deployed model. Chroma has a 32 GiB container memory ceiling;
the four-worker embedding process has a 16 GiB ceiling. Initial observed embedding
RAM was about 1.7 GiB. See [local RAG commands](papers2-local-rag.md).
