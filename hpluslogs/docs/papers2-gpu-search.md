# Paper vector search and GPU assessment

Checked September 20, 2026. Recommendation: keep the current CPU Chroma search;
profile embedding/API latency and answer-model prefill/generation before replacing it.
No service configuration, model allocation, embeddings or indexes were changed.

## Chroma support

The deployed Chroma 1.4.0 collection uses CPU HNSW with cosine distance. Current
single-node configuration documents HNSW tuning, with no supported CUDA search
switch. Passing a GPU into this container would not convert its index to GPU search.
[Chroma index configuration](https://docs.trychroma.com/docs/collections/configure).

Chroma's GPU examples accelerate **embedding functions**, a separate operation.
Our embedding function is disabled in Chroma: the project explicitly supplies
4,096-dimensional Qwen vectors from OpenRouter. Enabling a local CUDA embedding
function would not accelerate these existing HNSW searches.
[Chroma embedding GPU support](https://cookbook.chromadb.dev/embeddings/gpu-support/).

## Observed host capacity

| Measurement | Observation |
| --- | ---: |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition |
| Driver-reported GPU memory | 97,887 MiB / 95.59 GiB |
| GPU memory used / free | 95,273 MiB / 2,614 MiB (about 2.55 GiB free) |
| SGLang scheduler GPU allocation | 90,474 MiB / 88.35 GiB |
| Other GPU allocations | Python 3,938 MiB; SGLang Python 676 MiB |
| System RAM total / available | Approximately 372.9 / 245 GiB |
| Chroma container RAM / limit | 36.17 / 96 GiB |
| Chroma GPU device requests | None |
| Paper vectors / dimensions | 1,876,298 / 4,096 |
| Raw FP32 vector payload | 28.630 GiB |

GPU utilization was 0% at the instant sampled, but its memory remained allocated.
These are point-in-time readings, not reservations or guaranteed availability.
The answer model already uses this GPU. System RAM and GPU VRAM are separate.

## Read-only search timing

Queried the running collection using one existing stored embedding, twice at
each candidate count. Included documents, metadata and distances in the response.
No embedding or answer-model calls were made.

| Chroma candidates | Corresponding application top-k | Client round-trip times |
| ---: | ---: | --- |
| 500 | 100 | 0.0562 s; 0.0336 s |
| 1,500 | 300 | 0.1049 s; 0.0914 s |

Application retrieval requests `5 * top_k` candidates, then filters stale revisions
and limits each paper to two excerpts. These timings cover warm Chroma queries,
not SSH startup, query embedding, application filtering, LLM generation or SCP.
They are a small diagnostic sample, not a recall, cold-start, concurrency or p99
benchmark. At this observed load, GPU search could only save a fraction of a second.

Collection configuration: `space=cosine`, `ef_construction=100`, `ef_search=100`,
`max_neighbors=16`. No tuning was applied during the assessment.

## GPU alternatives if search becomes the bottleneck

- **Faiss GPU / NVIDIA cuVS:** add a GPU search service alongside Chroma, retaining
  Chroma for document/metadata lookup. Export the existing vectors and IDs, build
  the new index, preserve cosine normalization and revision/deletion semantics,
  and keep incremental updates synchronized. This needs application work, but
  does not require paying to regenerate embeddings or reducing their dimensions.
  [Faiss GPU documentation](https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU),
  [NVIDIA cuVS integrations](https://docs.nvidia.com/cuvs/getting-started/integrations).
- **Milvus GPU:** migrate to a database with supported GPU index types, including
  CAGRA, IVF and brute force. This is a database migration, not a Chroma setting.
  [Milvus GPU indexes](https://milvus.io/docs/gpu_index.md).

Raw FP32 vectors need 28.63 GiB plus graph/IDs/workspace. Milvus documents roughly
1.8 times raw vector size for GPU_CAGRA: about **51.5 GiB** for this corpus before
additional headroom, as a planning estimate rather than a measurement. A dedicated
96-GiB card could plausibly accommodate this; the current 2.55-GiB free allocation
cannot. Freeing tens of GiB would require changing other GPU workloads.
[Milvus memory guidance](https://milvus.io/docs/gpu_index.md).

Check result-count limits before choosing an implementation: the Faiss GPU wiki
documents `k <= 2048` for its listed GPU indexes. Our current 1,500-candidate search
fits that, but a 500-passage request would ask for 2,500 candidates. Backend/version
limits must be tested; switching to GPU must not reintroduce an unexpected top-k cap.
[Faiss GPU limitations](https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU).

## Recheck GPU allocation

```bash
ssh "$PAPERS_USER@$PAPERS_HOST" \
  'nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv'
ssh "$PAPERS_USER@$PAPERS_HOST" \
  'nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv'
```

If latency becomes a problem, time query embedding, Chroma retrieval, model prefill,
model generation and publication independently. Consider GPU search only after
representative retrieval benchmarks justify the memory and integration costs.
