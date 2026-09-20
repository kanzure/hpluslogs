# Local Chroma paper RAG

## Verified archive — September 20, 2026

- **11,324 usable papers; 1,876,298 vectors; 4,096 dimensions.**
- Full passage audit: stable snapshot, zero errors. [Saved audit](papers2-chroma-audit-2026-09-20.json).
- 413 unreadable Markdown outputs and 46 unconverted PDFs deferred; OCR disabled.
- Two cited-answer queries verified against the completed collection; commands below.
- Watcher restart verified: zero new embedding requests for unchanged papers.
- [Observed embedding cost](papers2-openrouter-costs.md): **$3.47 once**.

## OpenRouter Qwen + local Chroma

Qwen3 Embedding 8B, 4,096 dimensions; 175-token chunks with 20-token overlap.
OpenRouter requests use the project client, 80 concurrent workers, batches of up
to 128 chunks, and a persistent $5 cumulative embedding ledger limit. Routing
allows Nebius/DeepInfra at no more than $0.01 per million input tokens.
MiniLM collection `papers2_minilm_v1` was deleted and is not used.
See [OpenRouter costs](papers2-openrouter-costs.md).

Current deployment uses 128-chunk batches after repeated provider `engine_overloaded`
responses with 1,000-chunk batches. Worker concurrency remains 80. Completed
checkpoints are reused; the embedding model, dimensions and chunk boundaries stay
the same. The CLI accepts `--batch-size` to tune request size.

## Remote parameters

```bash
source hpluslogs/venv/bin/activate
PAPERS_HOST=conversion-server.example
PAPERS_USER=your-user
PAPERS_PATH=/home/your-user/hpluslogs-papers-conversion
PAPERS_TARGET="$PAPERS_USER@$PAPERS_HOST"
PAPERS_REMOTE=(--host "$PAPERS_HOST" --user "$PAPERS_USER" --path "$PAPERS_PATH")
PAPERS_CONTAINER="hpluslogs-papers-$(python -c 'import hashlib,sys; from pathlib import PurePosixPath; print(hashlib.sha256(str(PurePosixPath(sys.argv[1])).encode()).hexdigest()[:12])' "$PAPERS_PATH")"
CHROMA_CONTAINER="${PAPERS_CONTAINER}-chroma"
PAPERS_LLM_MODEL=your-local-served-model-name
CHROMA_DATA_PATH=/srv/storage/disk01/hpluslogs-papers-chroma
PAPERS_ENV_FILE="$PAPERS_PATH/config/openrouter.env"
```

## Private credentials on remote host

Create `$PAPERS_ENV_FILE` with `OPENROUTER_API_KEY=...`, mode 600.
Existing deployments retain this file; do not commit it.

## Deploy / resume indexing

```bash
python hpluslogs/scripts/deploy_papers_chroma.py "${PAPERS_REMOTE[@]}" \
  --concurrency 80 --batch-size 128 --cost-limit 5 --port 18081 \
  --chroma-data-path "$CHROMA_DATA_PATH" --env-file "$PAPERS_ENV_FILE"
```

Chroma listens on remote loopback only. Its vectors, documents and index are
stored at `CHROMA_DATA_PATH` on the SSD. Conversion data and resumable checkpoints
remain under `$PAPERS_PATH/data`. No rust2 storage is used.
The indexer watches completed Markdown every 120 seconds; both containers restart
after reboot. Hash checkpoints and cached embedding batches preserve progress.
Outputs failing the text-encoding quality gate are skipped for extraction repair.

## Direct hpluslogs CLI

```bash
# With OPENROUTER_API_KEY exported and the Markdown manifest present:
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-chroma-index \
  --chroma-port 18081 --concurrency 80 --batch-size 128 --cost-limit 5 --watch
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-chroma-status --chroma-port 18081
```

Do not run a second indexer against the same data directory. The deployment
container runs the same registered commands through `hpluslogs.papers_chroma_cli`.

## Progress / indexing ETA / errors / resources

```bash
ssh "$PAPERS_TARGET" "docker exec ${CHROMA_CONTAINER}-index python -m hpluslogs.papers_chroma_cli papers-chroma-status"
ssh "$PAPERS_TARGET" "docker logs --tail 15 ${CHROMA_CONTAINER}-index"
ssh "$PAPERS_TARGET" "docker stats --no-stream $PAPERS_CONTAINER $CHROMA_CONTAINER ${CHROMA_CONTAINER}-index"
ssh "$PAPERS_TARGET" "docker inspect --format '{{.State.Status}} exit={{.State.ExitCode}}' $PAPERS_CONTAINER"
```

`ready_markdown` versus `papers.ready`: conversion versus indexing coverage.
`current_markdown_backlog_eta_hours`: measured indexing ETA for current Markdown;
excludes PDFs still being converted and repair work.
Indexing can catch up while conversion continues; this does not mean the entire
PDF archive has converted. Failed PDFs require inspection and retries.

## Status with explicit skipped-text counts

```bash
# Uses the latest image; does not interrupt the running indexer:
ssh "$PAPERS_TARGET" "docker run --rm --network host --user \$(id -u):\$(id -g) --mount type=bind,src=$PAPERS_PATH/data,dst=/data ${CHROMA_CONTAINER}:latest papers-chroma-status"
```

`skipped_unreadable_markdown`: deferred text repair. `retryable_or_other_failures`:
API/storage/other failures, retried on subsequent watch passes. No OCR is launched
by these commands. `estimated_chunk_backlog_eta_hours` uses the saved usable-corpus
token measurement and current vector throughput, accounting for large books.
Generate `data/papers2_usable_token_estimate.json` using the invocation in
[papers2-openrouter-costs.md](papers2-openrouter-costs.md). Stale size/count scopes
are rejected; paper-count ETA remains available separately.

## Retrieve source passages

```bash
ssh "$PAPERS_TARGET" "docker exec ${CHROMA_CONTAINER}-index python -m hpluslogs.papers_chroma_cli papers-chroma-query --nollm --top-k 8 'How can DNA molecules be aligned on surfaces?'"
```

## Generate a cited answer using a local model

```bash
ssh "$PAPERS_TARGET" "docker exec ${CHROMA_CONTAINER}-index python -m hpluslogs.papers_chroma_cli papers-chroma-query --llm-url http://127.0.0.1:8080/v1 --model '$PAPERS_LLM_MODEL' --top-k 8 'How can DNA molecules be aligned on surfaces?'"
```

Results and answers: remote `data/papers2_local_queries/*.json`.

## Query from this machine through SSH

```bash
ssh -N -L 18081:127.0.0.1:18081 "$PAPERS_TARGET"
# In another terminal, with the Markdown checkpoint pulled locally:
python -m hpluslogs.cli --data-dir hpluslogs/data papers-chroma-query --chroma-port 18081 --nollm 'What mechanisms enable molecular computation?'
```

## Add PDFs / convert / index

```bash
# See papers2-commands.md for transfer/restore and remote conversion deployment.
python -m hpluslogs.cli --data-dir hpluslogs/data papers-add /path/to/new-paper.pdf --path topic/new-paper.pdf
python -m hpluslogs.cli --data-dir hpluslogs/data papers-remote-deploy "${PAPERS_REMOTE[@]}" --workers 56 --timeout 600 --transfer tar
# The running Chroma watcher picks up completed Markdown automatically.
```

## Pause / resume embedding requests

```bash
ssh "$PAPERS_TARGET" "docker stop ${CHROMA_CONTAINER}-index"
ssh "$PAPERS_TARGET" "docker start ${CHROMA_CONTAINER}-index"
```

## Deployment — September 19, 2026

- Host: `kanzure@bigboy.local`; root: `/home/kanzure/hpluslogs-papers-conversion`.
- Chroma 1.4.0 on `127.0.0.1:18081`; collection `papers2_qwen3_8b_4096_v1`.
- Chroma storage: `/srv/storage/disk01/hpluslogs-papers-chroma`; 96 GiB RAM limit.
- First complete paper: 4,522 stored vectors, 842,914 billed tokens, $0.00842914.
- SSD migration preserved the collection and vectors; the stopped rust1 copy remains a backup.
- `/srv/storage/disk01`: nonrotating 7.3 TiB device, 5.1 TiB free. The dedicated model NVMe had only 77 GiB free, insufficient for the projected store plus headroom.
- Local answer endpoint: `http://127.0.0.1:8080/v1`; model `qwen-flash-next-uncensored-sglang`.
- Old MiniLM query artifacts are historical; they do not validate Qwen coverage.

[Chroma server deployment](https://docs.trychroma.com/guides/deploy/docker).

## Optional OCR retry — currently disabled

These commands are for a future explicitly requested OCR recovery pass.
English Tesseract data is included in the conversion image. A previously empty
scanned PDF produced 10,020 Markdown bytes in a real smoke test. Completed outputs
are preserved; the retry uses the same PyMuPDF4LLM settings. OCR can contain errors.
Empty or invalid source files still require replacement; inspect failures afterward.

```bash
# Build the updated image without stopping conversion or transferring PDFs:
python -m hpluslogs.cli papers-remote-build "${PAPERS_REMOTE[@]}"
# To schedule a retry while the existing converter is still running:
scp hpluslogs/scripts/retry_papers_conversion.py "$PAPERS_TARGET:$PAPERS_PATH/watcher/retry_papers_conversion.py"
ssh "$PAPERS_TARGET" "systemd-run --user --unit=${PAPERS_CONTAINER}-ocr-retry --collect python3 '$PAPERS_PATH/watcher/retry_papers_conversion.py' --config '$PAPERS_PATH/watcher/config.json' --workers 16 --timeout 1800"
ssh "$PAPERS_TARGET" "journalctl --user -u ${PAPERS_CONTAINER}-ocr-retry.service -n 10 --no-pager"
ssh "$PAPERS_TARGET" "cat '$PAPERS_PATH/data/papers2_conversion_retry.json'"
# Before intentionally stopping all conversion, cancel any waiting retry:
ssh "$PAPERS_TARGET" "systemctl --user stop ${PAPERS_CONTAINER}-ocr-retry.service"
python -m hpluslogs.cli --data-dir hpluslogs/data papers-remote-stop "${PAPERS_REMOTE[@]}"
```

The retry pins the built image and the observed container ID. If another operator
replaces that container, the watcher stops instead of replacing their new run.
The watcher needs the `watcher/config.json` produced by the restic handoff workflow.
[PyMuPDF4LLM OCR support](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/ocr-plugins.html).

## Verify complete Markdown indexing

Run after conversion finishes and usable Markdown is indexed. With OCR deferred,
`papers.ready + skipped_unreadable_markdown` should equal `ready_markdown`;
retryable API failures must be resolved.
Stop the embedding writer for a stable audit; leave the Chroma server running.
The command verifies every passage against Markdown bytes and offsets, identities,
source URLs and complete chunk counts. It reports incomplete coverage as failure.
Conversion failures remain a separate check. `--allow-skipped` explicitly permits
encoding-quality exclusions, lists every excluded paper, and requires every other
Markdown document and stored passage to pass verification.
Readback uses 20,000-passage pages without embedding payloads. On the completed
index, 500- and 5,000-passage reads both took about 3.2 seconds; larger pages reduce
request overhead while preserving the same per-passage checks.

```bash
ssh "$PAPERS_TARGET" "docker stop ${CHROMA_CONTAINER}-index"
ssh "$PAPERS_TARGET" "docker run --rm --network host --user \$(id -u):\$(id -g) --mount type=bind,src=$PAPERS_PATH/data,dst=/data --cpus 4 --memory 64g --memory-swap 64g ${CHROMA_CONTAINER}:latest papers-chroma-audit --allow-skipped"
ssh "$PAPERS_TARGET" "cat '$PAPERS_PATH/data/papers2_chroma_audit.json'"
ssh "$PAPERS_TARGET" "docker start ${CHROMA_CONTAINER}-index"
```

## Change embedding concurrency

```bash
# Recreates only the embedding worker; preserves stored batches:
python hpluslogs/scripts/deploy_papers_chroma.py "${PAPERS_REMOTE[@]}" \
  --concurrency 80 --batch-size 128 --cost-limit 5 \
  --chroma-data-path "$CHROMA_DATA_PATH" --env-file "$PAPERS_ENV_FILE"
```

## Classify conversion failures

```bash
ssh "$PAPERS_TARGET" "docker run --rm --network none --user \$(id -u):\$(id -g) --cpus 1 --memory 2g --mount type=bind,src=$PAPERS_PATH/data,dst=/data ${PAPERS_CONTAINER}:latest failures"
ssh "$PAPERS_TARGET" "cat '$PAPERS_PATH/data/papers2_failure_audit.json'"
```

Separates empty files, unreadable/zero-page PDFs, mislabeled non-PDFs, password
requirements and readable PDFs whose extraction failed. Includes source hashes
and conversion errors; does not change source files or checkpoints.

## Long-document resume checkpoints

PDFs with 200 or more pages now save 20-page batches in `data/papers2_page_cache`.
Cache keys include source bytes, converter settings and English OCR data. Each
batch is hash checked and written atomically. Timeouts/restarts reuse finished
batches; completed Markdown files are still skipped by the existing manifest.
Keep this directory on the remote host across retries.

```bash
# Count completed page batches (not completed papers):
ssh "$PAPERS_TARGET" "find '$PAPERS_PATH/data/papers2_page_cache' -name '*.json' -type f | wc -l"
# Retry a stopped conversion with more time; preserve remote data/checkpoints:
python -m hpluslogs.cli --data-dir hpluslogs/data papers-remote-deploy "${PAPERS_REMOTE[@]}" --workers 32 --timeout 1800 --transfer tar
```

Verified on 201 pages: 11 saved batches, all page markers retained, identical
Markdown on resume; initial 28.9 seconds, resumed 0.9 seconds. This measures cache
reuse, not general conversion throughput.

Conversion progress includes 5-, 15-, 60- and 120-minute rates. Use the short window
when the remaining work changes from articles to long books; timeouts and OCR
retries can still make the final tail slower than an extrapolated ETA.

## Empty extraction / hidden OCR fallback

Empty or failed layout extraction retries PyMuPDF4LLM's alternate parser with
hidden text enabled. Native segmentation faults retry in a fresh process within
the original deadline. Fallback output can retain headers/footers. Parser provenance:
`data/papers2_extraction_metadata/<markdown-relative-path>.json`.
Existing completed Markdown stays unchanged; old empty page batches are rechecked.
Malformed Unicode is normalized before writing/cache hashing: valid surrogate pairs
become their original character; isolated surrogates become U+FFFD. The same metadata
records these repairs. Verified on WordGesture-GAN: 85,946 Markdown bytes recovered.

Verified recovery: pneumatic stepping motor paper, 21,621 Markdown bytes including
title, abstract and body. Conversion tests: 60 passing.

```bash
# Build while conversion is running, then queue one additional recovery pass:
python -m hpluslogs.cli papers-remote-build "${PAPERS_REMOTE[@]}"
scp hpluslogs/scripts/retry_papers_conversion.py "$PAPERS_TARGET:$PAPERS_PATH/watcher/retry_papers_conversion.py"
ssh "$PAPERS_TARGET" "systemd-run --user --unit=${PAPERS_CONTAINER}-text-retry --collect python3 '$PAPERS_PATH/watcher/retry_papers_conversion.py' --config '$PAPERS_PATH/watcher/config.json' --workers 32 --timeout 1800 --state-name papers2_text_retry"
ssh "$PAPERS_TARGET" "cat '$PAPERS_PATH/data/papers2_text_retry.json'"
ssh "$PAPERS_TARGET" "journalctl --user -u ${PAPERS_CONTAINER}-text-retry.service -n 10 --no-pager"
# Cancel queued recovery before intentionally stopping all conversion:
ssh "$PAPERS_TARGET" "systemctl --user stop ${PAPERS_CONTAINER}-text-retry.service ${PAPERS_CONTAINER}-ocr-retry.service"
```

Use separate state names for successive watchers; do not queue multiple watchers
to replace the same running container. OCR and text-repair watchers are inactive
in the current deployment.

## Optional OCR repair — disabled at user request

OCR recovery is deferred. The repair watcher was canceled before any archive
repair container started. Existing usable Markdown continues indexing; text that
fails the encoding-quality gate remains skipped. The commands below are optional
future invocations, not part of the current run.

```bash
# Build without stopping current conversion or indexing:
python -m hpluslogs.cli papers-remote-build "${PAPERS_REMOTE[@]}"
# Automatically start repair after the current converter exits:
python -m hpluslogs.cli papers-remote-repair "${PAPERS_REMOTE[@]}" --workers 16 --timeout 1800
ssh "$PAPERS_TARGET" "cat '$PAPERS_PATH/data/papers2_repair_watch.json'"
ssh "$PAPERS_TARGET" "docker logs --tail 10 ${PAPERS_CONTAINER}-repair"
ssh "$PAPERS_TARGET" "cat '$PAPERS_PATH/data/papers2_repair_report.json'"
# Local equivalents, once normal conversion has stopped:
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-repair
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-repair --apply --workers 16 --timeout 1800
```

Repair keeps PDFs unchanged, backs up replaced Markdown, and checkpoints each OCR
page. Repeating the command skips repaired text and reuses completed page batches.
The Chroma watcher picks up repaired Markdown on its next pass. OCR can misread
characters; citations should be checked against the PDF when precision matters.

Verified on the 20-page mix-design paper: 47,105 readable Markdown bytes;
42.0 seconds initially, 0.48 seconds resumed, identical SHA-256. Unit checks cover
source changes, failed repair preserving prior output, backups and idempotency.

## Verified Qwen RAG example — September 19, 2026

```bash
ssh "$PAPERS_TARGET" "docker exec ${CHROMA_CONTAINER}-index python -m hpluslogs.papers_chroma_cli papers-chroma-query --model '$PAPERS_LLM_MODEL' --top-k 4 --output-name qwen-dna-answer 'How does DNA interact with an insulating post in an electric field?'"
ssh "$PAPERS_TARGET" "cat '$PAPERS_PATH/data/papers2_local_queries/qwen-dna-answer.json'"
```

Verified retrieval of the electrophoretic DNA/post collision paper and electric-field
DNA placement paper. The generated answer cites stretching, hook/roll-off events
and Deborah-number dependence from retrieved excerpts. This validates the query
path; full usable-Markdown coverage subsequently passed the audit linked above.

## Archive inventory — September 20, 2026

| Archive check | Count |
| --- | ---: |
| Source PDFs | 11,783 |
| Completed Markdown, all hashes verified | 11,737 |
| Indexed usable Markdown | 11,324 |
| Garbled Markdown deferred; OCR disabled | 413 |
| Unconverted PDFs deferred | 46 |
| Markdown hash/read failures | 0 |

Usable Markdown: 1,132,819,233 bytes (1.055 GiB). All generated Markdown:
1,193,113,963 bytes (1.111 GiB). Saved remote inventory:
`data/papers2_markdown_eligibility.json`. Conversion is finished. The full Chroma
audit verified all 11,324 usable documents and 1,876,298 stored passages, with
413 encoding exclusions listed explicitly. The watcher remains available for new
or changed Markdown; it reuses completed checkpoints.

## Embedding transfer / resume-cache optimization

The SDK requests compact base64 embedding transport and decodes to the same
4,096-dimensional float vectors. Resume-cache JSON uses `orjson`; existing cache
files remain readable. Model, chunking, collection, price cap and checkpoint IDs
are unchanged. A 100-vector serialization benchmark was 0.453s versus 0.018s
(24.7x for serialization only), with identical decoded values. A live CLI query
using compact transport retrieved the expected DNA/post collision paper.

[OpenRouter embedding formats](https://openrouter.ai/docs/api/api-reference/embeddings/submit-an-embedding-request).

```bash
# Drain active papers before deliberately replacing the embedding worker:
ssh "$PAPERS_TARGET" "docker update --restart=no ${CHROMA_CONTAINER}-index"
ssh "$PAPERS_TARGET" "docker kill --signal=SIGINT ${CHROMA_CONTAINER}-index"
ssh "$PAPERS_TARGET" "docker wait ${CHROMA_CONTAINER}-index"
# Then use the deploy/resume command above. Long papers can take time to drain.
```

## Second Qwen RAG example — cellular signaling

```bash
ssh "$PAPERS_TARGET" "docker exec ${CHROMA_CONTAINER}-index python -m hpluslogs.papers_chroma_cli papers-chroma-query --model '$PAPERS_LLM_MODEL' --top-k 6 --output-name qwen-feedback-answer 'How do positive and negative feedback affect cellular signaling pathways?'"
ssh "$PAPERS_TARGET" "cat '$PAPERS_PATH/data/papers2_local_queries/qwen-feedback-answer.json'"
```

Verified retrieval includes the Handbook of Cell Signaling, an excitable gene
regulatory circuit paper, and trainable molecular-network computation. The saved
artifact contains the generated answer, numbered citations and exact retrieved
passages.

## Verified queries against the complete usable archive

DNA polymerase selectivity/proofreading; cellular positive/negative feedback.
Both saved results contain answers, numbered citations, exact passages and URLs.

```bash
ssh "$PAPERS_TARGET" "docker exec ${CHROMA_CONTAINER}-index python -m hpluslogs.papers_chroma_cli papers-chroma-query --model '$PAPERS_LLM_MODEL' --top-k 6 --output-name qwen-full-dna-answer 'How do DNA polymerase selectivity and proofreading contribute to replication fidelity?'"
ssh "$PAPERS_TARGET" "docker exec ${CHROMA_CONTAINER}-index python -m hpluslogs.papers_chroma_cli papers-chroma-query --model '$PAPERS_LLM_MODEL' --top-k 6 --output-name qwen-full-feedback-answer 'How do positive and negative feedback affect cellular signaling pathways?'"
ssh "$PAPERS_TARGET" "cat '$PAPERS_PATH/data/papers2_local_queries/qwen-full-dna-answer.json'"
ssh "$PAPERS_TARGET" "cat '$PAPERS_PATH/data/papers2_local_queries/qwen-full-feedback-answer.json'"
```
