# Local Chroma paper RAG

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
```

## Deploy / resume indexing

```bash
python hpluslogs/scripts/deploy_papers_chroma.py "${PAPERS_REMOTE[@]}" --workers 4 --port 18081
```

Local MiniLM-L6-v2 embeddings: 384 dimensions; 224-wordpiece chunks, 32 overlap.
Model downloaded during Docker build; papers remain local. Chroma listens on remote
loopback only. Indexer watches completed Markdown every 120 seconds, including new
PDF conversions. Both services restart after reboot. Index checkpoints persist in
`data/papers2_chroma.sqlite3`; vectors in `data/chroma`. PDF conversion runs separately.

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

## Release embedding CPU / restart

```bash
ssh "$PAPERS_TARGET" "docker stop ${CHROMA_CONTAINER}-index"
ssh "$PAPERS_TARGET" "docker start ${CHROMA_CONTAINER}-index"
```

## References

[Chroma server deployment](https://docs.trychroma.com/guides/deploy/docker).
[MiniLM model and context limit](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
This local index uses MiniLM, not the 4096-dimensional Qwen sizing scenario in
`papers2-chroma-sizing.md`. No hosted embedding or vector-storage charges.

## Verified deployment — September 19, 2026

- Host: `kanzure@bigboy.local`; root: `/home/kanzure/hpluslogs-papers-conversion`.
- Chroma 1.4.0 on `127.0.0.1:18081`; collection `papers2_minilm_v1`.
- Local answer model: `qwen-flash-next-uncensored-sglang`, `http://127.0.0.1:8080/v1`.
- Verified DNA surface-alignment answer retrieved COMMIC and electric-field stretching papers, with numbered citations.
- Verified carbon-nanotube placement retrieval returned source passages.
- Example artifacts: remote `data/papers2_local_queries/dna-surface-example.json` and `nanotube-placement-example.json`.
- Initial coverage check: 290 indexed papers / 19,932 chunks; 6,753 converted papers. Indexing continues; examples do not prove full-archive coverage.

## OCR retry after the current conversion pass

English Tesseract data is included in the conversion image. A previously empty
scanned PDF produced 10,020 Markdown bytes in a real smoke test. Completed outputs
are preserved; the retry uses the same PyMuPDF4LLM settings. OCR can contain errors.
Empty or invalid source files still require replacement; inspect failures afterward.

```bash
# Build/deploy an OCR-capable image using the normal remote conversion workflow.
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

Run after conversion/retries finish and `papers.ready` equals `ready_markdown`.
Stop the embedding writer for a stable audit; leave the Chroma server running.
The command verifies every passage against Markdown bytes and offsets, identities,
source URLs and complete chunk counts. It reports incomplete coverage as failure.
Conversion failures remain a separate check.

```bash
ssh "$PAPERS_TARGET" "docker stop ${CHROMA_CONTAINER}-index"
ssh "$PAPERS_TARGET" "docker run --rm --network host --user \$(id -u):\$(id -g) --mount type=bind,src=$PAPERS_PATH/data,dst=/data ${CHROMA_CONTAINER}:latest papers-chroma-audit"
ssh "$PAPERS_TARGET" "cat '$PAPERS_PATH/data/papers2_chroma_audit.json'"
ssh "$PAPERS_TARGET" "docker start ${CHROMA_CONTAINER}-index"
```

## Embedding concurrency

```bash
# Restarts only the embedding worker; stored batches are preserved:
python hpluslogs/scripts/deploy_papers_chroma.py "${PAPERS_REMOTE[@]}" --workers 8
# After PDF conversion frees CPU, increase if other host workloads permit:
python hpluslogs/scripts/deploy_papers_chroma.py "${PAPERS_REMOTE[@]}" --workers 16
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
reuse, not general conversion throughput. The scheduled OCR retry uses 32 workers.

Conversion progress includes 5-, 15-, 60- and 120-minute rates. Use the short window
when the remaining work changes from articles to long books; timeouts and OCR
retries can still make the final tail slower than an extrapolated ETA.
