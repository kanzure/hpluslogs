# papers2 operations runbook

Next: finish transfer → automatic remote conversion → pull → inspect/retry failures →
review costs → explicit upload approval → index → query.

[xAI cost reference](papers2-xai-costs.md) · [Local Chroma RAM estimate](papers2-chroma-sizing.md)

## Working directory and Python environment

```bash
cd /home/kanzure/code/hplusroadmap
source hpluslogs/venv/bin/activate
export PAPERS_DATA="$PWD/hpluslogs/data"
python -m pip install -r hpluslogs/requirements.txt
```

API environment, for future indexing/queries: `XAI_API_KEY`,
`XAI_MANAGEMENT_API_KEY` (or `XAI_MANAGEMENT_KEY`), `OPENROUTER_API_KEY`.

## Initial restore — papers2 only

```bash
PAPERS_RESTIC_REPOSITORY='sftp:user@backup-host:/path/to/restic-repo'
restic -r "$PAPERS_RESTIC_REPOSITORY" \
  --no-lock restore 'latest:/mnt/diyhplus/public_html/papers2' \
  --target /home/kanzure/code/hplusroadmap/hpluslogs/data/papers2 \
  --overwrite if-changed --verify
```

## Add a local PDF

```bash
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-add /path/to/paper.pdf \
  --path biology/paper.pdf
```

## Add a PDF from a URL

```bash
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-add 'https://example.org/paper.pdf' \
  --path biology/paper.pdf
```

## Replace an existing PDF

```bash
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-add /path/to/revised-paper.pdf \
  --path biology/paper.pdf --replace
```

Citation links use `https://diyhpl.us/~bryan/papers2/<relative-path>`;
new PDFs must also exist there for public links to resolve.

## New/changed server PDFs — optional rsync

```bash
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-sync --dry-run
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-sync
```

## Whole archive / new PDFs / resume — local Markdown and costs

After restoration or additions finish. One run at a time; no uploads.

```bash
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-prepare --workers 4
```

## Conversion only — whole archive, resumable

```bash
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-markdown --workers 4
```

## Status / retry interrupted or failed conversions

Completed, unchanged files are skipped. Empty/broken PDFs need inspection.

```bash
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-status
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-prepare --workers 4 --timeout 1800
```

## Remote conversion — SSH parameters

SSH, rsync, tar, zstd, Python 3 and Docker required on the remote host; tar/zstd/rsync locally. Choose a dedicated directory. No local converter running.

```bash
# Saved parameters for this deployment, when present:
if [ -f "$PAPERS_DATA/papers2-remote.env" ]; then source "$PAPERS_DATA/papers2-remote.env"; fi
# Otherwise replace these defaults:
PAPERS_HOST="${PAPERS_HOST:-conversion-server.example}"
PAPERS_USER="${PAPERS_USER:-your-user}"
PAPERS_PATH="${PAPERS_PATH:-/home/your-user/hpluslogs-papers-conversion}"
PAPERS_REMOTE=(--host "$PAPERS_HOST" --user "$PAPERS_USER" --path "$PAPERS_PATH")
PAPERS_TARGET="$PAPERS_USER@$PAPERS_HOST"
PAPERS_CONTAINER="hpluslogs-papers-$(python -c 'import hashlib,sys; from pathlib import PurePosixPath; print(hashlib.sha256(str(PurePosixPath(sys.argv[1])).encode()).hexdigest()[:12])' "$PAPERS_PATH")"
```

## Deploy / resume / process newly added PDFs — 48 workers

Streams changed PDFs via tar + zstd over SSH; transfers completed Markdown and a SQLite checkpoint. Later deployments preserve remote progress.

```bash
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-remote-deploy \
  "${PAPERS_REMOTE[@]}" --workers 48 --timeout 600 --transfer tar
```

## Current deployment log — transfer, then container launch

```bash
tail -n 30 "$PAPERS_DATA/papers2-remote-deploy.log"
tail -f "$PAPERS_DATA/papers2-remote-deploy.log"
```

## Transfer progress / measured transfer ETA — two samples, 20 seconds apart

```bash
python hpluslogs/scripts/papers_transfer_progress.py \
  --data-dir "$PAPERS_DATA" "${PAPERS_REMOTE[@]}" --interval 20
```

`conversion_container_state: absent`: conversion has not started.
Transfer ETA excludes the later Markdown/checkpoint copy and conversion.

## Container state / conversion logs / exit code

```bash
ssh "$PAPERS_TARGET" "docker ps -a --filter name=^/${PAPERS_CONTAINER}$ --format '{{.Names}} {{.Status}}'"
ssh "$PAPERS_TARGET" "docker logs --tail 30 $PAPERS_CONTAINER"
ssh "$PAPERS_TARGET" "docker logs --follow --tail 30 $PAPERS_CONTAINER"
ssh "$PAPERS_TARGET" "docker inspect --format '{{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}}' $PAPERS_CONTAINER"
```

Exit 0: pass finished. Nonzero: inspect failures/timeout/OOM; completed files remain saved.

## Remote sizes / conversion ETA / projected monthly costs

After the container starts. Read `throughput[].first_pass_remaining_hours`.
ETA excludes failure repair; null means insufficient recent completions or no remaining work.


```bash
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-remote-progress "${PAPERS_REMOTE[@]}" \
  --days 30 --searches 1000 --index-multiplier 1
```

## Stop remote conversion — preserve active results

May wait up to the per-PDF timeout. Redeploy to resume.

```bash
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-remote-stop "${PAPERS_REMOTE[@]}"
```

## Pull results — after completion or stop

After the container exits, pull even if it reported conversion failures.


Backs up the local manifest; remote manifest becomes authoritative.

```bash
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-remote-pull "${PAPERS_REMOTE[@]}"
```

## Local sizes / ETA / projected monthly costs

```bash
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-progress \
  --days 30 --searches 1000 --index-multiplier 1
```

## After pulling — inspect remaining conversion failures

```bash
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-status > "$PAPERS_DATA/papers2-status.json"
python -c 'import json,sys; r=json.load(open(sys.argv[1])); print(json.dumps(r["conversion_problems"], indent=2))' "$PAPERS_DATA/papers2-status.json"
```

## Retry remotely — longer per-paper timeout, unchanged outputs skipped

```bash
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-remote-deploy \
  "${PAPERS_REMOTE[@]}" --workers 48 --timeout 1800 --transfer tar
```

Empty-text/scanned/broken PDFs need inspection or OCR; increasing the timeout alone may not help.
Pull again after retries. All eligible papers must convert successfully before upload.

## Add more PDFs — remote workflow

```bash
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-remote-stop "${PAPERS_REMOTE[@]}"
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-remote-pull "${PAPERS_REMOTE[@]}"
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-add /path/to/new-paper.pdf --path topic/new-paper.pdf
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-remote-deploy "${PAPERS_REMOTE[@]}" --workers 48
```

## Cost review — after conversion finishes

```bash
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-cost --days 30 --searches 1000
python -m json.tool hpluslogs/data/papers2_markdown_cost.json
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-upload --dry-run --searches 1000
```

**Stop here until upload costs are approved.**
Pricing assumptions/details: [papers2-rag.md](papers2-rag.md).

## Future: index the whole Markdown archive in xAI

After cost approval and successful conversion of every eligible PDF.
`25` is the monthly estimate threshold, not a provider billing cap.

```bash
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-upload \
  --approve-upload --monthly-budget 25 --searches 1000 \
  --concurrency 4 --wait
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-status --remote
```

## Future: index additions/changes or resume indexing

After `papers-add`/restore/sync → `papers-prepare` → cost review and approval.

```bash
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-update \
  --approve-upload --monthly-budget 25 --searches 1000 \
  --concurrency 4 --wait
```

## Future: recover interrupted uploads / retry failed indexing

```bash
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-reconcile
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-update \
  --approve-upload --monthly-budget 25 --searches 1000 \
  --retry-failed --wait
```

## Future RAG: answer with paper citations

After xAI indexing. Query usage incurs additional charges; output stays local.

```bash
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-query \
  --model openrouter/x-ai/grok-4.3 \
  --search-mode hybrid --top-k 20 --no-upload \
  'Compare approaches to molecular assembly described in these papers.'
```

## Future RAG: passages only, without answer generation

```bash
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-query \
  --search-mode semantic --top-k 10 --nollm \
  'Amorphous computing and distributed coordination'
```

## Future RAG: additional instructions / named output

```bash
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-query \
  --prompt-fragment 'Compare experimental evidence and limitations; cite sources.' \
  --output-name molecular-assembly --no-upload \
  'Molecular assembly methods'
```

## Saved answers, context, and retrieval metadata

```bash
ls -lh hpluslogs/data/papers2_queries/
```

## Recalculate local Chroma RAM requirements — no embeddings or API calls

```bash
python hpluslogs/scripts/estimate_papers_chroma.py --data-dir "$PAPERS_DATA" \
  --dimensions 1024 4096 > "$PAPERS_DATA/papers2_chroma_estimate.json"
python -m json.tool "$PAPERS_DATA/papers2_chroma_estimate.json"
```

Counts ready Markdown with the project tokenizer. Local Chroma ingestion/query commands
for this paper archive are not implemented; the `papers-query` commands above use xAI.

## Alternative: restore on the conversion host from its local repository

Run on that host. Use the same snapshot as the original restore; stage separately
while the network transfer is active. Current collection snapshot: `f21af8e9`.

```bash
PAPERS_PATH=/home/your-user/hpluslogs-papers-conversion
PAPERS_RESTIC_REPOSITORY=/srv/storage/rust1/backups/restic-repo-diyhplus
restic -r "$PAPERS_RESTIC_REPOSITORY" --no-lock \
  restore 'f21af8e9:/mnt/diyhplus/public_html/papers2' \
  --target "$PAPERS_PATH/data/papers2-restic" \
  --overwrite if-changed --verify
```

After successful verification: stop the active transfer and its pipeline
processes; preserve its partial directory; move `papers2-restic` to `papers2`;
rerun `papers-remote-deploy`. Matching PDFs are skipped, saved Markdown/checkpoint
are transferred, and conversion starts. Do not rename directories while tar or
conversion is still running. Keep the completed Markdown and SQLite manifest.


## Automatic handoff from remote restic restore to conversion

Cancel the prior transfer first. Requires user systemd with lingering enabled.
No repo password is copied; waits for restore processes to exit and independently
checks every expected PDF SHA-256 before switching directories and starting Docker.
Also verifies the saved Markdown checkpoint. Preserves the partial transfer directory.

```bash
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-remote-watch-restore \
  "${PAPERS_REMOTE[@]}" --staging-name papers2-restic --workers 48 --timeout 600
```

The watcher lives on the destination and survives SSH disconnection. Once setup
prints `Watcher ready`, the local terminal can be closed. Errors are logged and
retried by systemd; mismatched/incomplete PDFs are never silently accepted.

## Watcher status / logs

```bash
ssh "$PAPERS_TARGET" "systemctl --user status ${PAPERS_CONTAINER}-restore-watch.service --no-pager"
ssh "$PAPERS_TARGET" "journalctl --user -u ${PAPERS_CONTAINER}-restore-watch.service -n 30 --no-pager"
ssh "$PAPERS_TARGET" "docker logs --tail 30 $PAPERS_CONTAINER"
```

After handoff, use `papers-remote-progress` for conversion ETA. The completed
watcher unit can disappear (`--collect`); Docker logs remain available and the
last watcher state persists in remote `data/papers2_restore_watch.json`.

## Automatic progress history — every five minutes, until conversion exits

```bash
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-remote-monitor \
  "${PAPERS_REMOTE[@]}" --interval 300
ssh "$PAPERS_TARGET" "journalctl --user -u ${PAPERS_CONTAINER}-progress.service -n 10 --no-pager"
ssh "$PAPERS_TARGET" "journalctl --user -u ${PAPERS_CONTAINER}-progress.service --follow"
```

The host records `data/papers2_progress_latest.json` and
`data/papers2_progress_history.jsonl`, including ready/failed counts, Markdown
size, ETA, costs and container status. Survives SSH disconnection; records a final
snapshot and exits when conversion stops. This saves reports on the host; it does
not send chat messages. Start it again for a subsequent conversion run.
