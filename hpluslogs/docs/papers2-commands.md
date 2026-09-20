# papers2 commands

## Working directory and Python environment

```bash
cd /home/kanzure/code/hplusroadmap
source hpluslogs/venv/bin/activate
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
python -m hpluslogs.cli papers-add /path/to/paper.pdf \
  --path biology/paper.pdf
```

## Add a PDF from a URL

```bash
python -m hpluslogs.cli papers-add 'https://example.org/paper.pdf' \
  --path biology/paper.pdf
```

## Replace an existing PDF

```bash
python -m hpluslogs.cli papers-add /path/to/revised-paper.pdf \
  --path biology/paper.pdf --replace
```

Citation links use `https://diyhpl.us/~bryan/papers2/<relative-path>`;
new PDFs must also exist there for public links to resolve.

## New/changed server PDFs — optional rsync

```bash
python -m hpluslogs.cli papers-sync --dry-run
python -m hpluslogs.cli papers-sync
```

## Whole archive / new PDFs / resume — local Markdown and costs

After restoration or additions finish. One run at a time; no uploads.

```bash
python -m hpluslogs.cli papers-prepare --workers 4
```

## Conversion only — whole archive, resumable

```bash
python -m hpluslogs.cli papers-markdown --workers 4
```

## Status / retry interrupted or failed conversions

Completed, unchanged files are skipped. Empty/broken PDFs need inspection.

```bash
python -m hpluslogs.cli papers-status
python -m hpluslogs.cli papers-prepare --workers 4 --timeout 1800
```

## Remote conversion — SSH parameters

SSH, rsync, tar, zstd, Python 3 and Docker required on the remote host; tar/zstd/rsync locally. Choose a dedicated directory. No local converter running.

```bash
PAPERS_HOST=conversion-server.example
PAPERS_USER=your-user
PAPERS_PATH=/home/your-user/hpluslogs-papers-conversion
PAPERS_REMOTE=(--host "$PAPERS_HOST" --user "$PAPERS_USER" --path "$PAPERS_PATH")
```

## Deploy / resume / process newly added PDFs — 48 workers

Streams changed PDFs via tar + zstd over SSH; transfers completed Markdown and a SQLite checkpoint. Later deployments preserve remote progress.

```bash
python -m hpluslogs.cli --data-dir hpluslogs/data papers-remote-deploy \
  "${PAPERS_REMOTE[@]}" --workers 48 --timeout 600 --transfer tar
```

## Remote sizes / ETA / projected monthly costs

```bash
python -m hpluslogs.cli papers-remote-progress "${PAPERS_REMOTE[@]}" \
  --days 30 --searches 1000 --index-multiplier 1
```

## Stop remote conversion — preserve active results

May wait up to the per-PDF timeout. Redeploy to resume.

```bash
python -m hpluslogs.cli papers-remote-stop "${PAPERS_REMOTE[@]}"
```

## Pull results — after completion or stop

Backs up the local manifest; remote manifest becomes authoritative.

```bash
python -m hpluslogs.cli --data-dir hpluslogs/data papers-remote-pull "${PAPERS_REMOTE[@]}"
```

## Local sizes / ETA / projected monthly costs

```bash
python -m hpluslogs.cli --data-dir hpluslogs/data papers-progress \
  --days 30 --searches 1000 --index-multiplier 1
```

## Add more PDFs — remote workflow

```bash
python -m hpluslogs.cli papers-remote-stop "${PAPERS_REMOTE[@]}"
python -m hpluslogs.cli --data-dir hpluslogs/data papers-remote-pull "${PAPERS_REMOTE[@]}"
python -m hpluslogs.cli --data-dir hpluslogs/data papers-add /path/to/new-paper.pdf --path topic/new-paper.pdf
python -m hpluslogs.cli --data-dir hpluslogs/data papers-remote-deploy "${PAPERS_REMOTE[@]}" --workers 48
```

## Cost review — after conversion finishes

```bash
python -m hpluslogs.cli papers-cost --days 30 --searches 1000
python -m json.tool hpluslogs/data/papers2_markdown_cost.json
python -m hpluslogs.cli papers-upload --dry-run --searches 1000
```

**Stop here until upload costs are approved.**
Pricing assumptions/details: [papers2-rag.md](papers2-rag.md).

## Future: index the whole Markdown archive in xAI

After cost approval and successful conversion of every eligible PDF.
`25` is the monthly estimate threshold, not a provider billing cap.

```bash
python -m hpluslogs.cli papers-upload \
  --approve-upload --monthly-budget 25 --searches 1000 \
  --concurrency 4 --wait
python -m hpluslogs.cli papers-status --remote
```

## Future: index additions/changes or resume indexing

After `papers-add`/restore/sync → `papers-prepare` → cost review and approval.

```bash
python -m hpluslogs.cli papers-update \
  --approve-upload --monthly-budget 25 --searches 1000 \
  --concurrency 4 --wait
```

## Future: recover interrupted uploads / retry failed indexing

```bash
python -m hpluslogs.cli papers-reconcile
python -m hpluslogs.cli papers-update \
  --approve-upload --monthly-budget 25 --searches 1000 \
  --retry-failed --wait
```

## Future RAG: answer with paper citations

After xAI indexing. Query usage incurs additional charges; output stays local.

```bash
python -m hpluslogs.cli papers-query \
  --model openrouter/x-ai/grok-4.3 \
  --search-mode hybrid --top-k 20 --no-upload \
  'Compare approaches to molecular assembly described in these papers.'
```

## Future RAG: passages only, without answer generation

```bash
python -m hpluslogs.cli papers-query \
  --search-mode semantic --top-k 10 --nollm \
  'Amorphous computing and distributed coordination'
```

## Future RAG: additional instructions / named output

```bash
python -m hpluslogs.cli papers-query \
  --prompt-fragment 'Compare experimental evidence and limitations; cite sources.' \
  --output-name molecular-assembly --no-upload \
  'Molecular assembly methods'
```

## Saved answers, context, and retrieval metadata

```bash
ls -lh hpluslogs/data/papers2_queries/
```
