# papers2: local PDF → Markdown → cost review → xAI

[Command reference: setup, additions, conversion, indexing, and RAG](papers2-commands.md).

Updated 2026-09-19. **No xAI uploads are authorized or performed at this stage.**
The previous raw-PDF ingestion plan has been replaced. The background rsync
copies were stopped at the user's request; completed files and partial downloads
remain on disk. The user will restore the full PDF collection using restic.

## Conversion and data layout

The converter matches `~/papers/physical-intelligence/run.py`:

```python
pymupdf4llm.to_markdown(str(pdf), header=False, footer=False)
```

The project pins `pymupdf4llm==1.28.2`, matching that script's environment. See
[PyMuPDF4LLM documentation](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/).
Conversion is local; it does not call xAI, OpenRouter, or an external LLM.
Each PDF runs in a separate subprocess because PyMuPDF is not thread-safe and a
malformed file must not stall the collection. Conversion has a per-file timeout.
The converter's default OCR behavior is retained; inspect failed or empty
outputs, and check extraction quality for scans, formulas, tables, and diagrams.
Images are not embedded as base64 or uploaded separately.

| Path under `hpluslogs/data/` | Purpose |
| --- | --- |
| `papers2/` | Original PDFs and restored source tree |
| `papers2_markdown/` | Generated UTF-8 Markdown only |
| `papers2_markdown.sqlite3` | Conversion fingerprints, versions, source paths, output paths, failures |
| `papers2_markdown_cost.json` | Measured Markdown size, conversion coverage, cost estimate |
| `papers2_markdown_collection.json` | Future Markdown collection ID and configuration |
| `papers2_markdown_index.sqlite3` | Future xAI upload/indexing checkpoints |
| `papers2_queries/` | Retrieved context, source metadata, and generated answers |

The repository-root `data` symlink already points to `hpluslogs/data/`.
Markdown filenames contain a readable stem plus a full hash of the original
relative PDF path. This avoids duplicate-basename collisions, overly long
filenames, and invalid UTF-8 filenames found in this collection. The conversion
manifest retains reversible percent-encoded original paths. Future xAI metadata
and citations refer to the original PDF URLs, not generated Markdown filenames.

Resume checks hash both PDF and Markdown, and include the converter version and
settings. Changed PDFs, modified/missing Markdown, changed converter settings,
and failed conversions are reprocessed. Writes are atomic. Old Markdown from a
failed replacement is excluded from cost estimates and upload eligibility.
Symlinks and hidden/partial-download directories are excluded.

## Restore locally with restic

The repository below was located in `~/remember/restic` and reached successfully.
The critical syntax is **`latest:/path` as one argument**. Restic will
prompt for the repository password, or use a locally configured
`RESTIC_PASSWORD_FILE` / `RESTIC_PASSWORD_COMMAND`.

```bash
restic -r sftp:kanzure@bigboy.local:/srv/storage/rust1/backups/restic-repo-diyhplus \
  --no-lock restore 'latest:/mnt/diyhplus/public_html/papers2' \
  --target /home/kanzure/code/hplusroadmap/hpluslogs/data/papers2 \
  --overwrite if-changed --verify
```

This restores only that subtree. It preserves/restores complete PDF contents and
verifies the restored files, including files already present from the stopped
fallback transfer. Do not run conversion concurrently with restoration.

There is **no extra `restic/` directory** between `rust1/` and `backups/`.
The snapshot subdirectory comes from the backup notes; it could not be inspected
without the repository password. If necessary, verify the stored path first:

```bash
restic -r YOUR_REPOSITORY --no-lock snapshots --latest 1
restic -r YOUR_REPOSITORY --no-lock ls latest
```

For reproducibility replace `latest` with a verified snapshot ID. The existing
`papers-restore` CLI also accepts `--repository`, `--snapshot`, and
`--snapshot-path`; its defaults match the command above.

## Prepare and price — stop here

From `/home/kanzure/code/hplusroadmap`, after restic completes:

```bash
hpluslogs/venv/bin/python -m pip install -r hpluslogs/requirements.txt

hpluslogs/venv/bin/python -m hpluslogs.cli papers-prepare --workers 4

# Optional: include a monthly search allowance in the report.
hpluslogs/venv/bin/python -m hpluslogs.cli papers-cost --searches 1000
```

`papers-prepare` converts locally, writes the measured cost report (including
conversion failures), and stops without uploading. `papers-markdown` runs only
the conversion stage. Both resume automatically. Options:
`--workers` (default 2), `--timeout` (default 600 seconds/document), and `--limit`
(for a bounded sample). `papers-cost` independently verifies current PDF and
Markdown fingerprints, measures only usable Markdown, and writes the report.
It does not submit files or create a collection. `papers-status` lists conversion
failures and upload states. Resolve failures and rerun conversion before
considering the collection complete.

The earlier live-server inventory recorded 11,793 PDF-named entries totaling 25.9517 GiB. Ten are hidden macOS
`._…` metadata sidecars; the eligible corpus has **11,783 PDFs**.
That is **PDF size, not Markdown size**. When this reference inventory is present,
the cost report compares local source filenames and sizes and explicitly labels
incomplete coverage as `partial collection only`. A restic snapshot may differ
from the live server; this comparison is a completeness diagnostic, not proof of
snapshot integrity. Do not extrapolate a full-corpus cost from a few sample PDFs.
Without the reference inventory the report says that completeness of the local
restore has not been independently verified.

## Pricing

[xAI pricing](https://docs.x.ai/developers/pricing), checked 2026-09-19:

| Resource | Rate |
| --- | ---: |
| File storage | $0.025/GiB/day |
| Collection storage | $0.10/GiB/day |
| Collection search tool calls | $2.50 / 1,000 |
| File / collection downloads | $0.20/GiB |

For measured Markdown size **M GiB** and billable index size **I GiB**, 30-day
storage is **$0.75 × M + $3.00 × I**. The default planning assumption is I = M,
yielding **$3.75 per GiB of Markdown per 30 days**. Actual xAI index bytes cannot
be established from Markdown file size alone before indexing. `--index-gib`
accepts a different explicit assumption, and `--days` handles other month
lengths. The report includes any previously uploaded Markdown retained remotely
for source PDFs no longer present locally.

Example scenarios (not measurements of this collection):

| Markdown GiB | Assumed index GiB | 30-day storage |
| ---: | ---: | ---: |
| 0.25 | 0.25 | $0.94 |
| 0.50 | 0.50 | $1.88 |
| 1.00 | 1.00 | $3.75 |
| 1.00 | 4.00 | $12.75 |

`--searches 1000` adds $2.50 of search-tool allowance. That published rate covers
`collections_search` / `file_search`; standalone SDK `documents/search` billing
is not separately clarified in the table. Model tokens and downloads are extra.
The default answer model is OpenRouter `x-ai/grok-4.3`; its previously verified
short-context rates were $1.25/million input tokens and $2.50/million output.
Prices and usage can change. No separate embedding creation rate is specified
on the pricing page. These estimates are not a provider-enforced spending cap.

The old $97.32/month estimate assumed raw PDF files and an equally sized index.
It **does not apply to the new Markdown workflow**. The full Markdown estimate
will be determined after the restored collection has been converted.
Even if it falls below $25/month, **stop and review before any upload**.

## Future indexing and queries, only after approval

`papers-upload`, `papers-update`, and `papers-collect` now submit **Markdown only**.
They never convert or fall back to uploading raw PDFs. By default they print a
cost report and stop without calling xAI. They require all local PDFs to have
current successful conversions, a complete matching source inventory when one
is present, a budget check, and an explicit `--approve-upload` flag. That flag
must only be used after the user authorizes the reviewed cost. It has not been
used in this session.

```bash
# Safe inspection only:
hpluslogs/venv/bin/python -m hpluslogs.cli papers-upload --dry-run
```

A future approved upload streams Markdown to the Files API and attaches it to a
separate `papers2-markdown` collection for xAI chunking/embeddings. Separate state
files prevent mixing the abandoned PDF plan with this collection. Upload IDs are
saved before attachment/polling; changed Markdown updates the existing file ID.
No automatic remote deletions occur. `papers-reconcile` recovers unambiguous IDs
following a lost upload response. Failed indexing is reported explicitly.

After an approved upload, `papers-query` supports hybrid/semantic/keyword search,
`--filter`, `--top-k`, `--nollm`, and answers with linked PDF citations. Results
and context are saved locally; `--upload` on the query command separately opts
into publishing through the project's existing scp workflow. Credentials are
`XAI_API_KEY`, `XAI_MANAGEMENT_API_KEY` (or `XAI_MANAGEMENT_KEY`), and
`OPENROUTER_API_KEY` for answers.

For additions: `papers-add SOURCE --path category/paper.pdf` (optionally
`--replace`), then `papers-markdown`, then `papers-cost`, then stop for cost
review. A local addition must also exist at the collection's configured public
base URL for its citation link to resolve.

## Validation

```bash
hpluslogs/venv/bin/python -m unittest discover -s hpluslogs/tests -v
```

Tests cover conversion settings matching the reference script, duplicate and
legacy filenames, changed inputs and outputs, converter failures, resume,
Markdown-based billing estimates, incomplete-source coverage, mandatory approval,
raw-PDF upload rejection, budget rejection, upload interruption recovery, update
and indexing state, original source citations, and retrieval-only CLI behavior.
All 31 offline tests passed. Three real PDFs were converted with the exact
reference settings and their output inspected:

| PDF | PDF bytes | Markdown bytes |
| --- | ---: | ---: |
| Amorphous computing | 669,245 | 51,683 |
| Representing a circle or a sphere with NURBS | 70,339 | 3,882 |
| A convenient and rapid method for genetic transformation of E. coli with plasmids | 94,572 | 18,933 |
| **Sample total** | **834,156** | **74,498** |

For this sample only, equal Markdown/index sizes imply about **$0.00026 per
30 days** of storage. The full collection price is still unknown. The cost check
saw 10,816 local PDFs versus 11,793 in the source reference inventory, with only
these three converted. It correctly labeled its scope `partial collection only`.
The generated report is `data/papers2_markdown_cost.json`; its $2.50 search
allowance was requested using `--searches 1000` and is separate from storage.
No xAI upload or indexing was attempted in this revision.

The user confirmed restic snapshot `f21af8e9` restored 3,425 files/directories
(8.943 GiB), skipped 23,356 (26.026 GiB), and verified 1,751 restored files.
The full local Markdown preparation was started with four workers; progress is
in `data/papers2-prepare.log` and durable per-file state is in SQLite.

The local conversion was interrupted and resumed to validate durability: all
26 completed Markdown outputs retained identical hashes and modification times.
The resumed run is logged in `data/papers2-prepare-resumed.log`; it will write
the final measured cost report after processing the collection. Some files
produce empty Markdown under the reference converter settings; these are
recorded as failures and excluded from the ready-byte total, never silently
reported as converted.

## Interim cost measurement

Measured 2026-09-19T21:27:22.659438+00:00; conversion still running. No xAI uploads.

| Metric | Value |
| --- | ---: |
| Completed Markdown | 222 / 11,783 PDFs |
| Current Markdown bytes | 13,821,283 (13.18 MiB) |
| Failed conversions | 17 |
| Previous three-file storage projection | $8.70 / 30 days |
| New projection, average Markdown size per completed file | 0.683 GiB; $2.56 / 30 days |
| New projection, measured PDF-to-Markdown byte ratio | 0.935 GiB; $3.51 / 30 days |

Both projections assume index bytes equal Markdown bytes. Only about
1.9% of files have converted successfully;
this subset is not representative, and failed/unfinished conversions are
excluded. These are provisional storage estimates, not a final bill or budget
guarantee. Add search-tool allowance ($2.50/1,000 calls) and model tokens.
Snapshot: `data/papers2_markdown_progress.json`. The final cost report will be
written to `data/papers2_markdown_cost.json` after preparation finishes.


## Remote Docker conversion and automated estimates

The local conversion was stopped with **2,009 ready files, 141,324,532 bytes
(134.78 MiB)** preserved. Its 155 failures and four interrupted rows remain
retryable. The completed subset projects **0.772–1.015 GiB** for the full archive,
or **$2.89–$3.81 per 30 days** with index size equal to Markdown size. Search and
model tokens are additional. These are provisional estimates; nothing was
uploaded to xAI.

`papers-remote-deploy --host HOST --user USER --path /absolute/directory`
builds a small conversion image and streams changed local PDFs using tar,
zstd and SSH. No HTTP archive downloads. The directory must be dedicated to
this job; code/build files are synchronized separately from persistent data.
Hostnames and SSH usernames are parameters. The initial migration uses 48
workers on a host with 64 CPUs; choose `--workers` for the available resources.

The initial checkpoint is copied through SQLite's backup API, including any
committed WAL transactions. Completed Markdown accompanies it. Subsequent
deployments preserve the remote manifest and skip unchanged PDF/Markdown
hashes with the same converter recipe. Package versions match the local run.
Local checkpoints and original PDFs are retained. A stopped remote container
can be pulled back; the local manifest is backed up before replacement.
Use the remote directory as the authoritative conversion state until pulling.

The conversion container runs without network access or API credentials, as
the SSH user's numeric UID/GID, with the PDF source mount read-only. Docker
build requires package downloads. Conversion state and Markdown persist under
the deployment directory's `data/`; stopping a container lets active PDFs finish
(up to the configured timeout), cancels queued work, and keeps completed output.
Failed documents remain explicit failures and may require OCR or inspection.

`papers-progress` and `papers-remote-progress` report Markdown bytes, ready/failed
counts, throughput over 15/60/120 minutes, first-pass ETA, and two full-archive
cost projections (per-paper average and PDF-byte ratio). Each run records its
start so local throughput is not mixed into the remote run's ETA. Progress is
a quick size-based estimate; `papers-cost` remains the hash-verified upload gate.
`--days`, `--searches`, and `--index-multiplier` control the estimate assumptions.

Invocations: [papers2-commands.md](papers2-commands.md).


## Reference documents

- [Operations, progress/ETA, next steps, uploads and queries](papers2-commands.md)
- [Saved xAI rates, estimates and budget scenarios](papers2-xai-costs.md)
- [Local Chroma RAM estimate and recalculation](papers2-chroma-sizing.md)
