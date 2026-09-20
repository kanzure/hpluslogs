# papers2 xAI cost reference

## Latest saved estimate — 2026-09-19, America/Chicago

**$2.89–$3.81 per 30 days for storage**, assuming the billable collection index
is the same size as the Markdown. This is a projection, not a measured xAI bill.
Nothing has been uploaded; this collection's incurred xAI cost is **$0**.

| Measured locally | Value |
| --- | ---: |
| Eligible source PDFs | 11,783 |
| Completed Markdown files | 2,009 |
| Completed Markdown bytes | 141,324,532 |
| Completed Markdown size | 134.78 MiB |
| Full archive projection, per-paper average | 0.772 GiB |
| Full archive projection, PDF-to-Markdown byte ratio | 1.015 GiB |

The successful subset is nonrandom. Failures, scans requiring OCR, and the
remaining papers can change the final size. The range spans two extrapolation
methods, not a statistical confidence interval.

## Published rates — checked 2026-09-19

| Item | USD |
| --- | ---: |
| File storage | $0.025 / GiB / day |
| Collection storage | $0.10 / GiB / day |
| Collection-search tool calls | $2.50 / 1,000 calls |
| File or collection downloads | $0.20 / GiB |
| Grok 4.3 input, below 200k context | $1.25 / million tokens |
| Grok 4.3 output, below 200k context | $2.50 / million tokens |

Source: [xAI pricing](https://docs.x.ai/developers/pricing).
Rates can change. The query CLI defaults to OpenRouter's Grok 4.3 route; verify
that provider's current token rates separately before budgeting actual queries.
The published search-tool price is an allowance here: this implementation uses
the direct Collections search API, whose exact billing should be verified in
the console. A user query can involve multiple billed calls.

## Formula — 30 days

```text
M = stored Markdown GiB
I = billable collection/index GiB
Q = monthly collection-search tool calls

Storage = 0.75 × M + 3.00 × I
Storage + search allowance = 0.75 × M + 3.00 × I + 0.0025 × Q
Model tokens and downloads are additional.
```

xAI index size is unknown before indexing. The Chroma RAM estimate is for a
different implementation and cannot determine xAI's billable index size.
No separate initial embedding/ingestion fee was listed on the checked pricing
page; verify actual account charges before a bulk upload.

## Sensitivity to index size

For projected Markdown of 0.772–1.015 GiB:

| Assumed index size | Storage / 30 days |
| --- | ---: |
| 1× Markdown | $2.89–$3.81 |
| 3× Markdown | $7.53–$9.90 |
| 5× Markdown | $12.16–$15.98 |
| 10× Markdown | $23.74–$31.21 |

The original $25/month limit is not guaranteed by the 1× assumption.
`--monthly-budget 25` is an estimated upload gate, not a provider spending cap;
it excludes model-token and download charges.

## Query examples — illustrative token counts

Using the published Grok 4.3 rates above, one search per query, no downloads:

| Monthly usage | Additional search + model allowance |
| --- | ---: |
| 1,000 searches, no answer generation | $2.50, subject to direct-search billing |
| 1,000 answers; 5,000 input + 500 output tokens each | $10.00 |
| 1,000 answers; 16,500 input + 1,000 output tokens each | $25.63 |

Add storage. Retrieved context and reasoning/output token usage affect the bill.
Thus storage can be below $25 while total RAG usage exceeds it.

## Earlier estimates — historical, superseded

| Measurement | Projected storage / 30 days, index = Markdown |
| --- | ---: |
| Initial three-file sample | about $8.70 |
| 222 files; 13.18 MiB | $2.56–$3.51 |
| 1,978 files; 133.13 MiB | $2.90–$3.82 |
| 2,009 files; 134.78 MiB | $2.89–$3.81 |

## Refresh while conversion runs

```bash
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-remote-progress \
  "${PAPERS_REMOTE[@]}" --days 30 --searches 1000 --index-multiplier 1
```

## Refresh after pulling completed Markdown

```bash
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-progress --days 30 --searches 1000
python -m hpluslogs.cli --data-dir "$PAPERS_DATA" papers-cost --days 30 --searches 1000
python -m json.tool "$PAPERS_DATA/papers2_markdown_cost.json"
```

Setup, approval, upload and query commands: [operations runbook](papers2-commands.md).
