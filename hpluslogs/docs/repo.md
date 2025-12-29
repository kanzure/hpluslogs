# hpluslogs Repository Overview

This project is a RAG (Retrieval-Augmented Generation) system for querying IRC logs and other document collections. It supports multiple backends (local Chroma, xAI Collections) and multiple data sources.

## Project Structure

```
hpluslogs/
├── cli.py              # Command-line interface (entry point)
├── webapp.py           # Tornado web server for HTTP API
├── core/               # Core utilities and shared logic
│   ├── chunking.py     # Text parsing and chunking functions
│   ├── prompts.py      # LLM prompt templates
│   └── utils.py        # Helper functions (daterange, token_count, etc.)
├── integrations/       # External service adapters (API wrappers)
│   ├── chroma.py       # Local Chroma vector database
│   ├── gnusha.py       # gnusha.org IRC log fetcher
│   ├── openrouter.py   # OpenRouter API (embeddings + completions)
│   ├── pandoc.py       # Pandoc HTML generation
│   ├── scp.py          # SCP file upload
│   └── xai.py          # xAI Collections API
├── services/           # Business logic orchestration
│   ├── download.py     # Fetch raw data from sources
│   ├── embedding.py    # Vectorize chunks into Chroma
│   ├── generation.py   # Build prompts and call LLMs
│   ├── preprocess.py   # Convert raw data to JSONL chunks
│   ├── publishing.py   # Output files, generate HTML, upload
│   ├── search.py       # Retrieve chunks from any backend
│   └── xai_upload.py   # Upload documents to xAI Collections
└── docs/               # Documentation
```

## Architecture Layers

### 1. Integrations (`integrations/`)

Low-level adapters that wrap external APIs and services. Each module handles a single external dependency:

- **chroma.py** - Local vector database operations (get_client, get_collection, add_embeddings, search)
- **openrouter.py** - OpenRouter API for embeddings and LLM completions
- **xai.py** - xAI Collections API for cloud-hosted vector search
- **gnusha.py** - HTTP fetcher for IRC logs from gnusha.org
- **pandoc.py** - Shell wrapper for pandoc HTML generation
- **scp.py** - Shell wrapper for scp file uploads

Integrations should be stateless and focused on a single external service.

### 2. Services (`services/`)

Business logic that orchestrates integrations. Services implement the actual workflows:

- **download.py** - Fetches raw data using integrations (e.g., gnusha)
- **preprocess.py** - Parses raw files into JSONL chunks
- **embedding.py** - Embeds chunks and stores in Chroma
- **search.py** - Unified search interface across backends (Chroma, xAI)
- **generation.py** - Builds RAG prompts and calls LLMs
- **publishing.py** - Saves output, generates HTML, uploads to server
- **xai_upload.py** - Uploads documents to xAI Collections

### 3. Core (`core/`)

Shared utilities and configuration:

- **chunking.py** - Text parsing and sliding-window chunking
- **prompts.py** - LLM prompt templates for different collections
- **utils.py** - Helper functions (daterange, token_count, ensure_directory)

### 4. CLI (`cli.py`)

Click-based command-line interface that exposes services as commands.

## Data Flow

### Local Chroma Pipeline
```
download → preprocess → embed → query
   ↓           ↓          ↓        ↓
 raw/      chunks/     index/   outputs/
```

### xAI Collections Pipeline
```
download → xai-upload → xai-query
   ↓           ↓            ↓
 raw/    (cloud storage)  outputs/
```

## Adding a New Collection / Data Source

### Step 1: Create an Integration (if needed)

If your data source requires fetching from a new API or service, add a new module in `integrations/`:

```python
# integrations/myservice.py
def fetch_document(doc_id: str) -> str:
    """Fetch a document from MyService."""
    # Implementation here
    pass
```

### Step 2: Add Download Logic

Either extend `services/download.py` or create a new download service:

```python
# In services/download.py or a new file
def download_mycollection(data_dir: Path, ...) -> None:
    """Download documents for mycollection."""
    raw_dir = data_dir / "raw_mycollection"
    ensure_directory(raw_dir)
    # Fetch and save files
```

### Step 3: Add Upload Function (for xAI Collections)

Add a new `run_*` function in `services/xai_upload.py`:

```python
def run_mycollection(
    data_dir: Path,
    collection_name: str = "mycollection",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    resume: bool = True,
    concurrency: int = 100,
    wait_for_indexing: bool = False,
) -> None:
    """Upload mycollection files to xAI Collections."""
    raw_dir = data_dir / "raw_mycollection"
    config_file = data_dir / "mycollection_collection.json"
    
    # Define field schema
    field_definitions = [
        {"name": "date", "type": "string", "description": "Document date"},
        # Add more fields as needed
    ]
    
    # Get or create collection
    collection_id = get_or_create_collection(...)
    
    # Build file list with metadata
    files = []
    for f in sorted(raw_dir.glob("*.txt")):
        fields = {"date": extract_date(f)}
        files.append((f, fields))
    
    # Upload using shared async function
    asyncio.run(upload_files_async(...))
```

### Step 4: Add Prompts

Add collection-specific prompts in `core/prompts.py`:

```python
MYCOLLECTION_SYSTEM_PROMPT = """You are an assistant with access to mycollection..."""

MYCOLLECTION_SEARCH_QUERY_PROMPT = """You are a search query generator for mycollection..."""
```

### Step 5: Add CLI Commands

Add commands in `cli.py`:

```python
@cli.command("mycollection-collect")
@click.option("--collection-name", default="mycollection", ...)
@click.pass_obj
def mycollection_collect_cmd(obj: dict, collection_name: str, ...) -> None:
    """Upload mycollection files to xAI Collections."""
    xai_upload.run_mycollection(obj["data_dir"], collection_name, ...)


@cli.command("mycollection-query")
@click.option("--collection-id", default=None, ...)
@click.pass_obj
def mycollection_query_cmd(obj: dict, collection_id: str, ...) -> None:
    """Query mycollection and generate an answer."""
    # Similar to fightaging_query_cmd
```

### Step 6: Update generation.py (if needed)

If your collection needs special context formatting or search query generation:

```python
def generate_search_query(prompt_fragment: str, model: str, for_mycollection: bool = False) -> str:
    if for_mycollection:
        prompt = MYCOLLECTION_SEARCH_QUERY_PROMPT.format(...)
    # ...
```

## Existing Collections

### hplusroadmap (IRC logs)
- **Source**: gnusha.org IRC logs
- **Commands**: `download`, `preprocess`, `embed`, `query`, `xai-upload`, `xai-query`
- **Config file**: `data/xai_collection.json`

### fightaging (articles)
- **Source**: Fight Aging! HTML articles
- **Commands**: `fightaging-collect`, `fightaging-query`
- **Config file**: `data/fightaging_collection.json`

### lesswrong (IRC logs)
- **Source**: LessWrong IRC logs (rationality, AI alignment, effective altruism)
- **Raw data**: `data-lesswrong/raw/`
- **Commands**: `lesswrong-upload`, `lesswrong-query`
- **Config file**: `data/lesswrong_collection.json`

## Environment Variables

- `OPENROUTER_API_KEY` - Required for embeddings and LLM calls
- `XAI_API_KEY` - Required for xAI Collections

## Running the CLI

```bash
# Download IRC logs
python -m hpluslogs.cli download --start 2024-01-01 --end 2024-12-31

# Preprocess into chunks
python -m hpluslogs.cli preprocess

# Embed into local Chroma
python -m hpluslogs.cli embed

# Query local Chroma
python -m hpluslogs.cli query "What is CRISPR?"

# Upload to xAI Collections
python -m hpluslogs.cli xai-upload

# Query xAI Collections
python -m hpluslogs.cli xai-query "What is CRISPR?"
```
