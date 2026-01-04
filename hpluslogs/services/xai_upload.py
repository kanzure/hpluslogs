"""xAI upload service for uploading files to xAI Collections."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List, Optional, Set, Tuple

import click

from hpluslogs.integrations import xai


async def upload_file_async(
    client,
    collection_id: str,
    log_file: Path,
    semaphore: asyncio.Semaphore,
    fields: dict,
    wait_for_indexing: bool = False,
) -> Tuple[str, bool, Optional[str]]:
    """Upload a single file to xAI collection asynchronously.
    
    Returns a tuple of (filename, success, file_id_or_error).
    """
    async with semaphore:
        try:
            file_id = await xai.upload_document_async(
                client, collection_id, log_file, fields, wait_for_indexing
            )
            return (log_file.name, True, file_id)
        except Exception as e:
            return (log_file.name, False, str(e))


async def upload_files_async(
    collection_id: str,
    files: List[Tuple[Path, dict]],  # List of (file_path, fields)
    uploaded_files: Set[str],
    concurrency: int,
    wait_for_indexing: bool,
    resume: bool,
    file_key_fn=lambda f, fields: f.name,
) -> Tuple[int, int, Set[str]]:
    """Upload multiple files to xAI collection concurrently.
    
    Returns (uploaded_count, failed_count, newly_uploaded_files).
    """
    client = xai.get_async_client()
    semaphore = asyncio.Semaphore(concurrency)
    
    # Filter files to upload
    files_to_upload = []
    for file_path, fields in files:
        file_key = file_key_fn(file_path, fields)
        if resume and file_key in uploaded_files:
            click.echo(f"✓ {file_key} already uploaded, skipping")
            continue
        files_to_upload.append((file_path, fields, file_key))
    
    if not files_to_upload:
        click.echo("No new files to upload.")
        await client.close()
        return (0, 0, set())
    
    click.echo(f"Uploading {len(files_to_upload)} files with concurrency={concurrency}...")
    
    # Create upload tasks
    tasks = [
        upload_file_async(client, collection_id, file_path, semaphore, fields, wait_for_indexing)
        for file_path, fields, _ in files_to_upload
    ]
    
    # Process with progress reporting
    uploaded_count = 0
    failed_count = 0
    newly_uploaded = set()
    
    file_keys = [file_key for _, _, file_key in files_to_upload]
    
    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
        filename, success, result = await coro
        # Find the file_key for this filename
        file_key = next((fk for fp, _, fk in files_to_upload if fp.name == filename), filename)
        if success:
            click.echo(f"[{i}/{len(tasks)}] ✓ {file_key} -> {result}")
            newly_uploaded.add(file_key)
            uploaded_count += 1
        else:
            click.echo(f"[{i}/{len(tasks)}] ✗ {file_key}: {result}")
            failed_count += 1
    
    await client.close()
    return (uploaded_count, failed_count, newly_uploaded)


def run_hplusroadmap(
    data_dir: Path,
    collection_name: str = "hplusroadmap-logs",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    resume: bool = True,
    concurrency: int = 100,
    wait_for_indexing: bool = False,
) -> None:
    """Upload raw IRC logs to xAI Collections."""
    raw_dir = data_dir / "raw"
    config_file = data_dir / "xai_collection.json"

    if not raw_dir.exists():
        raise click.UsageError(f"Raw logs directory does not exist: {raw_dir}")

    log_files = sorted(raw_dir.glob("*.log"))
    if not log_files:
        click.echo("No log files found to upload.")
        return

    # Check if we have an existing collection
    collection_id = None
    uploaded_files: Set[str] = set()
    if config_file.exists():
        config = json.loads(config_file.read_text(encoding="utf-8"))
        collection_id = config.get("collection_id")
        uploaded_files = set(config.get("uploaded_files", []))
        click.echo(f"Found existing collection: {collection_id}")
        click.echo(f"Already uploaded: {len(uploaded_files)} files")

    # Create collection if needed
    if collection_id is None:
        click.echo(f"Creating collection '{collection_name}'...")
        collection_id = xai.create_collection(
            name=collection_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            field_definitions=[
                {"key": "date", "required": True, "unique": False, "inject_into_chunk": True},
                {"key": "filename", "required": True, "unique": False, "inject_into_chunk": False},
            ],
        )
        click.echo(f"Created collection: {collection_id}")

        config = {"collection_id": collection_id, "collection_name": collection_name, "uploaded_files": []}
        config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Prepare files with fields
    files_with_fields = [
        (f, {"date": f.stem, "filename": f.name})
        for f in log_files
    ]

    click.echo(f"\nTotal files: {len(log_files)}, Already uploaded: {len(uploaded_files)}")
    uploaded_count, failed_count, newly_uploaded = asyncio.run(
        upload_files_async(
            collection_id=collection_id,
            files=files_with_fields,
            uploaded_files=uploaded_files,
            concurrency=concurrency,
            wait_for_indexing=wait_for_indexing,
            resume=resume,
        )
    )

    # Update config with newly uploaded files
    if newly_uploaded:
        uploaded_files.update(newly_uploaded)
        config = json.loads(config_file.read_text(encoding="utf-8"))
        config["uploaded_files"] = list(uploaded_files)
        config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")

    click.echo(f"\nDone! Uploaded: {uploaded_count}, Failed: {failed_count}, Total in collection: {len(uploaded_files)}")
    click.echo(f"Collection ID: {collection_id}")
    if not wait_for_indexing and uploaded_count > 0:
        click.echo("\nNote: Documents were uploaded without waiting for indexing.")


def run_lesswrong(
    data_dir: Path,
    collection_name: str = "lesswrong-logs",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    resume: bool = True,
    concurrency: int = 100,
    wait_for_indexing: bool = False,
) -> None:
    """Upload LessWrong IRC logs to xAI Collections."""
    raw_dir = data_dir / "data-lesswrong" / "raw"
    config_file = data_dir / "lesswrong_collection.json"

    if not raw_dir.exists():
        raise click.UsageError(f"LessWrong raw logs directory does not exist: {raw_dir}")

    log_files = sorted(raw_dir.glob("*.log"))
    if not log_files:
        # Try without extension in case files don't have .log
        log_files = sorted(f for f in raw_dir.iterdir() if f.is_file())
    
    if not log_files:
        click.echo("No log files found to upload.")
        return

    click.echo(f"Found {len(log_files)} log files")

    # Check if we have an existing collection
    collection_id = None
    uploaded_files: Set[str] = set()
    if config_file.exists():
        config = json.loads(config_file.read_text(encoding="utf-8"))
        collection_id = config.get("collection_id")
        uploaded_files = set(config.get("uploaded_files", []))
        click.echo(f"Found existing collection: {collection_id}")
        click.echo(f"Already uploaded: {len(uploaded_files)} files")

    # Create collection if needed
    if collection_id is None:
        click.echo(f"Creating collection '{collection_name}'...")
        collection_id = xai.create_collection(
            name=collection_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            field_definitions=[
                {"key": "date", "required": True, "unique": False, "inject_into_chunk": True},
                {"key": "filename", "required": True, "unique": False, "inject_into_chunk": False},
            ],
        )
        click.echo(f"Created collection: {collection_id}")

        config = {"collection_id": collection_id, "collection_name": collection_name, "uploaded_files": []}
        config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Prepare files with fields
    files_with_fields = [
        (f, {"date": f.stem, "filename": f.name})
        for f in log_files
    ]

    click.echo(f"\nTotal files: {len(log_files)}, Already uploaded: {len(uploaded_files)}")
    uploaded_count, failed_count, newly_uploaded = asyncio.run(
        upload_files_async(
            collection_id=collection_id,
            files=files_with_fields,
            uploaded_files=uploaded_files,
            concurrency=concurrency,
            wait_for_indexing=wait_for_indexing,
            resume=resume,
        )
    )

    # Update config with newly uploaded files
    if newly_uploaded:
        uploaded_files.update(newly_uploaded)
        config = json.loads(config_file.read_text(encoding="utf-8"))
        config["uploaded_files"] = list(uploaded_files)
        config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")

    click.echo(f"\nDone! Uploaded: {uploaded_count}, Failed: {failed_count}, Total in collection: {len(uploaded_files)}")
    click.echo(f"Collection ID: {collection_id}")
    if not wait_for_indexing and uploaded_count > 0:
        click.echo("\nNote: Documents were uploaded without waiting for indexing.")


def run_orionsarm(
    data_dir: Path,
    collection_name: str = "orionsarm-encyclopedia",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    resume: bool = True,
    concurrency: int = 100,
    wait_for_indexing: bool = False,
) -> None:
    """Upload Orion's Arm files to xAI Collections."""
    raw_dir = data_dir / "raw-more" / "orionsarm"
    config_file = data_dir / "orionsarm_collection.json"

    if not raw_dir.exists():
        raise click.UsageError(f"Orion's Arm directory does not exist: {raw_dir}")

    # Collect all files (html, txt, etc.)
    all_files: List[Tuple[Path, dict]] = []
    for f in sorted(raw_dir.iterdir()):
        if f.is_file():
            all_files.append((f, {"filename": f.name}))

    if not all_files:
        click.echo("No files found to upload.")
        return

    click.echo(f"Found {len(all_files)} files")

    # Check if we have an existing collection
    collection_id = None
    uploaded_files: Set[str] = set()
    if config_file.exists():
        config = json.loads(config_file.read_text(encoding="utf-8"))
        collection_id = config.get("collection_id")
        uploaded_files = set(config.get("uploaded_files", []))
        click.echo(f"Found existing collection: {collection_id}")
        click.echo(f"Already uploaded: {len(uploaded_files)} files")

    # Create collection if needed
    if collection_id is None:
        click.echo(f"Creating collection '{collection_name}'...")
        collection_id = xai.create_collection(
            name=collection_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            field_definitions=[
                {"key": "filename", "required": True, "unique": False, "inject_into_chunk": False},
            ],
        )
        click.echo(f"Created collection: {collection_id}")

        config = {"collection_id": collection_id, "collection_name": collection_name, "uploaded_files": []}
        config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")

    click.echo(f"\nTotal files: {len(all_files)}, Already uploaded: {len(uploaded_files)}")
    uploaded_count, failed_count, newly_uploaded = asyncio.run(
        upload_files_async(
            collection_id=collection_id,
            files=all_files,
            uploaded_files=uploaded_files,
            concurrency=concurrency,
            wait_for_indexing=wait_for_indexing,
            resume=resume,
        )
    )

    # Update config with newly uploaded files
    if newly_uploaded:
        uploaded_files.update(newly_uploaded)
        config = json.loads(config_file.read_text(encoding="utf-8"))
        config["uploaded_files"] = list(uploaded_files)
        config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")

    click.echo(f"\nDone! Uploaded: {uploaded_count}, Failed: {failed_count}, Total in collection: {len(uploaded_files)}")
    click.echo(f"Collection ID: {collection_id}")
    if not wait_for_indexing and uploaded_count > 0:
        click.echo("\nNote: Documents were uploaded without waiting for indexing.")


def run_grg(
    data_dir: Path,
    collection_name: str = "grg-mailing-list",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    resume: bool = True,
    concurrency: int = 100,
    wait_for_indexing: bool = False,
) -> None:
    """Upload GRG (Gerontology Research Group) mailing list files to xAI Collections."""
    raw_dir = data_dir / "raw-more" / "grg"
    config_file = data_dir / "grg_collection.json"

    if not raw_dir.exists():
        raise click.UsageError(f"GRG directory does not exist: {raw_dir}")

    # Collect all files
    all_files: List[Tuple[Path, dict]] = []
    for f in sorted(raw_dir.iterdir()):
        if f.is_file():
            all_files.append((f, {"filename": f.name}))

    if not all_files:
        click.echo("No files found to upload.")
        return

    click.echo(f"Found {len(all_files)} files")

    # Check if we have an existing collection
    collection_id = None
    uploaded_files: Set[str] = set()
    if config_file.exists():
        config = json.loads(config_file.read_text(encoding="utf-8"))
        collection_id = config.get("collection_id")
        uploaded_files = set(config.get("uploaded_files", []))
        click.echo(f"Found existing collection: {collection_id}")
        click.echo(f"Already uploaded: {len(uploaded_files)} files")

    # Create collection if needed
    if collection_id is None:
        click.echo(f"Creating collection '{collection_name}'...")
        collection_id = xai.create_collection(
            name=collection_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            field_definitions=[
                {"key": "filename", "required": True, "unique": False, "inject_into_chunk": False},
            ],
        )
        click.echo(f"Created collection: {collection_id}")

        config = {"collection_id": collection_id, "collection_name": collection_name, "uploaded_files": []}
        config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")

    click.echo(f"\nTotal files: {len(all_files)}, Already uploaded: {len(uploaded_files)}")
    uploaded_count, failed_count, newly_uploaded = asyncio.run(
        upload_files_async(
            collection_id=collection_id,
            files=all_files,
            uploaded_files=uploaded_files,
            concurrency=concurrency,
            wait_for_indexing=wait_for_indexing,
            resume=resume,
        )
    )

    # Update config with newly uploaded files
    if newly_uploaded:
        uploaded_files.update(newly_uploaded)
        config = json.loads(config_file.read_text(encoding="utf-8"))
        config["uploaded_files"] = list(uploaded_files)
        config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")

    click.echo(f"\nDone! Uploaded: {uploaded_count}, Failed: {failed_count}, Total in collection: {len(uploaded_files)}")
    click.echo(f"Collection ID: {collection_id}")
    if not wait_for_indexing and uploaded_count > 0:
        click.echo("\nNote: Documents were uploaded without waiting for indexing.")


def run_fightaging(
    data_dir: Path,
    collection_name: str = "fightaging-articles",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    resume: bool = True,
    concurrency: int = 100,
    wait_for_indexing: bool = False,
) -> None:
    """Upload Fight Aging! HTML files to xAI Collections."""
    fightaging_dir = data_dir / "raw-more" / "fightaging.org"
    newsletters_dir = fightaging_dir / "newsletters"
    pages_dir = fightaging_dir / "pages"
    config_file = data_dir / "fightaging_collection.json"

    if not fightaging_dir.exists():
        raise click.UsageError(f"Fight Aging! directory does not exist: {fightaging_dir}")

    # Collect all HTML files with their source type
    html_files: List[Tuple[Path, dict]] = []
    
    if newsletters_dir.exists():
        for f in sorted(newsletters_dir.glob("*.html")):
            html_files.append((f, {"source_type": "newsletters", "filename": f.name}))
        click.echo(f"Found {len([f for f, _ in html_files if 'newsletters' in str(f)])} newsletter files")

    if pages_dir.exists():
        for f in sorted(pages_dir.glob("*.html")):
            html_files.append((f, {"source_type": "pages", "filename": f.name}))
        click.echo(f"Found {len([f for f, fields in html_files if fields.get('source_type') == 'pages'])} page files")

    if not html_files:
        click.echo("No HTML files found to upload.")
        return

    click.echo(f"Total HTML files: {len(html_files)}")

    # Check if we have an existing collection
    collection_id = None
    uploaded_files: Set[str] = set()
    if config_file.exists():
        config = json.loads(config_file.read_text(encoding="utf-8"))
        collection_id = config.get("collection_id")
        uploaded_files = set(config.get("uploaded_files", []))
        click.echo(f"Found existing collection: {collection_id}")
        click.echo(f"Already uploaded: {len(uploaded_files)} files")

    # Create collection if needed
    if collection_id is None:
        click.echo(f"Creating collection '{collection_name}'...")
        collection_id = xai.create_collection(
            name=collection_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            field_definitions=[
                {"key": "source_type", "required": True, "unique": False, "inject_into_chunk": True},
                {"key": "filename", "required": True, "unique": False, "inject_into_chunk": False},
            ],
        )
        click.echo(f"Created collection: {collection_id}")

        config = {"collection_id": collection_id, "collection_name": collection_name, "uploaded_files": []}
        config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")

    # File key function for fightaging (includes source_type)
    def fightaging_file_key(f: Path, fields: dict) -> str:
        return f"{fields['source_type']}/{f.name}"

    click.echo(f"\nTotal files: {len(html_files)}, Already uploaded: {len(uploaded_files)}")
    uploaded_count, failed_count, newly_uploaded = asyncio.run(
        upload_files_async(
            collection_id=collection_id,
            files=html_files,
            uploaded_files=uploaded_files,
            concurrency=concurrency,
            wait_for_indexing=wait_for_indexing,
            resume=resume,
            file_key_fn=fightaging_file_key,
        )
    )

    # Update config with newly uploaded files
    if newly_uploaded:
        uploaded_files.update(newly_uploaded)
        config = json.loads(config_file.read_text(encoding="utf-8"))
        config["uploaded_files"] = list(uploaded_files)
        config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")

    click.echo(f"\nDone! Uploaded: {uploaded_count}, Failed: {failed_count}, Total in collection: {len(uploaded_files)}")
    click.echo(f"Collection ID: {collection_id}")
    if not wait_for_indexing and uploaded_count > 0:
        click.echo("\nNote: Documents were uploaded without waiting for indexing.")
