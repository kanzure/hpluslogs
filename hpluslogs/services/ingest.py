"""Ingestion service for the diyhpl.us wiki source repository.

Wraps the ``diyhplus`` integration adapter: clone (or refresh) the wiki's git
repository under ``raw-more/diyhpl.us/`` so that ``diyhplus-collect`` can
submit the pages to an xAI collection.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import click

from hpluslogs.core.utils import ensure_directory
from hpluslogs.integrations import diyhplus


DIYHPLUS_SUBDIR = Path("raw-more") / "diyhpl.us"


def run_diyhplus(
    data_dir: Path,
    repo_url: Optional[str] = None,
    resume: bool = True,
) -> None:
    """Fetch/update the diyhplus wiki sources under ``data_dir``.

    With ``resume`` (the default) an existing checkout is refreshed in place
    with a fast-forward pull; ``--no-resume`` re-clones from scratch.
    """
    raw_dir = data_dir / DIYHPLUS_SUBDIR
    ensure_directory(raw_dir.parent)

    if not resume and raw_dir.exists():
        click.echo(f"Refreshing checkout from scratch: {raw_dir}")
        shutil.rmtree(raw_dir)

    kwargs = {}
    if repo_url:
        kwargs["repo_url"] = repo_url

    click.echo(f"Fetching diyhplus wiki sources into {raw_dir} …")
    commit = diyhplus.clone_or_update(raw_dir, **kwargs)

    pages = diyhplus.iter_pages(raw_dir)
    click.echo(f"Wiki checkout at commit {commit}: {len(pages)} indexable page(s)")
