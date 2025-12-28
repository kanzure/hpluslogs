"""Download service for fetching IRC logs."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Optional

import click

from hpluslogs.core.utils import daterange, ensure_directory
from hpluslogs.integrations import gnusha


def run(
    data_dir: Path,
    start_date: Optional[_dt.datetime],
    end_date: Optional[_dt.datetime],
    resume: bool = True,
) -> None:
    """Download raw IRC logs from gnusha.org for a date range.

    The logs are saved in the ``data_dir/raw`` folder with filenames like
    ``YYYY-MM-DD.log``.  If ``resume`` is enabled (the default) then
    existing files will not be re‑downloaded.
    """
    raw_dir = data_dir / "raw"
    ensure_directory(raw_dir)
    
    today = _dt.date.today()
    if start_date is None:
        start = today
    else:
        start = start_date.date()
    if end_date is None:
        end = start
    else:
        end = end_date.date()
    
    for day in daterange(start, end):
        fname = raw_dir / f"{day.isoformat()}.log"
        if fname.exists() and resume:
            click.echo(f"✓ {fname.name} exists, skipping")
            continue
        click.echo(f"Downloading {day} …")
        try:
            text = gnusha.fetch_log(day)
        except Exception as e:
            click.echo(f"Error downloading {day}: {e}")
            continue
        fname.write_text(text, encoding="utf-8")
        click.echo(f"Saved {fname}")
