"""Download service for fetching IRC logs."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Iterable, Optional

import click

from hpluslogs.core.utils import daterange, ensure_directory
from hpluslogs.integrations import gnusha


DEFAULT_RECONCILE_START = _dt.date(2021, 1, 1)


def _existing_log_dates(raw_dir: Path) -> set[_dt.date]:
    """Return dates for raw logs named YYYY-MM-DD.log."""
    dates: set[_dt.date] = set()
    for path in raw_dir.glob("*.log"):
        try:
            dates.add(_dt.date.fromisoformat(path.stem))
        except ValueError:
            continue
    return dates


def _default_download_dates(
    raw_dir: Path,
    today: _dt.date,
    earliest_date: _dt.date = DEFAULT_RECONCILE_START,
) -> list[_dt.date]:
    """Find calendar dates that are missing raw logs.

    With no explicit range, reconcile the local raw log directory against the
    full calendar span from the later of the first downloaded log and the
    default cutoff through today. If there are no existing logs yet, default to
    today.
    """
    existing_dates = _existing_log_dates(raw_dir)
    if not existing_dates:
        return [today]

    first_log_date = max(min(existing_dates), earliest_date)
    return [
        day
        for day in daterange(first_log_date, today)
        if day not in existing_dates
    ]


def _explicit_download_dates(
    start_date: Optional[_dt.datetime],
    end_date: Optional[_dt.datetime],
    today: _dt.date,
) -> Iterable[_dt.date]:
    if start_date is None:
        start = today
    else:
        start = start_date.date()
    if end_date is None:
        end = start
    else:
        end = end_date.date()
    return daterange(start, end)


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

    if start_date is None and end_date is None:
        days = _default_download_dates(raw_dir, today)
        if not days:
            first_log_date = max(min(_existing_log_dates(raw_dir)), DEFAULT_RECONCILE_START)
            click.echo(f"All logs from {first_log_date} through {today} are already downloaded.")
            return
        click.echo(f"Found {len(days)} missing log(s) through {today}.")
    else:
        days = _explicit_download_dates(start_date, end_date, today)

    for day in days:
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
