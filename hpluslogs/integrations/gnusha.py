"""Adapter for fetching IRC logs from gnusha.org."""

from __future__ import annotations

import datetime as _dt

import requests


def fetch_log(date: _dt.date) -> str:
    """Download a single day's IRC log as text.

    The logs are hosted at ``https://gnusha.org/logs/`` with filenames
    formatted as ``YYYY-MM-DD.log``.  This function retrieves the
    plain text version.  It raises ``requests.HTTPError`` on failure.
    """
    url = f"https://gnusha.org/logs/{date.isoformat()}.log"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text
