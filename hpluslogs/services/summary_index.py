"""Summary index service - static HTML index of cached chat summaries.

Scans the canonical summary cache under ``data_dir/summaries/`` (written by
the summarize-day/week/month commands) and renders a single markdown page
listing the daily, weekly and monthly digests in reverse-chronological
order. The page is published through the shared publishing service so it
gets the same markdown + HTML + scp treatment as query outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from hpluslogs.services import publishing


def _summaries_dir(data_dir: Path) -> Path:
    return data_dir / "summaries"


def _collect_md(directory: Path, prefix: str) -> List[Tuple[str, str]]:
    """Return (label, published_basename) pairs for summaries, newest-first.

    ``prefix`` matches the naming used by the summarizer when it publishes
    each file (``daily-<label>``, ``weekly-<label>``, ``monthly-<label>``) so
    the links in the index resolve against the remote summaries directory.
    """
    if not directory.exists():
        return []
    entries = [(p.stem, f"{prefix}-{p.stem}") for p in sorted(directory.glob("*.md"))]
    entries.sort(key=lambda item: item[0], reverse=True)
    return entries


def _section(title: str, entries: List[Tuple[str, str]]) -> List[str]:
    lines = [f"## {title}", ""]
    if not entries:
        lines.append("None.")
    lines.extend(f"- [{label}]({base}.html)" for label, base in entries)
    lines.append("")
    return lines


def build_markdown(data_dir: Path) -> str:
    """Build the index markdown content from the cached summaries."""
    base = _summaries_dir(data_dir)
    daily = _collect_md(base / "daily", "daily")
    weekly = _collect_md(base / "weekly", "weekly")
    monthly = _collect_md(base / "monthly", "monthly")

    lines = ["# hplusroadmap chat summaries", ""]
    lines += _section("Monthly summaries", monthly)
    lines += _section("Weekly summaries", weekly)
    lines += _section("Daily summaries", daily)
    return "\n".join(lines).rstrip() + "\n"


def generate(
    data_dir: Path,
    upload: bool = True,
    remote_user: str = "bryan",
    remote_host: str = "gnusha.org",
    remote_path: str = "~/public_html/irc/chatgpt/hplusroadmap/chatsummaries/",
) -> Optional[Path]:
    """Generate index.md/index.html for the cached summaries and publish it.

    Returns the path of the written markdown file.
    """
    content = build_markdown(data_dir)
    return publishing.output(
        data_dir, content, output_name="index", upload=upload,
        remote_user=remote_user, remote_host=remote_host, remote_path=remote_path,
        prefix="chatsummaries",
    )
