"""Adapter for the diyhpl.us wiki source repository.

The diyhpluswiki (http://diyhpl.us/wiki/) is an ikiwiki instance whose source
lives in a plain git repository of ``.mdwn`` markdown files.  Fetching it is a
matter of cloning/updating that repository, so this adapter shells out to
``git`` rather than scraping rendered HTML.

Only the text sources matter for RAG purposes: ``.mdwn`` pages plus the few
``.md``/``.txt`` notes.  Images and other assets are skipped by
:data:`PAGE_SUFFIXES`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Tuple

WIKI_REPO_URL = "git://diyhpl.us/diyhpluswiki"

# Text sources worth indexing. ikiwiki uses .mdwn for most pages.
PAGE_SUFFIXES = (".mdwn", ".md", ".txt")


def _run_git(args: List[str], timeout: int = 120) -> str:
    """Run a git command and return stdout, raising RuntimeError on failure."""
    proc = subprocess.run(
        ["git"] + args, capture_output=True, text=True, timeout=timeout
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return proc.stdout.strip()


def clone_or_update(
    dest: Path, repo_url: str = WIKI_REPO_URL, timeout: int = 120
) -> str:
    """Clone the wiki repository into ``dest``, or update an existing checkout.

    Returns the short commit hash of the resulting checkout.  An existing
    checkout is updated with a fast-forward pull; if that fails (offline, for
    example) the current copy is kept and a warning is reported through the
    return value's caller-visible logs rather than aborting.
    """
    if dest.exists() and (dest / ".git").exists():
        try:
            _run_git(["-C", str(dest), "pull", "--ff-only"], timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - stale checkout is still usable
            print(f"Could not update {dest}: {exc}")
        return _run_git(["-C", str(dest), "rev-parse", "--short", "HEAD"], timeout=timeout)

    dest.parent.mkdir(parents=True, exist_ok=True)
    _run_git(["clone", "--depth", "1", repo_url, str(dest)], timeout=timeout)
    return _run_git(["-C", str(dest), "rev-parse", "--short", "HEAD"], timeout=timeout)


def is_page(path: Path) -> bool:
    """True when ``path`` is a nonempty, indexable wiki text source."""
    return path.is_file() and path.suffix.lower() in PAGE_SUFFIXES and path.stat().st_size > 0


def iter_pages(root: Path) -> List[Tuple[Path, str]]:
    """Return ``(path, relative_name)`` pairs for every wiki page under ``root``.

    ``relative_name`` is the path relative to ``root``, which is stable across
    re-clones and unique enough to key the uploaded-files bookkeeping.
    """
    pages: List[Tuple[Path, str]] = []
    for path in sorted(root.rglob("*")):
        if is_page(path):
            pages.append((path, str(path.relative_to(root))))
    return pages
