"""Pandoc HTML generation adapter."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def generate_html(md_file: Path, html_file: Path, css_file: str = "wrap.css") -> bool:
    """Generate HTML from markdown using pandoc.
    
    Returns True on success, False on failure.
    """
    try:
        pandoc_cmd = [
            "pandoc",
            "-f", "markdown+autolink_bare_uris",
            "-s",
            "-c", css_file,
            str(md_file),
            "-o", str(html_file)
        ]
        subprocess.run(pandoc_cmd, check=True, capture_output=True)
        # Ensure HTML file is synced to disk
        if html_file.exists():
            with html_file.open("a") as f:
                os.fsync(f.fileno())
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        return False
