"""Publishing service - file output, HTML generation, and upload."""

from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path
from typing import Optional

import click

from hpluslogs.core.utils import ensure_directory
from hpluslogs.integrations import pandoc, scp


def output(
    data_dir: Path,
    content: str,
    output_name: Optional[str] = None,
    css_file: str = "wrap.css",
    upload: bool = True,
    remote_user: str = "bryan",
    remote_host: str = "gnusha.org",
    remote_path: str = "~/public_html/irc/chatgpt/hplusroadmap/",
    prefix: str = "query",
) -> tuple[Path, Optional[Path]]:
    """Save content to markdown, generate HTML, and optionally upload.
    
    Returns tuple of (md_file_path, html_file_path or None).
    """
    output_dir = data_dir / "outputs"
    ensure_directory(output_dir)

    # Generate filename
    if output_name is None:
        timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{prefix}_{timestamp}"
    else:
        base_name = output_name

    md_file = output_dir / f"{base_name}.md"
    html_file = output_dir / f"{base_name}.md.html"

    # Write markdown file with explicit flush and sync
    with md_file.open("w", encoding="utf-8") as f:
        f.write(content.strip())
        f.flush()
        os.fsync(f.fileno())
    click.echo(f"\n✓ Saved markdown to {md_file}")

    # Generate HTML with pandoc
    html_generated = pandoc.generate_html(md_file, html_file, css_file)
    if html_generated:
        click.echo(f"✓ Generated HTML at {html_file}")
    else:
        click.echo("⚠ Warning: pandoc failed or not found. HTML not generated.")
        html_file = None

    # Upload files via scp
    if upload:
        # Ensure remote directory exists
        scp.ensure_remote_directory(remote_user, remote_host, remote_path)

        if scp.upload_file(md_file, remote_user, remote_host, remote_path, f"{base_name}.md"):
            click.echo(f"✓ Uploaded {md_file.name} to {remote_user}@{remote_host}:{remote_path}")
        else:
            click.echo(f"⚠ Warning: scp failed for markdown")

        if html_file and html_file.exists():
            if scp.upload_file(html_file, remote_user, remote_host, remote_path, f"{base_name}.md.html"):
                click.echo(f"✓ Uploaded {html_file.name} to {remote_user}@{remote_host}:{remote_path}")
            else:
                click.echo(f"⚠ Warning: scp failed for HTML")

    return (md_file, html_file)
