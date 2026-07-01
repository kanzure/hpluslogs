"""Publishing service - output files, generate HTML, upload to server."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Optional

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
) -> Optional[Path]:
    """Output content to a markdown file, generate HTML, and optionally upload.
    
    Returns the path to the generated markdown file, or None if not created.
    """
    outputs_dir = data_dir / "outputs"
    ensure_directory(outputs_dir)
    
    # Generate filename
    if output_name:
        base_name = output_name
    else:
        timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        base_name = f"{prefix}-{timestamp}"
    
    md_file = outputs_dir / f"{base_name}.md"

    # Write content and upload the markdown immediately (before HTML generation).
    md_file.write_text(content, encoding="utf-8")
    if upload:
        scp.ensure_remote_directory(remote_user, remote_host, remote_path)
        scp.upload_file(md_file, remote_user, remote_host, remote_path, md_file.name)

    # Generate HTML, then upload it.
    html_file = outputs_dir / f"{base_name}.html"
    pandoc.generate_html(md_file, html_file, css_file)
    if upload:
        scp.upload_file(html_file, remote_user, remote_host, remote_path, html_file.name)

    return md_file


def output_context(
    data_dir: Path,
    context: str,
    query: str,
    output_name: Optional[str] = None,
    css_file: str = "wrap.css",
    upload: bool = True,
    remote_user: str = "bryan",
    remote_host: str = "gnusha.org",
    remote_path: str = "~/public_html/irc/chatgpt/hplusroadmap/",
    prefix: str = "query",
) -> Optional[Path]:
    """Output context to a .context.md file and optionally upload.
    
    Returns the path to the generated markdown file, or None if not created.
    """
    outputs_dir = data_dir / "outputs"
    ensure_directory(outputs_dir)
    
    # Generate filename
    if output_name:
        base_name = output_name
    else:
        timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        base_name = f"{prefix}-{timestamp}"
    
    context_name = f"{base_name}.context"
    md_file = outputs_dir / f"{context_name}.md"
    
    # Build context document
    content = f"# Context for: {query}\n\n{context}"
    md_file.write_text(content, encoding="utf-8")
    
    # Upload if requested (only .md, no HTML generation)
    if upload:
        scp.ensure_remote_directory(remote_user, remote_host, remote_path)
        scp.upload_file(md_file, remote_user, remote_host, remote_path, md_file.name)
    
    return md_file
