"""SCP file upload adapter."""

from __future__ import annotations

import subprocess
from pathlib import Path


def upload_file(
    local_file: Path,
    remote_user: str,
    remote_host: str,
    remote_path: str,
    remote_filename: str,
) -> bool:
    """Upload a file via scp.
    
    Returns True on success, False on failure.
    """
    remote_dest = f"{remote_user}@{remote_host}:{remote_path}{remote_filename}"
    try:
        scp_cmd = ["scp", "-p", str(local_file), remote_dest]
        subprocess.run(scp_cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        return False


def ensure_remote_directory(
    remote_user: str,
    remote_host: str,
    remote_path: str,
) -> bool:
    """Ensure remote directory exists via ssh mkdir.
    
    Returns True on success, False on failure.
    """
    try:
        mkdir_cmd = ["ssh", f"{remote_user}@{remote_host}", f"mkdir -p {remote_path}"]
        subprocess.run(mkdir_cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        return False
