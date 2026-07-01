"""SCP file upload adapter."""

from __future__ import annotations

import subprocess
from pathlib import Path

# Reuse a single SSH connection across all uploads/commands instead of paying a
# fresh TCP+SSH handshake per file. The first connection opens a master socket;
# subsequent scp/ssh calls (including concurrent ones) multiplex over it, which
# turns dozens of serial per-file handshakes into one. ControlPersist keeps the
# master alive briefly between calls; %C is a short hash of the connection tuple.
_SSH_MUX_OPTS = [
    "-o", "ControlMaster=auto",
    "-o", "ControlPath=/tmp/.hpluslogs-ssh-%C",
    "-o", "ControlPersist=120",
]


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
        scp_cmd = ["scp", "-p", *_SSH_MUX_OPTS, str(local_file), remote_dest]
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
        mkdir_cmd = ["ssh", *_SSH_MUX_OPTS, f"{remote_user}@{remote_host}", f"mkdir -p {remote_path}"]
        subprocess.run(mkdir_cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        return False
