from __future__ import annotations

from datetime import datetime

from gpu_queue import paths


def log_msg(msg: str, verbose: bool = False) -> None:
    """Log a message to the daemon log."""
    if verbose:
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    with open(paths.DAEMON_LOG, "a") as f:
        f.write(line)
