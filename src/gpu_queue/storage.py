from __future__ import annotations

import fcntl
import json
from contextlib import contextmanager
from typing import Iterator

from gpu_queue import paths
from gpu_queue.logs import log_msg
from gpu_queue.queue_state import empty_queue, load_queue_file, save_queue_file


def ensure_dirs() -> None:
    """Create queue directories if they don't exist."""
    paths.QUEUE_DIR.mkdir(exist_ok=True)
    paths.LOG_DIR.mkdir(exist_ok=True)


@contextmanager
def locked_queue() -> Iterator[dict[str, list]]:
    """Context manager for thread-safe and process-safe queue access."""
    ensure_dirs()
    with open(paths.LOCK_FILE, "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            queue = load_queue_raw()
            yield queue
            save_queue_raw(queue)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def load_queue_raw() -> dict[str, list]:
    """Load the job queue from disk without locking."""
    try:
        return load_queue_file(paths.QUEUE_FILE)
    except (json.JSONDecodeError, ValueError) as e:
        log_msg(f"Error loading queue JSON: {e}")
        return empty_queue()


def save_queue_raw(queue: dict[str, list]) -> None:
    """Save the job queue to disk without locking (atomic replace)."""
    save_queue_file(paths.QUEUE_FILE, queue)


def load_queue() -> dict[str, list]:
    """Load the job queue (backward compatibility, no lock)."""
    return load_queue_raw()


def save_queue(queue: dict[str, list]) -> None:
    """Save the job queue (backward compatibility, no lock)."""
    save_queue_raw(queue)
