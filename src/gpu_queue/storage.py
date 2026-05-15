from __future__ import annotations

import fcntl
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from gpu_queue import paths
from gpu_queue.domain import QueueState
from gpu_queue.logs import log_msg
from gpu_queue.queue_state import empty_queue, load_queue_file, save_queue_file


def ensure_dirs() -> None:
    """Create queue directories if they don't exist."""
    paths.QUEUE_DIR.mkdir(exist_ok=True)
    paths.LOG_DIR.mkdir(exist_ok=True)


class JsonQueueStore:
    def __init__(
        self,
        queue_file: Path | None = None,
        lock_file: Path | None = None,
        queue_dir: Path | None = None,
        log_dir: Path | None = None,
    ) -> None:
        self.queue_file = queue_file or paths.QUEUE_FILE
        self.lock_file = lock_file or paths.LOCK_FILE
        self.queue_dir = queue_dir or self.queue_file.parent
        self.log_dir = log_dir or paths.LOG_DIR

    def ensure_dirs(self) -> None:
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> QueueState:
        try:
            return load_queue_file(self.queue_file)
        except (json.JSONDecodeError, ValueError) as e:
            log_msg(f"Error loading queue JSON: {e}")
            return empty_queue()

    def save(self, queue: QueueState) -> None:
        save_queue_file(self.queue_file, queue)

    @contextmanager
    def transaction(self) -> Iterator[QueueState]:
        self.ensure_dirs()
        with open(self.lock_file, "w") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX)
                queue = self.load()
                yield queue
                self.save(queue)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)


def get_default_store() -> JsonQueueStore:
    return JsonQueueStore()


@contextmanager
def locked_queue() -> Iterator[QueueState]:
    """Context manager for thread-safe and process-safe queue access."""
    with get_default_store().transaction() as queue:
        yield queue


def load_queue_raw() -> QueueState:
    """Load the job queue from disk without locking."""
    return get_default_store().load()


def save_queue_raw(queue: QueueState) -> None:
    """Save the job queue to disk without locking (atomic replace)."""
    get_default_store().save(queue)


def load_queue() -> QueueState:
    """Load the job queue (backward compatibility, no lock)."""
    return load_queue_raw()


def save_queue(queue: QueueState) -> None:
    """Save the job queue (backward compatibility, no lock)."""
    save_queue_raw(queue)
