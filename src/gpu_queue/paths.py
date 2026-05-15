from __future__ import annotations

from pathlib import Path

QUEUE_DIR = Path.home() / ".gpu_queue"
QUEUE_FILE = QUEUE_DIR / "jobs.json"
PID_FILE = QUEUE_DIR / "daemon.pid"
DAEMON_LOG = QUEUE_DIR / "daemon.log"
LOG_DIR = QUEUE_DIR / "logs"
LOCK_FILE = QUEUE_DIR / "queue.lock"

POLL_INTERVAL = 2
MIN_FREE_GPUS = 2
SERVER_PORT = 12345


def get_server_url() -> str:
    return f"http://localhost:{SERVER_PORT}"
