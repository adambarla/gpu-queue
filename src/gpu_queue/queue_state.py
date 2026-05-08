from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def empty_queue() -> dict[str, list]:
    return {"staging": [], "pending": [], "running": [], "completed": []}


def normalize_queue(raw: Any) -> dict[str, list]:
    queue = empty_queue()
    if not isinstance(raw, dict):
        return queue
    for key in queue:
        val = raw.get(key, [])
        queue[key] = val if isinstance(val, list) else []
    return queue


def load_queue_file(queue_file: Path) -> dict[str, list]:
    if not queue_file.exists() or queue_file.stat().st_size == 0:
        return empty_queue()
    with open(queue_file) as f:
        return normalize_queue(json.load(f))


def save_queue_file(queue_file: Path, queue: dict[str, list]) -> None:
    queue_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = queue_file.with_suffix(".tmp")
    with open(temp_file, "w") as f:
        json.dump(queue, f, indent=2)
    temp_file.replace(queue_file)


def make_staged_job(job_id: str, cmd: str = "", gpus: int = 1) -> dict[str, Any]:
    now = datetime.now().isoformat()
    return {
        "id": job_id,
        "cmd": cmd,
        "gpus": gpus,
        "added": now,
        "staged_at": now,
    }


def insert_staged_job(queue: dict[str, list], job: dict[str, Any]) -> None:
    queue["staging"].insert(0, job)


def stage_completed_job(queue: dict[str, list], job: dict[str, Any]) -> None:
    queue["staging"].insert(0, job)


def send_staged_job_to_pending(queue: dict[str, list], job_id: str) -> bool:
    for i, job in enumerate(queue["staging"]):
        if job["id"] == job_id:
            moved = queue["staging"].pop(i)
            moved["added"] = datetime.now().isoformat()
            queue["pending"].append(moved)
            return True
    return False


def cancel_staged_job(queue: dict[str, list], job_id: str) -> bool:
    for i, job in enumerate(queue["staging"]):
        if job["id"] == job_id:
            cancelled = queue["staging"].pop(i)
            cancelled["status"] = "cancelled"
            cancelled["ended"] = datetime.now().isoformat()
            queue["completed"].insert(0, cancelled)
            return True
    return False


def stage_completed_retry(queue: dict[str, list], job_id: str, new_job: dict[str, Any]) -> bool:
    for i, job in enumerate(queue["completed"]):
        if job["id"] == job_id:
            queue["completed"].pop(i)
            queue["staging"].insert(0, new_job)
            return True
    return False


def move_pending_job(queue: dict[str, list], job_id: str, offset: int) -> bool:
    pending = queue["pending"]
    idx = -1
    for i, job in enumerate(pending):
        if job["id"] == job_id:
            idx = i
            break
    if idx == -1:
        return False
    new_idx = idx + offset
    if new_idx < 0 or new_idx >= len(pending):
        return False
    pending[idx], pending[new_idx] = pending[new_idx], pending[idx]
    return True
