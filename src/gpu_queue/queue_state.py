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


def move_pending_job_to_staging(queue: dict[str, list], job_id: str) -> bool:
    for i, job in enumerate(queue["pending"]):
        if job["id"] == job_id:
            moved = queue["pending"].pop(i)
            now = datetime.now().isoformat()
            moved["added"] = now
            moved["staged_at"] = now
            queue["staging"].insert(0, moved)
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


def move_pending_jobs(queue: dict[str, list], job_ids: list[str], offset: int) -> bool:
    """Move multiple pending jobs together by one row, preserving relative order."""
    if offset not in (-1, 1):
        return False
    pending = queue["pending"]
    if not pending or not job_ids:
        return False

    wanted = {str(job_id) for job_id in job_ids}
    indexed = [
        i for i, job in enumerate(pending) if job.get("id") is not None and str(job.get("id")) in wanted
    ]
    if not indexed:
        return False

    if offset < 0 and indexed[0] == 0:
        return False
    if offset > 0 and indexed[-1] == len(pending) - 1:
        return False

    if offset < 0:
        for idx in indexed:
            pending[idx - 1], pending[idx] = pending[idx], pending[idx - 1]
        return True

    for idx in reversed(indexed):
        pending[idx + 1], pending[idx] = pending[idx], pending[idx + 1]
    return True
