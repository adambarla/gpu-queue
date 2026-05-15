from __future__ import annotations

import copy
from collections.abc import Callable, Iterable
from datetime import datetime

from gpu_queue.domain import Job
from gpu_queue.ids import generate_job_id
from gpu_queue.ports import QueueStore
from gpu_queue.queue_state import (
    cancel_staged_job,
    insert_staged_job,
    move_pending_job,
    move_pending_job_to_staging,
    move_pending_jobs,
    send_staged_job_to_pending,
    stage_completed_retry,
)
from gpu_queue.storage import get_default_store


class QueueService:
    def __init__(
        self,
        store: QueueStore | None = None,
        id_factory: Callable[[], str] = generate_job_id,
        now_factory: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.store = store or get_default_store()
        self.id_factory = id_factory
        self.now_factory = now_factory

    def _now(self) -> str:
        return self.now_factory().isoformat()

    def _make_staged_job(self, cmd: str = "", gpus: int = 1) -> Job:
        now = self._now()
        return {
            "id": self.id_factory(),
            "cmd": cmd,
            "gpus": gpus,
            "added": now,
            "staged_at": now,
        }

    def snapshot(self) -> dict[str, list[Job]]:
        return self.store.load()

    def add_job(
        self,
        command: str,
        gpus: int = 2,
        priority: str = "medium",
        front: bool = False,
        cwd: str | None = None,
    ) -> Job:
        priorities = {"low": 0, "medium": 1, "high": 2}
        prio = 3 if front else priorities.get(priority, 1)
        job: Job = {
            "id": self.id_factory(),
            "cmd": command,
            "gpus": gpus,
            "added": self._now(),
            "priority": prio,
        }
        if cwd is not None:
            job["cwd"] = cwd

        with self.store.transaction() as queue:
            if front:
                queue["pending"].insert(0, job)
            else:
                queue["pending"].append(job)
        return copy.deepcopy(job)

    def stage_new_job(self, cmd: str = "", gpus: int = 1) -> Job:
        job = self._make_staged_job(cmd, gpus)
        with self.store.transaction() as queue:
            insert_staged_job(queue, job)
        return copy.deepcopy(job)

    def duplicate_jobs(self, job_ids: Iterable[str]) -> list[Job]:
        created: list[Job] = []
        wanted = [str(job_id) for job_id in job_ids]
        if not wanted:
            return created

        with self.store.transaction() as queue:
            jobs_by_id = {
                str(job.get("id")): job
                for key in ["running", "pending", "staging", "completed"]
                for job in queue.get(key, [])
                if job.get("id") is not None
            }
            for job_id in wanted:
                job = jobs_by_id.get(job_id)
                if job is None:
                    continue
                dup_job = self._make_staged_job(
                    str(job.get("cmd", "")), int(job.get("gpus", 1))
                )
                insert_staged_job(queue, dup_job)
                created.append(copy.deepcopy(dup_job))
        return created

    def update_staged_job(
        self, job_id: str, cmd: str | None = None, gpus: int | None = None
    ) -> bool:
        with self.store.transaction() as queue:
            for job in queue["staging"]:
                if job.get("id") == job_id:
                    if cmd is not None:
                        job["cmd"] = cmd
                    if gpus is not None:
                        job["gpus"] = gpus
                    return True
        return False

    def send_staged_to_pending(self, job_id: str) -> bool:
        with self.store.transaction() as queue:
            return send_staged_job_to_pending(queue, job_id)

    def move_pending_to_staging(self, job_id: str) -> bool:
        with self.store.transaction() as queue:
            return move_pending_job_to_staging(queue, job_id)

    def move_pending_to_staging_bulk(self, job_ids: Iterable[str]) -> list[str]:
        requested = [str(job_id) for job_id in job_ids]
        moved: list[str] = []
        with self.store.transaction() as queue:
            for job_id in reversed(requested):
                if move_pending_job_to_staging(queue, job_id):
                    moved.append(job_id)
        moved_set = set(moved)
        return [job_id for job_id in requested if job_id in moved_set]

    def cancel_staged(self, job_id: str) -> bool:
        with self.store.transaction() as queue:
            return cancel_staged_job(queue, job_id)

    def cancel_pending(self, job_id: str) -> bool:
        with self.store.transaction() as queue:
            for i, job in enumerate(queue["pending"]):
                if job.get("id") == job_id:
                    cancelled = queue["pending"].pop(i)
                    cancelled["status"] = "cancelled"
                    cancelled["ended"] = self._now()
                    queue["completed"].insert(0, cancelled)
                    return True
        return False

    def delete_completed(self, job_id: str) -> bool:
        with self.store.transaction() as queue:
            for i, job in enumerate(queue["completed"]):
                if job.get("id") == job_id:
                    queue["completed"].pop(i)
                    return True
        return False

    def stage_retry(self, job_id: str) -> Job | None:
        with self.store.transaction() as queue:
            for job in queue["completed"]:
                if job.get("id") == job_id:
                    new_job = self._make_staged_job(
                        str(job.get("cmd", "")), int(job.get("gpus", 1))
                    )
                    if stage_completed_retry(queue, job_id, new_job):
                        return copy.deepcopy(new_job)
                    return None
        return None

    def requeue_completed(
        self, job_id: str, front: bool = False, preserve_id: bool = True
    ) -> Job | None:
        with self.store.transaction() as queue:
            for i, job in enumerate(queue["completed"]):
                if job.get("id") == job_id:
                    queue["completed"].pop(i)
                    new_job: Job = {
                        "id": str(job.get("id")) if preserve_id else self.id_factory(),
                        "cmd": str(job.get("cmd", "")),
                        "gpus": int(job.get("gpus", 1)),
                        "added": self._now(),
                        "retried_at": self._now(),
                        "priority": 1,
                    }
                    if front:
                        queue["pending"].insert(0, new_job)
                    else:
                        queue["pending"].append(new_job)
                    return copy.deepcopy(new_job)
        return None

    def pause_running_to_pending(self, job_id: str) -> Job | None:
        with self.store.transaction() as queue:
            for i, job in enumerate(queue["running"]):
                if job.get("id") == job_id:
                    queue["running"].pop(i)
                    new_job: Job = {
                        "id": self.id_factory(),
                        "cmd": str(job.get("cmd", "")),
                        "gpus": int(job.get("gpus", 1)),
                        "added": self._now(),
                        "priority": 3,
                        "paused_from": str(job.get("id")),
                    }
                    queue["pending"].insert(0, new_job)
                    return copy.deepcopy(new_job)
        return None

    def move_pending(self, job_id: str, offset: int) -> bool:
        with self.store.transaction() as queue:
            return move_pending_job(queue, job_id, offset)

    def move_pending_bulk(self, job_ids: list[str], offset: int) -> bool:
        with self.store.transaction() as queue:
            return move_pending_jobs(queue, job_ids, offset)
