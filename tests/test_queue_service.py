from __future__ import annotations

import copy
import unittest
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator

from gpu_queue.domain import QueueState
from gpu_queue.queue_state import empty_queue
from gpu_queue.service import QueueService


class MemoryQueueStore:
    def __init__(self, queue: QueueState | None = None) -> None:
        self.queue = copy.deepcopy(queue or empty_queue())

    def load(self) -> QueueState:
        return copy.deepcopy(self.queue)

    def save(self, queue: QueueState) -> None:
        self.queue = copy.deepcopy(queue)

    @contextmanager
    def transaction(self) -> Iterator[QueueState]:
        queue = self.load()
        yield queue
        self.save(queue)


class QueueServiceTest(unittest.TestCase):
    def service(
        self, queue: QueueState | None = None
    ) -> tuple[QueueService, MemoryQueueStore]:
        ids = iter(["j1", "j2", "j3", "j4"])
        store = MemoryQueueStore(queue)
        service = QueueService(
            store=store,
            id_factory=lambda: next(ids),
            now_factory=lambda: datetime(2026, 1, 2, 3, 4, 5),
        )
        return service, store

    def test_add_job_front_uses_priority_and_head_insert(self):
        service, store = self.service()

        service.add_job("first", gpus=2, priority="low")
        front_job = service.add_job("urgent", gpus=1, front=True)

        self.assertEqual(front_job["priority"], 3)
        self.assertEqual(
            [job["cmd"] for job in store.queue["pending"]], ["urgent", "first"]
        )

    def test_duplicate_jobs_stages_copies_in_requested_order(self):
        service, store = self.service(
            {
                "staging": [],
                "pending": [
                    {"id": "p1", "cmd": "cmd 1", "gpus": 1},
                    {"id": "p2", "cmd": "cmd 2", "gpus": 2},
                ],
                "running": [],
                "completed": [],
            }
        )

        created = service.duplicate_jobs(["p1", "p2"])

        self.assertEqual([job["id"] for job in created], ["j1", "j2"])
        self.assertEqual(
            [(job["cmd"], job["gpus"]) for job in store.queue["staging"]],
            [("cmd 2", 2), ("cmd 1", 1)],
        )

    def test_cancel_pending_moves_to_completed_with_status(self):
        service, store = self.service(
            {
                "staging": [],
                "pending": [{"id": "p1", "cmd": "cmd", "gpus": 1}],
                "running": [],
                "completed": [],
            }
        )

        self.assertTrue(service.cancel_pending("p1"))

        self.assertEqual(store.queue["pending"], [])
        self.assertEqual(store.queue["completed"][0]["id"], "p1")
        self.assertEqual(store.queue["completed"][0]["status"], "cancelled")
        self.assertEqual(store.queue["completed"][0]["ended"], "2026-01-02T03:04:05")

    def test_stage_retry_creates_new_staged_job_and_removes_completed(self):
        service, store = self.service(
            {
                "staging": [],
                "pending": [],
                "running": [],
                "completed": [{"id": "c1", "cmd": "eval", "gpus": 4}],
            }
        )

        retry = service.stage_retry("c1")

        self.assertIsNotNone(retry)
        self.assertEqual(store.queue["completed"], [])
        self.assertEqual(store.queue["staging"][0]["id"], "j1")
        self.assertEqual(store.queue["staging"][0]["cmd"], "eval")

    def test_bulk_pending_move_preserves_relative_order(self):
        service, store = self.service(
            {
                "staging": [],
                "pending": [
                    {"id": "p0"},
                    {"id": "p1"},
                    {"id": "p2"},
                    {"id": "p3"},
                ],
                "running": [],
                "completed": [],
            }
        )

        self.assertTrue(service.move_pending_bulk(["p1", "p2"], 1))

        self.assertEqual(
            [job["id"] for job in store.queue["pending"]], ["p0", "p3", "p1", "p2"]
        )
