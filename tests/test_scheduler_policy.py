import unittest

from gpu_queue.domain import GpuSnapshot, QueueState, SchedulerConfig
from gpu_queue.policies import FifoSchedulerPolicy


def _gpu(index: int, free: bool = True) -> GpuSnapshot:
    return {
        "index": index,
        "used_mb": 0,
        "total_mb": 100,
        "util": 0,
        "free": free,
        "processes": [],
    }


class FifoSchedulerPolicyTest(unittest.TestCase):
    def test_plan_respects_min_free_and_excluded_gpus(self):
        queue: QueueState = {
            "staging": [],
            "pending": [
                {"id": "p1", "gpus": 1},
                {"id": "p2", "gpus": 1},
                {"id": "p3", "gpus": 1},
            ],
            "running": [],
            "completed": [],
        }

        plans = FifoSchedulerPolicy().plan(
            queue,
            [_gpu(0), _gpu(1), _gpu(2), _gpu(3)],
            SchedulerConfig(min_free=1, excluded_gpus=frozenset({0})),
        )

        self.assertEqual(
            [(plan.job_id, plan.gpu_indices) for plan in plans],
            [("p1", [1]), ("p2", [2]), ("p3", [3])],
        )

    def test_plan_counts_existing_running_jobs_against_max_use(self):
        queue: QueueState = {
            "staging": [],
            "pending": [{"id": "p1", "gpus": 1}, {"id": "p2", "gpus": 1}],
            "running": [{"id": "r1", "gpus": 1, "assigned_gpus": [0]}],
            "completed": [],
        }

        plans = FifoSchedulerPolicy().plan(
            queue,
            [_gpu(0, free=False), _gpu(1), _gpu(2)],
            SchedulerConfig(min_free=0, max_use=2),
        )

        self.assertEqual(
            [(plan.job_id, plan.gpu_indices) for plan in plans],
            [("p1", [1])],
        )

    def test_plan_skips_too_large_jobs_but_keeps_fifo_order_for_started_jobs(self):
        queue: QueueState = {
            "staging": [],
            "pending": [
                {"id": "big", "gpus": 4},
                {"id": "small", "gpus": 1},
            ],
            "running": [],
            "completed": [],
        }

        plans = FifoSchedulerPolicy().plan(
            queue,
            [_gpu(0), _gpu(1)],
            SchedulerConfig(min_free=0),
        )

        self.assertEqual(
            [(plan.job_id, plan.gpu_indices) for plan in plans],
            [("small", [0])],
        )
