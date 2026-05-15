from __future__ import annotations

from gpu_queue.domain import GpuSnapshot, Job, JobStartPlan, QueueState, SchedulerConfig


def _running_gpu_usage(job: Job) -> int:
    assigned = job.get("assigned_gpus", [])
    if assigned:
        return len(assigned)
    return int(job.get("gpus", 1))


class FifoSchedulerPolicy:
    def plan(
        self, queue: QueueState, gpus: list[GpuSnapshot], config: SchedulerConfig
    ) -> list[JobStartPlan]:
        our_assigned = {
            idx for job in queue["running"] for idx in job.get("assigned_gpus", [])
        }
        our_usage = sum(_running_gpu_usage(job) for job in queue["running"])

        actual_free_count = sum(
            1 for gpu in gpus if gpu["free"] and gpu["index"] not in our_assigned
        )
        startable_gpus = max(0, actual_free_count - config.min_free)
        if config.max_use is not None:
            startable_gpus = min(startable_gpus, max(0, config.max_use - our_usage))

        free_indices = [
            gpu["index"]
            for gpu in gpus
            if gpu["free"]
            and gpu["index"] not in our_assigned
            and gpu["index"] not in config.excluded_gpus
        ]

        plans: list[JobStartPlan] = []
        for job in queue["pending"]:
            req = int(job.get("gpus", 1))
            if req <= startable_gpus and req <= len(free_indices):
                assigned = free_indices[:req]
                plans.append(JobStartPlan(str(job["id"]), assigned))
                startable_gpus -= req
                free_indices = free_indices[req:]

        return plans
