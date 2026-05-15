from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol

from gpu_queue.domain import GpuSnapshot, Job, JobStartPlan, QueueState, SchedulerConfig


class QueueStore(Protocol):
    def load(self) -> QueueState: ...

    def save(self, queue: QueueState) -> None: ...

    def transaction(self) -> AbstractContextManager[QueueState]: ...


class GpuProvider(Protocol):
    def snapshot(self) -> list[GpuSnapshot]: ...


class JobRunner(Protocol):
    def start(self, job: Job, gpu_indices: list[int]) -> int: ...


class SchedulerPolicy(Protocol):
    def startable_jobs(
        self, queue: QueueState, gpus: list[GpuSnapshot], config: SchedulerConfig
    ) -> list[JobStartPlan]: ...
