from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol

from gpu_queue.domain import GpuSnapshot, QueueState, SchedulerConfig


class QueueStore(Protocol):
    def load(self) -> QueueState: ...

    def save(self, queue: QueueState) -> None: ...

    def transaction(self) -> AbstractContextManager[QueueState]: ...


class GpuProvider(Protocol):
    def snapshot(self) -> list[GpuSnapshot]: ...


class SchedulerPolicy(Protocol):
    def startable_jobs(
        self, queue: QueueState, gpus: list[GpuSnapshot], config: SchedulerConfig
    ) -> list[tuple[str, list[int]]]: ...
