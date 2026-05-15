from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

QueueName = Literal["staging", "pending", "running", "completed"]
JobStatus = Literal["running", "success", "failed", "cancelled", "killed", "unknown"]

Job = dict[str, Any]
QueueState = dict[str, list[Job]]


class GpuProcess(TypedDict, total=False):
    pid: str
    user: str
    name: str
    mem_mb: int
    zombie: bool


class GpuSnapshot(TypedDict):
    index: int
    used_mb: int
    total_mb: int
    util: int
    free: bool
    processes: list[GpuProcess]


@dataclass(frozen=True)
class SchedulerConfig:
    min_free: int = 2
    max_use: int | None = None
    excluded_gpus: frozenset[int] = frozenset()


@dataclass(frozen=True)
class JobStartPlan:
    job_id: str
    gpu_indices: list[int]
