from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from gpu_queue import paths
from gpu_queue.domain import GpuSnapshot, Job, SchedulerConfig
from gpu_queue.gpu import get_free_gpus
from gpu_queue.logs import log_msg
from gpu_queue.policies import FifoSchedulerPolicy
from gpu_queue.runner import SubprocessJobRunner
from gpu_queue.storage import locked_queue


def is_daemon_running() -> bool:
    """Check if scheduler is running (placeholder)."""
    return True


def cleanup_dead_jobs() -> None:
    """Check running jobs and move dead ones to completed with status classification."""
    with locked_queue() as queue:
        still_running = []
        changed = False

        for job in queue["running"]:
            pid = job.get("pid")
            if pid:
                proc_path = Path(f"/proc/{pid}")
                if proc_path.exists():
                    still_running.append(job)
                    continue

                job["ended"] = datetime.now().isoformat()
                exit_file = paths.QUEUE_DIR / f"{job['id']}.exit"

                status = "unknown"
                for _ in range(10):
                    if exit_file.exists():
                        try:
                            code = int(exit_file.read_text().strip())
                            status = "success" if code == 0 else "failed"
                            break
                        except Exception:
                            pass
                    time.sleep(0.1)

                if status == "unknown":
                    status = "killed"

                job["status"] = status
                queue["completed"].append(job)
                if exit_file.exists():
                    exit_file.unlink(missing_ok=True)
                changed = True
            else:
                still_running.append(job)

        if changed:
            queue["running"] = still_running


def run_job(job: Job, gpu_indices: list[int]) -> int:
    """Run a job with the specified GPUs. Returns the PID."""
    return SubprocessJobRunner().start(job, gpu_indices)


def daemon_loop(
    min_free: int, excluded_gpus: set[int] | None = None, max_use: Optional[int] = None
) -> None:
    """Main scheduler loop."""
    if excluded_gpus is None:
        excluded_gpus = set()
    policy = FifoSchedulerPolicy()

    while True:
        try:
            cleanup_dead_jobs()

            with locked_queue() as queue:
                if queue["pending"]:
                    all_gpus = get_free_gpus()
                    config = SchedulerConfig(
                        min_free=min_free,
                        max_use=max_use,
                        excluded_gpus=frozenset(excluded_gpus),
                    )
                    plans = policy.plan(queue, all_gpus, config)
                    if plans:
                        pending_by_id = {
                            str(job.get("id")): job for job in queue["pending"]
                        }
                        started_ids = set()
                        for plan in plans:
                            job = pending_by_id.get(plan.job_id)
                            if job is None:
                                continue
                            log_msg(
                                f"Starting job {job['id']} on GPUs {plan.gpu_indices}"
                            )

                            pid = run_job(job, plan.gpu_indices)

                            job["pid"] = pid
                            job["assigned_gpus"] = plan.gpu_indices
                            job["status"] = "running"
                            job["started"] = datetime.now().isoformat()
                            queue["running"].append(job)
                            started_ids.add(plan.job_id)

                        if started_ids:
                            queue["pending"] = [
                                job
                                for job in queue["pending"]
                                if str(job.get("id")) not in started_ids
                            ]

                    _write_status(all_gpus, min_free, max_use, excluded_gpus)
                else:
                    all_gpus = get_free_gpus()
                    _write_status(all_gpus, min_free, max_use, excluded_gpus)

            time.sleep(paths.POLL_INTERVAL)

        except KeyboardInterrupt:
            break
        except Exception as e:
            log_msg(f"Error in daemon loop: {e}")
            time.sleep(paths.POLL_INTERVAL)


def _write_status(
    all_gpus: list[GpuSnapshot],
    min_free: int,
    max_use: Optional[int],
    excluded_gpus: set[int],
) -> None:
    try:
        status_data = {
            "ts": datetime.now().isoformat(),
            "gpus": all_gpus,
            "min_free": min_free,
            "max_use": max_use,
            "excluded": list(excluded_gpus),
        }
        (paths.QUEUE_DIR / "status.json").write_text(json.dumps(status_data))
    except Exception:
        pass
