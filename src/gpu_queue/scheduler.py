from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from gpu_queue import paths
from gpu_queue.gpu import get_free_gpus
from gpu_queue.logs import log_msg
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


def run_job(job: dict, gpu_indices: list[int]) -> int:
    """Run a job with the specified GPUs. Returns the PID."""
    log_file = paths.LOG_DIR / f"{job['id']}.log"
    exit_file = paths.QUEUE_DIR / f"{job['id']}.exit"
    gpu_str = ",".join(map(str, gpu_indices))

    cmd = " ".join(job["cmd"].split())

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_str
    local_bin = str(Path.home() / ".local" / "bin")
    if local_bin not in env.get("PATH", ""):
        env["PATH"] = f"{local_bin}:{env.get('PATH', '')}"

    with open(log_file, "w") as f:
        f.write(f"=== Job {job['id']} ===\n")
        f.write(f"Command: {cmd}\n")
        f.write(f"GPUs: {gpu_str}\n")
        f.write(f"Started: {datetime.now().isoformat()}\n")
        f.write("=" * 40 + "\n\n")

    q_log = f"'{log_file}'"
    q_exit = f"'{exit_file}'"
    wrapped_cmd = f"({cmd}) >> {q_log} 2>&1; echo $? > {q_exit}"

    proc = subprocess.Popen(
        wrapped_cmd,
        shell=True,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=Path.home() / "pepo",
        start_new_session=True,
    )

    return proc.pid


def _running_gpu_usage(job: dict) -> int:
    assigned = job.get("assigned_gpus", [])
    if assigned:
        return len(assigned)
    return int(job.get("gpus", 1))


def daemon_loop(
    min_free: int, excluded_gpus: set[int] | None = None, max_use: Optional[int] = None
) -> None:
    """Main scheduler loop."""
    if excluded_gpus is None:
        excluded_gpus = set()

    while True:
        try:
            cleanup_dead_jobs()

            with locked_queue() as queue:
                if queue["pending"]:
                    all_gpus = get_free_gpus()
                    gpus = [g for g in all_gpus if g["index"] not in excluded_gpus]

                    our_assigned = set()
                    our_usage = 0
                    for j in queue["running"]:
                        our_usage += _running_gpu_usage(j)
                        for idx in j.get("assigned_gpus", []):
                            our_assigned.add(idx)

                    actual_free_count = sum(
                        1
                        for g in all_gpus
                        if g["free"] and g["index"] not in our_assigned
                    )
                    startable_gpus = max(0, actual_free_count - min_free)
                    if max_use is not None:
                        startable_gpus = min(
                            startable_gpus, max(0, max_use - our_usage)
                        )

                    free_indices = [
                        g["index"]
                        for g in gpus
                        if g["free"] and g["index"] not in our_assigned
                    ]

                    if startable_gpus > 0:
                        jobs_started = False
                        remaining_pending = []

                        for job in queue["pending"]:
                            req = job.get("gpus", 1)

                            if req <= startable_gpus and req <= len(free_indices):
                                assigned = free_indices[:req]
                                log_msg(f"Starting job {job['id']} on GPUs {assigned}")

                                pid = run_job(job, assigned)

                                job["pid"] = pid
                                job["assigned_gpus"] = assigned
                                job["status"] = "running"
                                job["started"] = datetime.now().isoformat()
                                queue["running"].append(job)

                                startable_gpus -= req
                                free_indices = free_indices[req:]
                                jobs_started = True
                            else:
                                remaining_pending.append(job)

                        if jobs_started:
                            queue["pending"] = remaining_pending

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
    all_gpus: list[dict], min_free: int, max_use: Optional[int], excluded_gpus: set[int]
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
