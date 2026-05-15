from __future__ import annotations

import json
import os
import signal
import sys
from datetime import datetime

from gpu_queue import paths
from gpu_queue.ids import generate_job_id
from gpu_queue.logs import log_msg
from gpu_queue.scheduler import daemon_loop, is_daemon_running
from gpu_queue.service import QueueService
from gpu_queue.storage import ensure_dirs, locked_queue


def cmd_serve(args) -> None:
    """Run the scheduler loop in the foreground."""
    ensure_dirs()
    max_use = getattr(args, "max_use", None)
    print(f"✓ Scheduler started (keeping {args.min_free} GPUs physically idle)")
    if max_use is not None:
        print(f"  Max gpu-queue use: {max_use} GPUs")
    print(f"  Polling every {paths.POLL_INTERVAL}s")

    excluded = set()
    if getattr(args, "exclude_gpus", None):
        try:
            for p in args.exclude_gpus.split(","):
                if p.strip():
                    excluded.add(int(p.strip()))
        except ValueError:
            print("Error: Invalid format for --exclude-gpus.")
            sys.exit(1)

    daemon_loop(args.min_free, excluded, max_use=max_use)


def cmd_add(args) -> None:
    """Add a job to the queue."""
    job = QueueService().add_job(
        command=args.command,
        gpus=args.gpus,
        priority=args.priority,
        front=args.front,
    )

    print(f"✓ Added job {job['id']} (requires {args.gpus} GPUs)")
    print(f"  Command: {args.command}")


def cmd_start(args) -> None:
    """Start the daemon."""
    ensure_dirs()
    max_use = getattr(args, "max_use", None)

    if is_daemon_running():
        print("Daemon is already running!")
        return

    pid = os.fork()
    if pid > 0:
        print(f"✓ Daemon started (PID: {pid})")
        print(f"  Polling every {paths.POLL_INTERVAL}s for free GPUs")
        print(f"  Log: {paths.DAEMON_LOG}")
        return

    os.setsid()
    os.chdir("/")

    paths.PID_FILE.write_text(str(os.getpid()))
    (paths.QUEUE_DIR / "config.json").write_text(
        json.dumps({"min_free_gpus": args.min_free, "max_use_gpus": max_use})
    )

    sys.stdout = open(paths.DAEMON_LOG, "a")
    sys.stderr = sys.stdout

    def handle_signal(signum, frame):
        log_msg("Daemon stopped")
        paths.PID_FILE.unlink(missing_ok=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    daemon_loop(args.min_free, max_use=max_use)


def cmd_stop(args) -> None:
    """Stop the daemon."""
    if not is_daemon_running():
        print("Daemon is not running.")
        return

    pid = int(paths.PID_FILE.read_text().strip())
    os.kill(pid, signal.SIGTERM)
    print(f"✓ Stopped daemon (PID: {pid})")


def cmd_cancel(args) -> None:
    """Cancel a pending job."""
    with locked_queue() as queue:
        for i, job in enumerate(queue["pending"]):
            if job["id"] == args.job_id:
                queue["pending"].pop(i)
                print(f"✓ Cancelled pending job {args.job_id}")
                return

        for i, job in enumerate(queue["running"]):
            if job["id"] == args.job_id:
                pid = job.get("pid")
                if pid:
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except Exception:
                        pass
                queue["running"].pop(i)
                job["status"] = "cancelled"
                job["ended"] = datetime.now().isoformat()
                queue["completed"].append(job)
                print(f"✓ Cancelled running job {args.job_id}")
                return

    print(f"Job {args.job_id} not found")


def cmd_logs(args) -> None:
    """Show logs for a job."""
    log_file = paths.LOG_DIR / f"{args.job_id}.log"
    if not log_file.exists():
        print(f"No log file for job {args.job_id}")
        return

    lines = args.lines
    with open(log_file) as f:
        content = f.readlines()
        for line in content[-lines:]:
            print(line, end="")


def cmd_clear(args) -> None:
    """Clear completed jobs from the queue."""
    with locked_queue() as queue:
        count = len(queue["completed"])
        queue["completed"] = []
    print(f"✓ Cleared {count} completed jobs")


def cmd_delete(args) -> None:
    """Delete a job from the completed list."""
    with locked_queue() as queue:
        for i, job in enumerate(queue["completed"]):
            if job["id"] == args.job_id:
                queue["completed"].pop(i)
                print(f"✓ Deleted job {args.job_id}")
                return
    print(f"Job {args.job_id} not found in completed jobs")


def cmd_retry(args) -> None:
    """Re-queue a completed job."""
    new_job = QueueService().requeue_completed(args.job_id, front=args.front)
    if new_job is not None:
        where = "front" if args.front else "back"
        print(f"✓ Re-queued job {new_job['id']} ({where})")
        return

    print(f"Job {args.job_id} not found in completed jobs")


def cmd_pause(args) -> None:
    """Pause a running job (kill and re-queue at front)."""
    with locked_queue() as queue:
        for i, job in enumerate(queue["running"]):
            if job["id"] == args.job_id:
                pid = job.get("pid")
                if pid:
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except Exception:
                        pass

                queue["running"].pop(i)

                new_job = {
                    "id": generate_job_id(),
                    "cmd": job["cmd"],
                    "gpus": job.get("gpus", 1),
                    "added": datetime.now().isoformat(),
                    "priority": 3,
                    "paused_from": job["id"],
                }

                queue["pending"].insert(0, new_job)
                print(f"✓ Paused job {job['id']} (Killed process group {pid})")
                print(f"✓ Re-queued as {new_job['id']} at front")
                return

    print(f"Job {args.job_id} not found in running jobs")
