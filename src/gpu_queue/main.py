#!/usr/bin/env python3
"""
GPU Job Queue Scheduler

A lightweight job queue for shared GPU servers without SLURM.
Automatically runs jobs when the required number of GPUs become available.

Usage:
Usage:
    gpu-queue add --gpus 2 "uv run scripts/train.py model=deppo L=1"
    gpu-queue start
    gpu-queue status
    gpu-queue cancel <job_id>
    gpu-queue stop
    gpu-queue logs <job_id>
"""

import argparse
import copy
import curses
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

# Try importing requests. Env should have it. Fallback if needed.
try:
    import requests
except ImportError:
    requests = None

# Configuration
QUEUE_DIR = Path.home() / ".gpu_queue"
QUEUE_FILE = QUEUE_DIR / "jobs.json"
PID_FILE = QUEUE_DIR / "daemon.pid"
DAEMON_LOG = QUEUE_DIR / "daemon.log"
LOG_DIR = QUEUE_DIR / "logs"

POLL_INTERVAL = 2  # seconds between GPU checks
MIN_FREE_GPUS = 2  # Number of GPUs to always keep free for other users
SERVER_PORT = 12345


def get_server_url():
    return f"http://localhost:{SERVER_PORT}"


def ensure_dirs():
    """Create queue directories if they don't exist."""
    QUEUE_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)


LOCK_FILE = QUEUE_DIR / "queue.lock"


@contextmanager
def locked_queue():
    """Context manager for thread-safe and process-safe queue access."""
    ensure_dirs()
    with open(LOCK_FILE, "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            queue = load_queue_raw()
            yield queue
            save_queue_raw(queue)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def load_queue_raw() -> dict[str, list]:
    """Load the job queue from disk without locking."""
    if not QUEUE_FILE.exists():
        return {"pending": [], "running": [], "completed": []}

    try:
        with open(QUEUE_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        return {"pending": [], "running": [], "completed": []}


def sort_pending_queue(queue: dict[str, list]):
    """Sort pending queue by priority (higher first), then added time (older first)."""
    # Priority mapping: 0=low, 1=medium, 2=high
    queue["pending"].sort(key=lambda x: (-x.get("priority", 1), x["added"]))


def save_queue_raw(queue: dict[str, list]):
    """Save the job queue to disk without locking (atomic replace)."""
    ensure_dirs()
    sort_pending_queue(queue)
    temp_file = QUEUE_FILE.with_suffix(".tmp")
    with open(temp_file, "w") as f:
        json.dump(queue, f, indent=2)
    os.replace(temp_file, QUEUE_FILE)


def load_queue() -> dict[str, list]:
    """Load the job queue (backward compatibility, no lock)."""
    return load_queue_raw()


def save_queue(queue: dict[str, list]):
    """Save the job queue (backward compatibility, no lock)."""
    save_queue_raw(queue)


def generate_job_id() -> str:
    """Generate a short unique job ID."""
    import hashlib

    ts = datetime.now().isoformat()
    return hashlib.md5(ts.encode()).hexdigest()[:8]


def get_free_gpus() -> list[dict[str, Any]]:
    """Get list of GPUs with their status (free = no processes running)."""
    try:
        # First get basic GPU info
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        gpus = {}
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                idx = int(parts[0])
                used = int(parts[1])
                total = int(parts[2])
                util = 0
                if len(parts) >= 4 and parts[3].strip().isdigit():
                    util = int(parts[3].strip())

                gpus[idx] = {
                    "index": idx,
                    "used_mb": used,
                    "total_mb": total,
                    "util": util,
                    "free": True,  # Assume free, will mark busy if processes found
                    "processes": [],  # List of process info dicts
                }

        # Get GPU index to UUID mapping
        uuid_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        uuid_to_idx = {}
        for line in uuid_result.stdout.strip().split("\n"):
            if "," in line:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    uuid_to_idx[parts[1]] = int(parts[0])

        # Get process details
        proc_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        # Mark GPUs with processes as busy and collect process info
        for line in proc_result.stdout.strip().split("\n"):
            if "," in line:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    gpu_uuid, pid_str, proc_name, mem_str = (
                        parts[0],
                        parts[1],
                        parts[2],
                        parts[3],
                    )
                    if gpu_uuid in uuid_to_idx:
                        idx = uuid_to_idx[gpu_uuid]
                        if idx in gpus:
                            # Check if it's a zombie (process doesn't exist)
                            is_zombie = proc_name == "[Not Found]"
                            if not is_zombie:
                                # Verify the PID actually exists using /proc
                                try:
                                    pid = int(pid_str)
                                    if not Path(f"/proc/{pid}").exists():
                                        is_zombie = True
                                except ValueError:
                                    is_zombie = True

                            # Extract user from process path
                            user = "unknown"
                            if "/home/" in proc_name:
                                user = proc_name.split("/home/")[1].split("/")[0]
                            elif is_zombie:
                                user = "zombie"

                            gpus[idx]["processes"].append(
                                {
                                    "pid": pid_str,
                                    "user": user,
                                    "name": proc_name.split("/")[-1]
                                    if "/" in proc_name
                                    else proc_name,
                                    "mem_mb": int(mem_str) if mem_str.isdigit() else 0,
                                    "zombie": is_zombie,
                                }
                            )

                            # Only mark as busy if NOT a zombie
                            if not is_zombie:
                                gpus[idx]["free"] = False

        return list(gpus.values())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def get_available_gpu_indices() -> list[int]:
    """Get indices of free GPUs (excluding running jobs)."""
    gpus = get_free_gpus()
    free_indices = [g["index"] for g in gpus if g["free"]]

    # Also exclude GPUs already assigned to running jobs (race condition protection)
    queue = load_queue()
    reserved_gpus = set()
    for job in queue.get("running", []):
        for gpu_idx in job.get("assigned_gpus", []):
            reserved_gpus.add(gpu_idx)

    return [idx for idx in free_indices if idx not in reserved_gpus]


def is_daemon_running() -> bool:
    """Check if scheduler is running (placeholder)."""
    # For now, just return True to avoid warnings, or implement a real check.
    return True


def cleanup_dead_jobs():
    """Check running jobs and move dead ones to completed with status classification."""
    with locked_queue() as queue:
        still_running = []
        changed = False

        for job in queue["running"]:
            pid = job.get("pid")
            if pid:
                # Check if process is still running
                proc_path = Path(f"/proc/{pid}")
                if proc_path.exists():
                    still_running.append(job)
                    continue

                # Process is dead. Wait for exit file.
                job["ended"] = datetime.now().isoformat()
                exit_file = QUEUE_DIR / f"{job['id']}.exit"

                # Wait up to 1s for exit file to appear (shell might be finishing up)
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
            # save_queue_raw is called by locked_queue context manager
        else:
            still_running.append(job)  # ensure we don't lose them if not changed
            # But actually still_running is what we want.
            # If not changed, queue["running"] remains same.
            pass


def run_job(job: dict, gpu_indices: list[int]) -> int:
    """Run a job with the specified GPUs. Returns the PID."""
    log_file = LOG_DIR / f"{job['id']}.log"
    exit_file = QUEUE_DIR / f"{job['id']}.exit"
    gpu_str = ",".join(map(str, gpu_indices))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_str
    # Ensure ~/.local/bin is in PATH for uv
    local_bin = str(Path.home() / ".local" / "bin")
    if local_bin not in env.get("PATH", ""):
        env["PATH"] = f"{local_bin}:{env.get('PATH', '')}"

    with open(log_file, "w") as f:
        f.write(f"=== Job {job['id']} ===\n")
        f.write(f"Command: {job['cmd']}\n")
        f.write(f"GPUs: {gpu_str}\n")
        f.write(f"Started: {datetime.now().isoformat()}\n")
        f.write("=" * 40 + "\n\n")

    # Wrap in shell to capture exit code. Quote paths to be safe.
    q_log = f"'{log_file}'"
    q_exit = f"'{exit_file}'"
    wrapped_cmd = f"({job['cmd']}) >> {q_log} 2>&1; echo $? > {q_exit}"

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


def daemon_loop(min_free, excluded_gpus=None):
    """Main scheduler loop."""
    if excluded_gpus is None:
        excluded_gpus = set()

    while True:
        try:
            cleanup_dead_jobs()

            with locked_queue() as queue:
                if queue["pending"]:
                    # --- Quota and Availability Logic ---
                    all_gpus = get_free_gpus()

                    # Filter out excluded GPUs
                    gpus = [g for g in all_gpus if g["index"] not in excluded_gpus]

                    total_gpus = len(gpus)

                    # Excluded GPUs count towards the reserved quota
                    effective_min_free = max(0, min_free - len(excluded_gpus))

                    quota = total_gpus - effective_min_free

                    # Count GPUs currently assigned to our running jobs
                    our_usage = sum(job.get("gpus", 1) for job in queue["running"])
                    quota_remaining = quota - our_usage

                    # List of GPUs that are ACTUALLY free
                    our_assigned = set()
                    for j in queue["running"]:
                        for idx in j.get("assigned_gpus", []):
                            our_assigned.add(idx)

                    free_indices = [
                        g["index"]
                        for g in gpus
                        if g["free"] and g["index"] not in our_assigned
                    ]

                    if quota_remaining > 0:
                        # --- Backfilling Scheduler ---
                        jobs_started = False
                        remaining_pending = []

                        for job in queue["pending"]:
                            req = job.get("gpus", 1)

                            if req <= quota_remaining and req <= len(free_indices):
                                assigned = free_indices[:req]
                                log_msg(f"Starting job {job['id']} on GPUs {assigned}")

                                pid = run_job(job, assigned)

                                job["pid"] = pid
                                job["assigned_gpus"] = assigned
                                job["status"] = "running"
                                job["started"] = datetime.now().isoformat()
                                queue["running"].append(job)

                                quota_remaining -= req
                                free_indices = free_indices[req:]
                                jobs_started = True
                            else:
                                remaining_pending.append(job)

                        if jobs_started:
                            queue["pending"] = remaining_pending

                    # Save GPU status for TUI
                    try:
                        status_data = {
                            "ts": datetime.now().isoformat(),
                            "gpus": all_gpus,
                            "min_free": min_free,
                            "excluded": list(excluded_gpus),
                        }
                        (QUEUE_DIR / "status.json").write_text(json.dumps(status_data))
                    except Exception:
                        pass
                else:
                    try:
                        all_gpus = get_free_gpus()
                        status_data = {
                            "ts": datetime.now().isoformat(),
                            "gpus": all_gpus,
                            "min_free": min_free,
                            "excluded": list(excluded_gpus),
                        }
                        (QUEUE_DIR / "status.json").write_text(json.dumps(status_data))
                    except Exception:
                        pass

            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            break
        except Exception as e:
            log_msg(f"Error in daemon loop: {e}")
            time.sleep(POLL_INTERVAL)


def cmd_serve(args):
    """Run the scheduler loop in the foreground."""
    ensure_dirs()
    print(f"✓ Scheduler started (keeping {args.min_free} GPUs reserved)")
    print(f"  Polling every {POLL_INTERVAL}s")

    # Parse excluded GPUs
    excluded = set()
    if getattr(args, "exclude_gpus", None):
        try:
            for p in args.exclude_gpus.split(","):
                if p.strip():
                    excluded.add(int(p.strip()))
        except ValueError:
            print("Error: Invalid format for --exclude-gpus.")
            sys.exit(1)

    daemon_loop(args.min_free, excluded)


def log_msg(msg: str, verbose: bool = False):
    """Log a message to the daemon log."""
    if verbose:
        return  # Skip verbose messages for now
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    with open(DAEMON_LOG, "a") as f:
        f.write(line)


# === CLI Commands ===


def cmd_add(args):
    """Add a job to the queue."""
    priorities = {"low": 0, "medium": 1, "high": 2}
    prio = priorities.get(args.priority, 1)
    if args.front:
        prio = 3

    job = {
        "id": generate_job_id(),
        "cmd": args.command,
        "gpus": args.gpus,
        "added": datetime.now().isoformat(),
        "priority": prio,
    }

    with locked_queue() as queue:
        queue["pending"].append(job)

    print(f"✓ Added job {job['id']} (requires {args.gpus} GPUs)")
    print(f"  Command: {args.command}")


def cmd_start(args):
    """Start the daemon."""
    ensure_dirs()

    if is_daemon_running():
        print("Daemon is already running!")
        return

    # Fork to background
    pid = os.fork()
    if pid > 0:
        # Parent process
        print(f"✓ Daemon started (PID: {pid})")
        print(f"  Polling every {POLL_INTERVAL}s for free GPUs")
        print(f"  Log: {DAEMON_LOG}")
        return

    # Child process - become daemon
    os.setsid()
    os.chdir("/")

    # Write PID file with configuration
    PID_FILE.write_text(str(os.getpid()))

    # Store config in separate file or just run with args
    (QUEUE_DIR / "config.json").write_text(json.dumps({"min_free_gpus": args.min_free}))

    # Redirect stdout/stderr
    sys.stdout = open(DAEMON_LOG, "a")
    sys.stderr = sys.stdout

    # Handle termination
    def handle_signal(signum, frame):
        log_msg("Daemon stopped")
        PID_FILE.unlink(missing_ok=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    daemon_loop(args.min_free)


def cmd_stop(args):
    """Stop the daemon."""
    if not is_daemon_running():
        print("Daemon is not running.")
        return

    pid = int(PID_FILE.read_text().strip())
    os.kill(pid, signal.SIGTERM)
    print(f"✓ Stopped daemon (PID: {pid})")


def get_terminal_width() -> int:
    """Get the current terminal width."""
    return shutil.get_terminal_size((80, 20)).columns


def shorten_command(cmd: str, max_len: int) -> str:
    """Shorten a command string to max_len by removing the middle part."""
    if len(cmd) <= max_len:
        return cmd

    # Calculate lengths to keep
    head_len = (max_len - 3) // 2
    tail_len = max_len - 3 - head_len

    return f"{cmd[:head_len]}...{cmd[-tail_len:]}"


# cmd_status removed - functionality merged into watch TUI


def cmd_cancel(args):
    """Cancel a pending job."""
    with locked_queue() as queue:
        for i, job in enumerate(queue["pending"]):
            if job["id"] == args.job_id:
                queue["pending"].pop(i)
                print(f"✓ Cancelled pending job {args.job_id}")
                return

        # Also check running
        for i, job in enumerate(queue["running"]):
            if job["id"] == args.job_id:
                pid = job.get("pid")
                if pid:
                    try:
                        os.killpg(pid, signal.SIGKILL)  # Aggressive kill
                    except Exception:
                        pass
                queue["running"].pop(i)
                job["status"] = "cancelled"
                job["ended"] = datetime.now().isoformat()
                queue["completed"].append(job)
                print(f"✓ Cancelled running job {args.job_id}")
                return

    print(f"Job {args.job_id} not found")


def cmd_logs(args):
    """Show logs for a job."""
    log_file = LOG_DIR / f"{args.job_id}.log"
    if not log_file.exists():
        print(f"No log file for job {args.job_id}")
        return

    # Tail the log
    lines = args.lines
    with open(log_file) as f:
        content = f.readlines()
        for line in content[-lines:]:
            print(line, end="")


def cmd_clear(args):
    """Clear completed jobs from the queue."""
    with locked_queue() as queue:
        count = len(queue["completed"])
        queue["completed"] = []
    print(f"✓ Cleared {count} completed jobs")


def cmd_delete(args):
    """Delete a job from the completed list."""
    with locked_queue() as queue:
        for i, job in enumerate(queue["completed"]):
            if job["id"] == args.job_id:
                queue["completed"].pop(i)
                print(f"✓ Deleted job {args.job_id}")
                return
    print(f"Job {args.job_id} not found in completed jobs")


def cmd_retry(args):
    """Re-queue a completed job."""
    with locked_queue() as queue:
        # Find job in completed
        for i, job in enumerate(queue["completed"]):
            if job["id"] == args.job_id:
                # Remove from completed
                queue["completed"].pop(i)

                # Reset job metadata but KEEP THE ID if requested or default behavior?
                # User asked to not duplicate. Reusing ID is best.
                # But we should update 'added' time? Or keep original?
                # Let's update added time to reflect it's back in queue.

                new_job = {
                    "id": job["id"],  # Reuse ID
                    "cmd": job["cmd"],
                    "gpus": job.get("gpus", 1),
                    "added": datetime.now().isoformat(),
                    "retried_at": datetime.now().isoformat(),
                    "priority": 1,
                }

                # Add to front or back of pending
                if args.front:
                    queue["pending"].insert(0, new_job)
                    print(f"✓ Re-queued job {job['id']} (front)")
                else:
                    queue["pending"].append(new_job)
                    print(f"✓ Re-queued job {job['id']} (back)")
                return

    print(f"Job {args.job_id} not found in completed jobs")


def cmd_pause(args):
    """Pause a running job (kill and re-queue at front)."""
    with locked_queue() as queue:
        # Find job in running
        for i, job in enumerate(queue["running"]):
            if job["id"] == args.job_id:
                pid = job.get("pid")
                if pid:
                    try:
                        # Aggressive kill to free GPU instantly
                        os.killpg(pid, signal.SIGKILL)
                    except Exception:
                        pass

                # Remove from running
                queue["running"].pop(i)

                # Reset metadata for re-queue
                new_job = {
                    "id": generate_job_id(),
                    "cmd": job["cmd"],
                    "gpus": job.get("gpus", 1),
                    "added": datetime.now().isoformat(),
                    "priority": 3,  # Urgent
                    "paused_from": job["id"],
                }

                # Add to front of pending queue
                queue["pending"].insert(0, new_job)
                print(f"✓ Paused job {job['id']} (Killed process group {pid})")
                print(f"✓ Re-queued as {new_job['id']} at front")
                return

    print(f"Job {args.job_id} not found in running jobs")


def get_status_data():
    """Gather all status data for the queue and GPUs."""
    cleanup_dead_jobs()
    queue = load_queue()
    gpus = get_free_gpus()

    # Get config (min_free and excluded)
    min_free = 2
    excluded = set()
    config_file = QUEUE_DIR / "config.json"
    if config_file.exists():
        try:
            cfg = json.loads(config_file.read_text())
            min_free = cfg.get("min_free_gpus", 2)
            excluded = set(cfg.get("excluded_gpus", []))
        except Exception:
            pass

    # Filter GPUs (hiding excluded ones entirely from the monitor view)
    gpus = [g for g in gpus if g["index"] not in excluded]

    return {
        "queue": queue,
        "gpus": gpus,
        "min_free": min_free,
        "excluded": list(excluded),
        "term_width": get_terminal_width(),
    }


class Window:
    def __init__(self, title, key, height_pct=0.3):
        self.title = title
        self.key = key
        self.items = []
        self.selected_idx = 0
        self.scroll_offset = 0
        self.height_pct = height_pct  # Target height percentage
        self.collapsed = False

    def update_items(self, items):
        self.items = items
        # Clamp selection
        if self.selected_idx >= len(self.items):
            self.selected_idx = max(0, len(self.items) - 1)
        # Clamp scroll offset
        if self.scroll_offset >= len(self.items):
            self.scroll_offset = max(0, len(self.items) - 1)

    def scroll(self, delta, h=None):
        """Scroll selection by delta."""
        if not self.items:
            return

        # Use stored height if available (more accurate), otherwise fallback
        effective_h = getattr(self, "height", h)
        if effective_h is None:
            effective_h = 10  # Fallback

        new_idx = self.selected_idx + delta
        self.selected_idx = max(0, min(len(self.items) - 1, new_idx))

        # Adjust scroll offset to keep selected item in view
        # visible items = h - 2 (border)
        visible_h = max(1, effective_h - 2)

        # Account for header in list windows
        if self.key in ["running", "pending", "completed"]:
            visible_h = max(1, visible_h - 1)

        if self.selected_idx < self.scroll_offset:
            self.scroll_offset = self.selected_idx
        elif self.selected_idx >= self.scroll_offset + visible_h:
            self.scroll_offset = self.selected_idx - visible_h + 1

    def get_selected(self):
        if 0 <= self.selected_idx < len(self.items):
            return self.items[self.selected_idx]
        return None


class GPUQueueTUI:
    def __init__(self, interval=2.0):
        self.interval = interval
        self.stdscr = None
        self.running = False
        self.lock = threading.Lock()

        # State
        self.data = {"running": [], "pending": [], "completed": []}
        self.data = {"running": [], "pending": [], "completed": []}
        self.gpu_status = []
        self.min_free = 2
        self.excluded = []
        self.server_status = "UNKNOWN"
        self.last_updated = 0
        self.action_msg = ""
        self.msg_clear_time = 0
        self.modal = None  # { type, title, text, val... }

        # Windows
        self.windows = [
            Window("RUNNING", "running", 0.2),
            Window("PENDING", "pending", 0.2),
            Window("COMPLETED", "completed", 0.2),
            Window("GPU STATUS", "gpu_status", 0.2),
            Window("SELECTED JOB", "job_details", 0.2),
        ]
        self.active_win_idx = 0  # Index in self.windows
        self.mode = "NAV"  # "NAV" (select window) or "ACTION" (interact with window)

        # Log view state
        self.viewing_logs = False
        self.log_job_id = None
        self.log_content = []
        self.log_scroll = 0

        # Edit Mode State
        self.edit_mode_active = False
        self.edit_job = None
        self.edit_field_idx = 0  # 0: GPUs, 1: Priority, 2: Command
        self.edit_is_new = False

    def start(self):
        self.running = True
        # Fast loop for queue updates
        t1 = threading.Thread(target=self._poll_queue_loop, daemon=True)
        t1.start()
        # Slow loop for GPU polling
        t2 = threading.Thread(target=self._poll_gpu_loop, daemon=True)
        t2.start()

    def stop(self):
        self.running = False

    def _poll_queue_loop(self):
        """Poll queue frequently for snappy UI."""
        while self.running:
            try:
                with locked_queue() as q:
                    with self.lock:
                        self.data = copy.deepcopy(q)

                        # Sort completed by ended time (descending)
                        # ended field is isoformat string
                        self.data["completed"].sort(
                            key=lambda x: x.get("ended", ""), reverse=True
                        )

                        self.last_updated = time.time()

                # Update window items
                with self.lock:
                    for w in self.windows:
                        items = self.data.get(w.key, [])
                        type_label = w.key
                        for item in items:
                            item["_type"] = type_label
                        w.update_items(items)

                time.sleep(0.5)  # Fast update
            except Exception:
                # with self.lock: self.action_msg = f"Q Poll Error: {str(e)}"
                time.sleep(1)

    def _poll_gpu_loop(self):
        """Poll GPU status at slower interval."""
        while self.running:
            try:
                status_file = QUEUE_DIR / "status.json"
                if status_file.exists():
                    try:
                        # Use file lock or retry read for atomic?
                        # Using simple read should be fine mostly
                        txt = status_file.read_text()
                        if txt.strip():
                            s = json.loads(txt)
                            with self.lock:
                                self.gpu_status = s.get("gpus", [])
                                self.min_free = s.get("min_free", 2)
                                self.excluded = s.get("excluded", [])
                                self.server_status = "DAEMON: ON"
                        else:
                            with self.lock:
                                self.server_status = "DAEMON: S?"
                    except json.JSONDecodeError:
                        pass  # Partial write?
                else:
                    with self.lock:
                        self.server_status = "DAEMON: OFF"
            except Exception:
                with self.lock:
                    self.server_status = "ERR"

            time.sleep(self.interval)

    def draw_box(self, y, x, h, w, title, active=False, focused=False):
        """Draw a bordered box."""
        try:
            color = curses.color_pair(4)  # Cyan default
            if focused:
                color = curses.color_pair(2)  # Green for focused interaction
                # self.stdscr.attron(curses.A_BOLD)
            elif active:
                color = curses.color_pair(3)  # Yellow for selected window

            # Draw border
            self.stdscr.attron(color)
            self.stdscr.box()
            # rectangle doesn't use relative coords well with subwin?

            # Manual box drawing using line characters if needed, or just addstr
            # Top
            self.stdscr.hline(y, x, curses.ACS_HLINE, w)
            self.stdscr.vline(y, x, curses.ACS_VLINE, h)
            self.stdscr.hline(y + h - 1, x, curses.ACS_HLINE, w)
            self.stdscr.vline(y, x + w - 1, curses.ACS_VLINE, h)

            # Corners
            self.stdscr.addch(y, x, curses.ACS_ULCORNER)
            self.stdscr.addch(y, x + w - 1, curses.ACS_URCORNER)
            self.stdscr.addch(y + h - 1, x, curses.ACS_LLCORNER)
            self.stdscr.addch(y + h - 1, x + w - 1, curses.ACS_LRCORNER)

            # Title
            title_str = f" {title} "
            if focused:
                title_str = f" [ {title} ] "
            elif active:
                title_str = f" {title} "

            self.stdscr.addstr(
                y, x + 2, title_str, color | (curses.A_BOLD if active else 0)
            )
            self.stdscr.attroff(color)
        except Exception:
            pass

    def _parse_iso(self, s):
        if not s:
            return None
        try:
            return datetime.fromisoformat(s)
        except Exception:
            return None

    def _fmt_delta(self, delta):
        if not delta:
            return "-"
        s = int(delta.total_seconds())
        if s < 60:
            return f"{s}s"
        m = s // 60
        if m < 60:
            return f"{m}m"
        h = m // 60
        m = m % 60
        return f"{h}h{m}m"

    def format_job_line(self, job, w):
        """Format a job dictionary into a single line string."""
        jid = job["id"]

        # Edit Mode Highlighting
        edit_idx = job.get("_edit_field_idx", -1)
        is_editing = edit_idx >= 0

        # Determine Color based on state/status
        color = curses.color_pair(0)

        prefix = ""

        if job["_type"] == "running":
            # Formt: ID(8) | PID(6) | GPUS(8) | ELAPSED(9) | CMD
            color = curses.color_pair(5)  # Blue

            gpus = ",".join(map(str, job.get("assigned_gpus", [])))
            if not gpus:
                gpus = "?"

            pid = str(job.get("pid", "?"))

            elapsed = "-"
            start_dt = self._parse_iso(job.get("started"))
            if start_dt:
                elapsed = self._fmt_delta(datetime.now() - start_dt)

            prefix = f" {jid:<8} {pid:<6} {gpus:<8} {elapsed:<9} "

        elif job["_type"] == "pending":
            # Format: ID(8) | PRIO(6) | WAITING(9) | CMD
            # Pending items usually white.

            gpus = str(job.get("gpus", 1))
            p = job.get("priority", 1)
            p_s = ["L", "M", "H", "U"][min(p, 3)]

            waiting = "-"
            add_dt = self._parse_iso(job.get("added"))
            if add_dt:
                waiting = self._fmt_delta(datetime.now() - add_dt)

            # Construct fields individually for highlighting
            # available_width = max_width - fixed_columns
            # Highlighting enabled by wrapping active field

            # To ensure visibility if colors fail, let's add markers or brackets?
            # But brackets change length.
            # Maybe just keep exact length but use background color.
            # User said "definitely not blinking and don't know what field".
            # So let's use brackets OR arrows if length permits.
            # Changing length breaks column alignment unless we account for it.
            # Simple approach: Return exact same string length but rely heavily on REVERSE.

            # Wait, if we use segments, we can just rely on REVERSE.
            # If REVERSE isn't showing, maybe the terminal is weird.
            # Let's try adding explicit indicators.

            prefix = f" {jid:<8} {gpus:<6} {p_s:<8} {waiting:<9} "

        else:
            # Completed/Finished
            # Format: ID(8) | RUNTIME(9) | AGO(9) | CMD

            # Color by status
            s_res = job.get("status", "?")
            if s_res == "success":
                color = curses.color_pair(2)  # Green
            elif s_res == "failed":
                color = curses.color_pair(1)  # Red
            elif s_res == "cancelled":
                color = curses.color_pair(3)  # Orange

            start_dt = self._parse_iso(job.get("started"))
            end_dt = self._parse_iso(job.get("ended"))

            run_s = "-"
            ago_s = "-"

            if start_dt and end_dt:
                run_s = self._fmt_delta(end_dt - start_dt)

            if end_dt:
                ago_s = self._fmt_delta(datetime.now() - end_dt)

            prefix = f" {jid:<8} {run_s:<9} {ago_s:<9} "

        cmd = job.get("cmd", "")
        avail_cmd = w - len(prefix)
        if len(cmd) > avail_cmd:
            cmd = cmd[: (avail_cmd - 1)] + "…"

        full_line = prefix + cmd

        if is_editing and job["_type"] == "pending":
            # Use Brackets for visibility even if colors fail
            # We need to construct segments with markers

            # Field 0: GPUS. Width 6.
            s_gpus = f"{gpus:<6} "
            if edit_idx == 0:
                s_gpus = f"[{gpus}]".ljust(7)  # Makes it slightly wider? No good.
                # Keep width 7 (6+1 space)
                # {gpus} is e.g. "1" (len 1). "1     "
                # "[1]   "
                s_val = f"[{gpus}]"
                s_gpus = f"{s_val:<7}"
            else:
                s_gpus = f"{gpus:<6} "

            # Field 1: PRIO. Width 8.
            # {p_s} e.g. "L"
            if edit_idx == 1:
                s_val = f"[{p_s}]"
                s_prio = f"{s_val:<9}"
            else:
                s_prio = f"{p_s:<8} "

            # Field 2: CMD.
            # Just highlight it.

            return {
                "type": "rich",
                "segments": [
                    (f" {jid:<8} ", curses.A_NORMAL),
                    (s_gpus, curses.A_REVERSE if edit_idx == 0 else curses.A_NORMAL),
                    (s_prio, curses.A_REVERSE if edit_idx == 1 else curses.A_NORMAL),
                    (f"{waiting:<9} ", curses.A_NORMAL),
                    (cmd, curses.A_REVERSE if edit_idx == 2 else curses.A_NORMAL),
                ],
                "base_color": color | curses.A_BLINK,
            }, color

        return full_line, color

    def draw(self):
        self.stdscr.erase()
        h, w = self.stdscr.getmaxyx()

        if h < 20 or w < 60:
            self.stdscr.addstr(0, 0, "Terminal too small!")
            return

        with self.lock:
            # 0. Header (1 row)
            # 0. Header (1 row)
            # Full width white bar
            self.stdscr.hline(0, 0, " ", w, curses.color_pair(6) | curses.A_REVERSE)

            status_col = curses.color_pair(6) | curses.A_REVERSE  # White BG

            # Title
            self.stdscr.addstr(0, 1, " GPU QUEUE WATCH ", status_col | curses.A_BOLD)

            # Daemon Status
            server_col = (
                curses.color_pair(2 if "ON" in self.server_status else 1)
                | curses.A_REVERSE
            )
            self.stdscr.addstr(
                0, 20, f" [{self.server_status}] ", server_col | curses.A_BOLD
            )

            # Daemon Info (Reserved/Excluded)
            info_str = f"Res: {self.min_free}"
            if self.excluded:
                ex_list = ",".join(map(str, sorted(self.excluded)))
                info_str += f" | Excl: [{ex_list}]"

            # Right aligned info
            info_x = w - len(info_str) - 2
            if info_x > 40:  # Prevent overlap
                self.stdscr.addstr(0, info_x, info_str, status_col)

            # Action message overlay
            if self.action_msg:
                if time.time() > self.msg_clear_time:
                    self.action_msg = ""
                else:
                    msg_x = w // 2 - len(self.action_msg) // 2
                    self.stdscr.addstr(
                        0,
                        msg_x,
                        f" {self.action_msg} ",
                        curses.color_pair(3) | curses.A_REVERSE,
                    )

            # 2. Main Windows
            # Calculate heights
            avail_h = h - 2  # -1 for header, -1 for footer

            # --- Dynamic Sizing Logic ---
            # 1. GPU Status: Content + 2 (border)
            gpu_h = 0
            if not self.windows[3].collapsed:
                gpu_content_len = len(self.gpu_status) if self.gpu_status else 1
                gpu_h = min(
                    gpu_content_len + 3, avail_h // 3
                )  # +3 for Border(2) + Header(1)
                gpu_h = max(3, gpu_h)  # Min height
            else:
                gpu_h = 1

            # 2. Running: Content + 2 (border)
            running_h = 0
            if not self.windows[0].collapsed:
                running_items = len(self.data.get("running", []))
                running_content_len = running_items + 1  # +1 for header
                running_h = min(running_content_len + 3, avail_h // 3)
                running_h = max(3, running_h)
            else:
                running_h = 1

            # 3. Selected Job: Fixed small
            job_h = 0
            if not self.windows[4].collapsed:
                job_h = 8  # Fixed small size
            else:
                job_h = 1

            # 4. Pending & Completed share the rest
            used_h = gpu_h + running_h + job_h
            remaining_h = max(0, avail_h - used_h)

            pending_h = 0
            completed_h = 0

            if not self.windows[1].collapsed and not self.windows[2].collapsed:
                pending_h = remaining_h // 2
                completed_h = remaining_h - pending_h
            elif self.windows[1].collapsed and not self.windows[2].collapsed:
                pending_h = 1
                completed_h = remaining_h - 1
            elif not self.windows[1].collapsed and self.windows[2].collapsed:
                pending_h = remaining_h - 1
                completed_h = 1
            else:
                pending_h = 1
                completed_h = 1

            # Assign calculated heights
            heights = [running_h, pending_h, completed_h, gpu_h, job_h]

            current_y = 1
            for i, win in enumerate(self.windows):
                wh = heights[i]
                win.height = wh  # Store actual height for scrolling
                if wh <= 0:
                    continue  # Skip hidden windows

                active = i == self.active_win_idx
                focused = active and self.mode == "ACTION"

                self.draw_window(win, current_y, 0, wh, w, active, focused)
                current_y += wh

            # 3. Footer
            self.draw_footer(h - 1, w)

            # 4. Log Overlay?
            if self.viewing_logs:
                self.draw_log_overlay(h, w)

            # 5. Modal
            if self.modal:
                self.draw_modal(h, w)

        self.stdscr.refresh()

    def draw_modal(self, h, w):
        """Draw a modal overlay."""
        m_h, m_w = 16, 80
        y = (h - m_h) // 2
        x = (w - m_w) // 2

        # Draw box
        try:
            # Clear area
            for i in range(m_h):
                self.stdscr.addstr(y + i, x, " " * m_w)

            # Border
            self.stdscr.attron(curses.color_pair(3))

            # Manual draw for modal box
            h_box = m_h
            w_box = m_w
            self.stdscr.hline(y, x, curses.ACS_HLINE, w_box)
            self.stdscr.hline(y + h_box - 1, x, curses.ACS_HLINE, w_box)
            self.stdscr.vline(y, x, curses.ACS_VLINE, h_box)
            self.stdscr.vline(y, x + w_box - 1, curses.ACS_VLINE, h_box)
            self.stdscr.addch(y, x, curses.ACS_ULCORNER)
            self.stdscr.addch(y, x + w_box - 1, curses.ACS_URCORNER)
            self.stdscr.addch(y + h_box - 1, x, curses.ACS_LLCORNER)
            self.stdscr.addch(y + h_box - 1, x + w_box - 1, curses.ACS_LRCORNER)

            self.stdscr.attroff(curses.color_pair(3))

            # Title
            self.stdscr.addstr(
                y,
                x + 2,
                f" {self.modal['title']} ",
                curses.color_pair(3) | curses.A_BOLD,
            )

            # Content
            text = self.modal.get("text", "")
            if text:
                self.stdscr.addstr(y + 2, x + 2, text[: m_w - 4])

            # Input field
            if self.modal["type"] == "INPUT":
                val = self.modal.get("value", "")
                cursor_pos = self.modal.get("cursor_pos", len(val))

                field_w = m_w - 6
                field_h = m_h - 6  # Leave space for buttons/title
                # Import textwrap or use simple slicing
                # Simple character wrapping
                lines = []
                for i in range(0, len(val), field_w):
                    lines.append(val[i : i + field_w])
                if not lines:
                    lines = [""]

                # If cursor is at exact end, handle it?
                # Logic puts it at end of last line.

                # Ensure we have enough lines to cover cursor
                # Cursor (row, col)
                c_row = cursor_pos // field_w
                c_col = cursor_pos % field_w

                # Draw lines
                # We might need scrolling if text exceeds box height?
                # For now assuming it fits or we enforce limit.
                # Let's implement basic vertical scrolling if needed

                scroll_row = self.modal.get("scroll_row", 0)
                if c_row < scroll_row:
                    scroll_row = c_row
                elif c_row >= scroll_row + field_h:
                    scroll_row = c_row - field_h + 1
                self.modal["scroll_row"] = scroll_row

                for i in range(field_h):
                    line_idx = scroll_row + i
                    draw_y = y + 4 + i

                    line_content = ""
                    if line_idx * field_w < len(val):
                        # Construct line from val directly to rely on consistent math
                        start = line_idx * field_w
                        end = start + field_w
                        line_content = val[start:end]
                    elif line_idx == 0 and not val:
                        line_content = ""

                        # Only draw if valid line or active cursor line
                        # Use White (pair 6) for input text
                        self.stdscr.addstr(
                            draw_y, x + 3, line_content, curses.color_pair(6)
                        )

                    # Cursor
                    if line_idx == c_row:
                        # Ensure c_col is within bounds of visual line
                        # If cursor is at end of line (col=0 of next), handle it?
                        # No, math handles it: c_col is 0..width-1
                        # Cursor pos logic handles new line wrapping

                        char_at = " "
                        if c_col < len(line_content):
                            char_at = line_content[c_col]

                        self.stdscr.addstr(
                            draw_y,
                            x + 3 + c_col,
                            char_at,
                            curses.A_REVERSE | curses.color_pair(6),
                        )

            # Buttons
            btn_y = y + m_h - 2
            if self.modal["type"] == "CONFIRM":
                btns = "[y] Yes   [n] No"
                self.stdscr.addstr(btn_y, x + (m_w - len(btns)) // 2, btns)
            elif self.modal["type"] == "INPUT":
                btns = "[Enter] Confirm   [Esc] Cancel"
                self.stdscr.addstr(btn_y, x + (m_w - len(btns)) // 2, btns)

        except Exception:
            pass

    def draw_compact_gpu_info(self, y, x, h, w):
        """Draw compact nvidia-smi style info."""
        try:
            if not self.gpu_status:
                self.stdscr.addstr(y + 1, x + 2, "No GPU info available", curses.A_DIM)
                return

            # Column headers
            # Column headers
            # IDX:0, UTIL:5, MEM:10, PROCS:26
            header = "IDX  UTIL MEM            PROCESSES (USER:PID)"
            if w > 30:
                self.stdscr.addstr(
                    y, x + 2, header[: w - 4], curses.A_DIM | curses.A_UNDERLINE
                )

            for i, g in enumerate(self.gpu_status[: h - 1]):
                idx = g.get("index", "?")
                used_mb = g.get("used_mb", 0)
                total_mb = g.get("total_mb", 0)
                util = g.get("util", 0)
                # busy = not g.get("free", True) # Unused now

                # Status text gray regardless of state
                color = curses.A_DIM  # Lighter Gray

                used_gb = used_mb / 1024.0
                total_gb = total_mb / 1024.0
                mem_s = f"{used_gb:.1f}/{total_gb:.0f}G"

                line_y = y + 1 + i
                self.stdscr.addstr(
                    line_y, x + 2, f"{idx:<4} {util:>3}% {mem_s:<15}", color
                )

                procs = g.get("processes", [])
                proc_strs = [
                    f"{p.get('user', '?')}:{p.get('pid', '?')}"
                    for p in procs
                    if not p.get("zombie")
                ]
                proc_line = ", ".join(proc_strs)

                avail_proc_w = w - 28
                if len(proc_line) > avail_proc_w:
                    proc_line = proc_line[: avail_proc_w - 3] + "..."

                self.stdscr.addstr(line_y, x + 28, proc_line, curses.A_DIM)
        except Exception:
            pass

    def draw_job_details(self, y, x, h, w):
        """Draw detailed job information for the selected job across all windows."""
        try:
            # In NAV mode, show placeholder
            if self.mode == "NAV":
                self.stdscr.addstr(y + 1, x + 2, "No job selected.", curses.A_DIM)
                return

            # Find which job is "selected" across the 3 main windows
            # Or just use the one from the active window if it's a queue
            job = None
            if self.active_win_idx < 3:
                job = self.windows[self.active_win_idx].get_selected()
            else:
                # Search all for the "first" non-focused selection?
                # Let's just track the last selected job globally
                job = getattr(self, "last_selected_job", None)

            if not job:
                self.stdscr.addstr(y + 1, x + 2, "No job selected.", curses.A_DIM)
                return

            self.last_selected_job = job  # Keep it

            jid = job["id"]
            st = job.get("status", "unk")
            prio = job.get("priority", 1)
            prio_s = ["Low", "Med", "High", "Urgent"][min(prio, 3)]

            meta_str = f"ID: {jid} | Status: {st.upper()} | Prio: {prio_s}"
            self.stdscr.addstr(y, x + 2, meta_str, curses.A_BOLD)

            cmd = job.get("cmd", "")
            # Normalize command (remove newlines)
            cmd = cmd.replace("\n", " ").replace("\r", " ")

            prefix = "Cmd: "
            # Safer width calc: W - Left(2) - Right(2) - ScrollBar(1) - Prefix - Safety(2)
            # w - 5 - len(prefix) ?
            # w is Full Width.
            # Draws at x+2.
            # Max index w-2 (border at w-1).
            # So length available = w-4.
            # Subtract prefix.
            # Subtract 2 more for safety.
            safe_width = w - 6 - len(prefix)
            if safe_width < 10:
                safe_width = 10

            import textwrap

            lines = textwrap.wrap(cmd, width=safe_width)

            for i, line in enumerate(lines[: h - 1]):
                self.stdscr.addstr(
                    y + 1 + i,
                    x + 2,
                    prefix if i == 0 else " " * len(prefix),
                    curses.A_DIM,
                )
                self.stdscr.addstr(y + 1 + i, x + 2 + len(prefix), line)
        except Exception:
            pass

    def draw_window(self, win, y, x, h, w, active, focused):
        # Draw Box
        try:
            # Border Color logic
            if focused:
                color = curses.color_pair(5)  # Blue
            elif active:
                color = curses.color_pair(6)  # White
            else:
                color = curses.color_pair(7)  # Gray

            self.stdscr.attron(color)
            self.stdscr.hline(y, x, curses.ACS_HLINE, w)
            if not win.collapsed:
                self.stdscr.hline(y + h - 1, x, curses.ACS_HLINE, w)
                self.stdscr.vline(y, x, curses.ACS_VLINE, h)
                self.stdscr.vline(y, x + w - 1, curses.ACS_VLINE, h)
                self.stdscr.addch(y, x, curses.ACS_ULCORNER)
                self.stdscr.addch(y, x + w - 1, curses.ACS_URCORNER)
                self.stdscr.addch(y + h - 1, x, curses.ACS_LLCORNER)
                self.stdscr.addch(y + h - 1, x + w - 1, curses.ACS_LRCORNER)
            self.stdscr.attroff(color)

            # Title
            count_str = ""
            if win.key in ["running", "pending", "completed"]:
                count_str = f"[{len(win.items)}]"

            title_s = f" {win.title} {count_str} "
            if win.collapsed:
                title_s = f" [+] {win.title} {count_str} "
            elif focused:
                title_s = f" [ {win.title} {count_str} ] "

            style = curses.A_BOLD if active else 0
            # Use white for title text as requested
            self.stdscr.addstr(y, 2, title_s, curses.color_pair(6) | style)

            if win.collapsed:
                return

            # Dispatch specialized drawing
            if win.key == "gpu_status":
                self.draw_compact_gpu_info(y + 1, x, h - 2, w)
                return
            if win.key == "job_details":
                self.draw_job_details(y + 1, x, h - 2, w)
                return

            # Header?
            header_offset = 1
            hdr = ""
            if win.key == "running":
                hdr = " ID       PID   GPUS    ELAPSED  CMD"
            elif win.key == "pending":
                hdr = " ID       GPUS   PRIO     WAITING   CMD"
            elif win.key == "completed":
                hdr = " ID       RUNTIME   AGO       CMD"

            if hdr:
                self.stdscr.addstr(y + 1, 1, hdr, curses.A_DIM | curses.A_UNDERLINE)

            # List items
            display_items = list(win.items)

            # 1. NEW EDIT JOB INJECTION
            if (
                active
                and self.edit_mode_active
                and self.edit_is_new
                and self.edit_job
                and win.key == "pending"
            ):
                self.edit_job["_edit_field_idx"] = self.edit_field_idx
                target_idx = win.selected_idx + 1
                if target_idx <= len(display_items):
                    display_items.insert(target_idx, self.edit_job)

            # 2. EXISTING JOB SWAP
            elif (
                active
                and self.edit_mode_active
                and not self.edit_is_new
                and self.edit_job
                and win.key == "pending"
            ):
                # Find the job in display_items and replace it
                self.edit_job["_edit_field_idx"] = self.edit_field_idx
                for idx, it in enumerate(display_items):
                    if it["id"] == self.edit_job["id"]:
                        display_items[idx] = self.edit_job
                        break

            # Recalculate list height or just use what we have
            list_h = h - 2 - header_offset
            if list_h < 1:
                return

            start_y = y + 1 + header_offset

            # We need to handle scroll offset carefully if we injected an item
            # If we injected, the list is 1 longer.
            visible_items = display_items[
                win.scroll_offset : win.scroll_offset + list_h
            ]

            for i, item in enumerate(visible_items):
                abs_idx = win.scroll_offset + i

                # Check if this is the injected item
                is_injected = item.get("id") == "EDITING"

                # Selection logic:
                # If editing new job, visual selection is on injected item (rel idx + 1)

                is_sel = False
                if active:
                    if (
                        self.edit_mode_active
                        and self.edit_is_new
                        and win.key == "pending"
                    ):
                        if abs_idx == win.selected_idx + 1:
                            is_sel = True
                    else:
                        if abs_idx == win.selected_idx:
                            is_sel = True

                line_res, line_col = self.format_job_line(item, w - 2)

                draw_style = curses.A_NORMAL
                if is_sel:
                    if focused and not is_injected:
                        draw_style = curses.A_REVERSE
                    else:
                        # In NAV mode, pass or different style
                        pass

                # Handle Rich Text (Dictionary)
                if isinstance(line_res, dict) and line_res.get("type") == "rich":
                    current_x = 1
                    segments = line_res["segments"]
                    base_attr = line_res.get("base_color", curses.A_NORMAL)

                    # Clear line with base attr first?
                    # self.stdscr.addstr(start_y + i, 1, " " * (w-2), base_attr)

                    for text, attr in segments:
                        try:
                            self.stdscr.addstr(
                                start_y + i, current_x, text, attr | base_attr
                            )
                            current_x += len(text)
                        except Exception:
                            pass
                else:
                    self.stdscr.addstr(start_y + i, 1, line_res, line_col | draw_style)

            # Scroll bar indicator?
            if len(display_items) > list_h:
                sb_h = max(1, int(list_h * (list_h / len(display_items))))
                sb_pos = int((win.scroll_offset / len(display_items)) * list_h)
                for k in range(list_h):
                    char = "│"
                    if k >= sb_pos and k < sb_pos + sb_h:
                        char = "█"
                    try:
                        self.stdscr.addstr(start_y + k, w - 1, char, color)
                    except Exception:
                        pass
        except Exception:
            # self.stdscr.addstr(y+1, 1, str(e))
            pass

    def draw_footer(self, y, w):
        try:
            if self.modal:
                return  # Don't draw footer over modal or distract

            help_str = " Q:Quit "
            if self.edit_mode_active:
                help_str += (
                    "e:Confirm  Esc:Cancel  h/l:Field  j/k:Value  Enter:Edit Command"
                )
            elif self.mode == "NAV":
                help_str += "j/k:Select  l:Focus  n:New Job  Tab:Collapse"
            else:
                help_str += (
                    "h:Back  Space:Log  c:Cancel  r:Remove  d:Duplicate  p:Pause"
                )

            # Mode display on the right
            mode_label = self.mode
            if self.edit_mode_active:
                mode_label = "EDIT"

            mode_s = f" MODE: {mode_label} "
            padding = " " * (w - len(help_str) - len(mode_s))

            full_str = help_str + padding + mode_s
            # Blue background (Pair 5)
            self.stdscr.addstr(y, 0, full_str, curses.color_pair(5) | curses.A_REVERSE)
        except Exception:
            pass

    def draw_log_overlay(self, h, w):
        # Draw a floating window for logs
        margin_x = 4
        margin_y = 2
        win_h = h - 2 * margin_y
        win_w = w - 2 * margin_x

        # Draw shadow or clear
        for i in range(win_h):
            self.stdscr.addstr(margin_y + i, margin_x, " " * win_w, curses.A_NORMAL)

        # Box Border
        try:
            self.stdscr.attron(curses.color_pair(3))
            self.stdscr.hline(margin_y, margin_x, curses.ACS_HLINE, win_w)
            self.stdscr.hline(margin_y + win_h - 1, margin_x, curses.ACS_HLINE, win_w)
            self.stdscr.vline(margin_y, margin_x, curses.ACS_VLINE, win_h)
            self.stdscr.vline(margin_y, margin_x + win_w - 1, curses.ACS_VLINE, win_h)
            self.stdscr.addch(margin_y, margin_x, curses.ACS_ULCORNER)
            self.stdscr.addch(margin_y, margin_x + win_w - 1, curses.ACS_URCORNER)
            self.stdscr.addch(margin_y + win_h - 1, margin_x, curses.ACS_LLCORNER)
            self.stdscr.addch(
                margin_y + win_h - 1, margin_x + win_w - 1, curses.ACS_LRCORNER
            )
            self.stdscr.attroff(curses.color_pair(3))
            # Title
            title = f" LOGS: {self.log_job_id} "
            self.stdscr.addstr(
                margin_y, margin_x + 2, title, curses.A_BOLD | curses.A_REVERSE
            )

            # Content
            content_h = win_h - 2
            visible_lines = self.log_content[
                self.log_scroll : self.log_scroll + content_h
            ]

            for i, line in enumerate(visible_lines):
                if len(line) > win_w - 2:
                    line = line[: win_w - 5] + "..."
                self.stdscr.addstr(
                    margin_y + 1 + i, margin_x + 1, line, curses.A_NORMAL
                )

            # Footer
            footer_str = " h/Esc:Close  j/k:Scroll  PGUP/DN:Jump  L:Full(less) "
            self.stdscr.addstr(
                margin_y + win_h - 1, margin_x + 2, footer_str, curses.A_REVERSE
            )

        except Exception:
            pass

    def action_view_logs(self):
        """View logs for selected job using external tool."""
        win = self.windows[self.active_win_idx]
        job = win.get_selected()
        if not job:
            return

        self.action_open_external_logs()

    def action_open_external_logs(self):
        """Open logs in an external tool (less +F)."""
        if self.mode != "ACTION":
            return
        win = self.windows[self.active_win_idx]
        job = win.get_selected()
        if not job:
            return

        log_path = Path.home() / f".gpu_queue/logs/{job['id']}.log"
        if not log_path.exists():
            self.action_msg = "Log file not found"
            self.msg_clear_time = time.time() + 2.0
            return

        # We need to temporarily exit curses
        curses.def_shell_mode()
        self.stdscr.clear()
        self.stdscr.refresh()
        curses.endwin()

        try:
            # Use +F for following if it's currently running, otherwise just open it
            cmd = ["less", "+G", str(log_path)]
            if job["_type"] == "running":
                cmd = ["less", "+F", str(log_path)]

            # Ignore SIGINT in parent (Python) so Ctrl+C only kills 'less'
            old_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                subprocess.run(cmd)
            finally:
                # Restore SIGINT handler
                signal.signal(signal.SIGINT, old_handler)
        finally:
            # Re-enter curses
            self.stdscr.refresh()
            curses.doupdate()
            # Some implementations need reset_shell_mode
            curses.reset_shell_mode()
            self.stdscr.keypad(True)
            self.stdscr.nodelay(True)

    def main(self, stdscr):
        import curses

        self.stdscr = stdscr

        curses.start_color()
        curses.use_default_colors()

        # curr_y, curr_x = 0, 0

        # Define colors (1: Red, 2: Green, 3: Yellow, 4: Cyan/Blue)
        curses.init_pair(1, curses.COLOR_RED, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
        curses.init_pair(4, curses.COLOR_CYAN, -1)
        curses.init_pair(5, curses.COLOR_BLUE, -1)
        curses.init_pair(6, curses.COLOR_WHITE, -1)
        curses.init_pair(
            7, 8, -1
        )  # Gray (if color 8 exists, else fallback happens or it's ignored)
        # Fallback for gray if 8 is not available
        if curses.COLORS > 8:
            pass
        else:
            curses.init_pair(
                7, curses.COLOR_WHITE, -1
            )  # Fallback to white but usually use A_DIM

        curses.curs_set(0)
        self.stdscr.nodelay(True)
        self.stdscr.keypad(True)

        self.start()

        try:
            while True:
                self.draw()

                ch = self.stdscr.getch()
                if ch == -1:
                    time.sleep(0.05)
                    continue

                if self.modal:
                    m = self.modal
                    if m["type"] == "CONFIRM":
                        if ch == ord("y") or ch == 10:  # Yes
                            if m["on_confirm"]:
                                m["on_confirm"]()
                            self.modal = None
                        elif ch == ord("n") or ch == 27:  # No / Esc
                            if m["on_cancel"]:
                                m["on_cancel"]()
                            self.modal = None

                    elif m["type"] == "INPUT":
                        cpos = m.get("cursor_pos", len(m["value"]))
                        val = m["value"]

                        if ch == 27:  # Esc
                            if m["on_cancel"]:
                                m["on_cancel"]()
                            self.modal = None
                        elif ch == 10:  # Enter
                            if m["on_confirm"]:
                                m["on_confirm"](val)
                            self.modal = None
                        elif ch == 127 or ch == curses.KEY_BACKSPACE:  # Backspace
                            if cpos > 0:
                                m["value"] = val[: cpos - 1] + val[cpos:]
                                m["cursor_pos"] = cpos - 1
                        elif ch == curses.KEY_DC:  # Delete
                            if cpos < len(val):
                                m["value"] = val[:cpos] + val[cpos + 1 :]
                        elif ch == curses.KEY_LEFT:
                            m["cursor_pos"] = max(0, cpos - 1)
                        elif ch == curses.KEY_RIGHT:
                            m["cursor_pos"] = min(len(val), cpos + 1)
                        elif ch == curses.KEY_UP:
                            # Move up one line (width 74 = 80 - 6)
                            width = 80 - 6
                            m["cursor_pos"] = max(0, cpos - width)
                        elif ch == curses.KEY_DOWN:
                            width = 80 - 6
                            m["cursor_pos"] = min(len(val), cpos + width)
                        elif ch == curses.KEY_HOME:
                            m["cursor_pos"] = 0
                        elif ch == curses.KEY_END:
                            m["cursor_pos"] = len(val)
                        elif ch >= 32 and ch <= 126:  # Printable
                            m["value"] = val[:cpos] + chr(ch) + val[cpos:]
                            m["cursor_pos"] = cpos + 1
                    continue

                if self.edit_mode_active:
                    # Force redraw of footer/status
                    # self.stdscr.touchwin()

                    if ch == 27:  # Esc -> Cancel
                        self.edit_mode_active = False
                        self.edit_job = None
                    elif ch == ord("e"):  # Confirm
                        if self.edit_is_new:
                            self.add_job_internal(
                                self.edit_job["cmd"],
                                self.edit_job["gpus"],
                                self.edit_job["priority"],
                            )
                        else:
                            # Update existing job
                            self.execute_action(
                                "update_job",
                                self.edit_job["id"],
                                cmd=self.edit_job["cmd"],
                                gpus=self.edit_job["gpus"],
                                priority=self.edit_job["priority"],
                            )
                        self.edit_mode_active = False
                        self.edit_job = None

                    elif ch == ord("h"):  # Cycle Left
                        self.edit_field_idx = max(0, self.edit_field_idx - 1)
                    elif ch == ord("l"):  # Cycle Right
                        self.edit_field_idx = min(2, self.edit_field_idx + 1)

                    elif ch == ord("j"):  # Decrease Value
                        if self.edit_field_idx == 0:  # GPUS
                            self.edit_job["gpus"] = max(
                                1, int(self.edit_job["gpus"]) - 1
                            )
                        elif self.edit_field_idx == 1:  # Prio
                            self.edit_job["priority"] = max(
                                1, int(self.edit_job["priority"]) - 1
                            )

                    elif ch == ord("k"):  # Increase Value
                        if self.edit_field_idx == 0:  # GPUS
                            self.edit_job["gpus"] = int(self.edit_job["gpus"]) + 1
                            # Assuming no hard max, or maybe max_gpus from status?
                        elif self.edit_field_idx == 1:  # Prio
                            self.edit_job["priority"] = min(
                                3, int(self.edit_job["priority"]) + 1
                            )

                    elif ch == 10 or ch == ord("i"):  # Enter/Edit Text
                        if self.edit_field_idx == 2:  # Command
                            self.prompt_edit_command()

                    continue

                # Log Viewing Mode
                if self.viewing_logs:
                    if ch == ord("h") or ch == ord("q") or ch == 27:  # h or q or Esc
                        self.viewing_logs = False
                    elif ch == ord("k"):
                        self.log_scroll = max(0, self.log_scroll - 1)
                    elif ch == ord("j"):
                        max_scroll = max(
                            0, len(self.log_content) - (self.stdscr.getmaxyx()[0] - 6)
                        )
                        self.log_scroll = min(max_scroll, self.log_scroll + 1)
                    continue

                # Normal Mode
                if ch == ord("q"):
                    break

                if self.mode == "NAV":
                    if ch == ord("k"):
                        self.active_win_idx = max(0, self.active_win_idx - 1)
                    elif ch == ord("j"):
                        self.active_win_idx = min(
                            len(self.windows) - 1, self.active_win_idx + 1
                        )
                    elif ch == ord("l"):  # l
                        # Disable l for non-interactive windows
                        curr_win = self.windows[self.active_win_idx]
                        if curr_win.key not in ["gpu_status", "job_details"]:
                            self.mode = "ACTION"
                            curr_win.collapsed = False
                    elif ch == 10:  # Enter
                        curr_win = self.windows[self.active_win_idx]
                        if curr_win.key not in ["gpu_status", "job_details"]:
                            self.mode = "ACTION"
                            curr_win.collapsed = False
                    elif ch == 9:  # Tab
                        self.windows[self.active_win_idx].collapsed = not self.windows[
                            self.active_win_idx
                        ].collapsed
                    elif ch == ord("n"):  # New Job (Global context)
                        self.prompt_new_job()

                elif self.mode == "ACTION":
                    win = self.windows[self.active_win_idx]
                    h = (self.stdscr.getmaxyx()[0] - 5) // 3  # approx height per window

                    if ch == ord("h") or ch == 27:  # h or Esc
                        self.mode = "NAV"
                        # Reset scroll to top
                        win.scroll_offset = 0
                        win.selected_idx = 0
                    elif ch == ord("k"):
                        win.scroll(-1, h)
                    elif ch == ord("j"):
                        win.scroll(1, h)
                    elif ch == ord(" "):
                        self.action_view_logs()
                    elif ch == ord("L"):
                        self.action_open_external_logs()

                    # Actions
                    elif ch == ord("c"):  # Cancel
                        self.do_action("cancel")
                    elif ch == ord("r"):  # Remove
                        self.do_action("remove")
                    elif ch == ord("d"):  # Dup
                        self.do_action("dup")
                    elif ch == ord("n"):  # New
                        self.prompt_new_job()
                    elif ch == ord("p"):  # Pause
                        if win.key == "running":
                            self.do_action("pause")
                    elif ch == ord("e"):  # Edit
                        self.do_action("edit")

        finally:
            self.stop()

    def add_job_internal(self, cmd, gpus=2, priority=1):
        job = {
            "id": generate_job_id(),
            "cmd": cmd,
            "gpus": gpus,
            "added": datetime.now().isoformat(),
            "priority": priority,
            "cwd": os.getcwd(),
        }
        with locked_queue() as queue:
            queue["pending"].append(job)
        self.action_msg = f"Added {job['id']}"
        self.msg_clear_time = time.time() + 2.0

    def prompt_edit_command(self):
        # Use external editor
        # 1. Write current cmd to temp file
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".sh") as tf:
            tf.write(self.edit_job["cmd"])
            tf_path = tf.name

        # 2. Open editor
        curses.def_shell_mode()
        self.stdscr.clear()
        self.stdscr.refresh()
        curses.endwin()

        try:
            editor = os.environ.get("EDITOR", "nano")
            subprocess.run([editor, tf_path])

            # 3. Read back
            with open(tf_path, "r") as f:
                new_cmd = f.read().strip()
                self._update_edit_cmd(new_cmd)

            os.unlink(tf_path)
        except Exception:
            pass

        finally:
            self.stdscr.refresh()
            curses.doupdate()
            curses.reset_shell_mode()
            self.stdscr.keypad(True)
            self.stdscr.nodelay(True)

    def _update_edit_cmd(self, val):
        if self.edit_job:
            self.edit_job["cmd"] = val

    def prompt_new_job(self):
        # Instead of modal, start Edit Mode with defaults
        self.edit_job = {
            "id": "NEW",
            "_type": "pending",
            "cmd": "",
            "gpus": 1,
            "priority": 1,  # Low
            "added": datetime.now().isoformat(),
        }
        self.edit_is_new = True
        self.edit_mode_active = True
        self.edit_field_idx = 0

        # Switch to PENDING window if not already there
        for i, w in enumerate(self.windows):
            if w.key == "pending":
                self.active_win_idx = i
                break

    def prompt_change_gpus(self):
        win = self.windows[self.active_win_idx]
        job = win.get_selected()
        if not job:
            return
        self.modal = {
            "type": "INPUT",
            "title": "Change GPU Requirement",
            "text": f"New GPU count for {job['id']}:",
            "value": str(job.get("gpus", 1)),
            "cursor_pos": len(str(job.get("gpus", 1))),
            "on_confirm": lambda val: self.execute_action(
                "change_gpus", job["id"], new_val=val
            ),
            "on_cancel": None,
        }

    def do_action(self, action):
        """Perform action on selected job."""
        win = self.windows[self.active_win_idx]
        job = win.get_selected()
        if not job:
            return

        jid = job["id"]

        # Actions that require modals
        if action == "cancel":
            self.modal = {
                "type": "CONFIRM",
                "title": "Cancel Job",
                "text": f"Cancel job {jid}?",
                "on_confirm": lambda: self.execute_action("cancel", jid),
                "on_cancel": None,
            }
            return

        elif action == "remove":
            if win.key != "completed":
                self.action_msg = "Can only remove finished jobs"
                self.msg_clear_time = time.time() + 2.0
                return

            self.modal = {
                "type": "CONFIRM",
                "title": "Remove Job",
                "text": f"Permanently delete {jid}?",
                "on_confirm": lambda: self.execute_action("delete", jid),
                "on_cancel": None,
            }
            return

        elif action == "edit":
            if win.key != "pending":
                self.action_msg = "Can only edit pending jobs"
                self.msg_clear_time = time.time() + 2.0
                return

            self.edit_job = copy.deepcopy(job)
            self.edit_job["_type"] = "pending"
            self.edit_is_new = False
            self.edit_mode_active = True
            self.edit_field_idx = 0
            return

        elif action == "dup":
            self.edit_job = copy.deepcopy(job)
            self.edit_job["id"] = "EDITING"  # Temp ID
            self.edit_job["_type"] = "pending"  # Force type to pending for rendering
            self.edit_is_new = True
            self.edit_mode_active = True
            self.edit_field_idx = 0

            # Switch to PENDING window if not already there, as new jobs go there
            # Find index of "pending" window
            for i, w in enumerate(self.windows):
                if w.key == "pending":
                    self.active_win_idx = i
                    break

            return

        # Immediate actions
        self.execute_action(action, jid)

    def execute_action(self, action, jid, **kwargs):
        # Re-use existing cmd functions if possible, or call logic directly
        msg = ""
        try:
            if action == "cancel":
                # If it's a pending job, move to completed with status 'cancelled'
                # If it's running, call the external tool
                is_pending = False
                with locked_queue() as q:
                    for j in q["pending"]:
                        if j["id"] == jid:
                            is_pending = True
                            break

                if is_pending:
                    with locked_queue() as q:
                        for i, j in enumerate(q["pending"]):
                            if j["id"] == jid:
                                job = q["pending"].pop(i)
                                job["status"] = "cancelled"
                                job["ended"] = datetime.now().isoformat()
                                q["completed"].insert(0, job)
                                msg = f"Cancelled {jid}"
                                break
                else:
                    # Running job
                    subprocess.Popen(
                        ["gpu-queue", "cancel", jid],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    msg = f"Cancelling {jid}..."

            elif action == "delete":
                with locked_queue() as q:
                    for i, j in enumerate(q["completed"]):
                        if j["id"] == jid:
                            q["completed"].pop(i)
                            msg = f"Deleted {jid}"
                            break

            elif action == "pause":
                subprocess.Popen(
                    ["gpu-queue", "pause", jid],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                msg = f"Pausing {jid}..."

            elif action == "priority_up" or action == "priority_down":
                with locked_queue() as q:
                    for job in q["pending"]:
                        if job["id"] == jid:
                            p = job.get("priority", 1)
                            if action == "priority_up":
                                job["priority"] = min(3, p + 1)
                                dir_s = "Increased"
                            else:
                                job["priority"] = max(0, p - 1)
                                dir_s = "Decreased"
                            msg = f"{dir_s} priority of {jid} to {job['priority']}"
                            break

            elif action == "update_job":
                cmd = kwargs.get("cmd")
                gpus = kwargs.get("gpus")
                prio = kwargs.get("priority")

                with locked_queue() as q:
                    for job in q["pending"]:
                        if job["id"] == jid:
                            if cmd:
                                job["cmd"] = cmd
                            if gpus:
                                job["gpus"] = gpus
                            if prio:
                                job["priority"] = prio
                            msg = f"Updated job {jid}"
                            break

        except Exception as e:
            msg = f"Err: {str(e)}"

        self.action_msg = msg
        self.msg_clear_time = time.time() + 2.0


def cmd_watch(args):
    """Interactive TUI for queue monitoring."""
    import curses

    tui = GPUQueueTUI(args.interval)
    try:
        curses.wrapper(tui.main)
    except KeyboardInterrupt:
        pass
    print("Exited TUI.")


def main():
    parser = argparse.ArgumentParser(description="GPU Job Queue Scheduler")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # add
    add_parser = subparsers.add_parser("add", help="Add a job to the queue")
    add_parser.add_argument("command", help="Command to run")
    add_parser.add_argument(
        "--gpus", "-g", type=int, default=2, help="Number of GPUs required"
    )
    add_parser.add_argument(
        "--priority",
        "-p",
        choices=["low", "medium", "high"],
        default="medium",
        help="Job priority",
    )
    add_parser.add_argument(
        "--front",
        "-f",
        action="store_true",
        help="Add to front of queue (Urgent priority)",
    )
    add_parser.set_defaults(func=cmd_add)

    # start
    start_parser = subparsers.add_parser(
        "start", help="Start the queue scheduler (background)"
    )
    start_parser.add_argument(
        "--min-free", type=int, default=2, help="Number of GPUs to always keep free"
    )
    start_parser.set_defaults(func=cmd_start)

    # stop
    stop_parser = subparsers.add_parser("stop", help="Stop the background scheduler")
    stop_parser.set_defaults(func=cmd_stop)

    # serve
    serve_parser = subparsers.add_parser(
        "serve", help="Run the queue scheduler (foreground)"
    )
    serve_parser.add_argument(
        "--min-free", type=int, default=2, help="Number of GPUs to always keep free"
    )
    serve_parser.add_argument(
        "--exclude-gpus",
        type=str,
        default="",
        help="Comma-separated list of GPU indices to ignore (e.g. '0,1')",
    )
    serve_parser.set_defaults(func=cmd_serve)

    # status removed

    # cancel
    cancel_parser = subparsers.add_parser("cancel", help="Cancel a pending job")
    cancel_parser.add_argument("job_id", help="Job ID to cancel")
    cancel_parser.set_defaults(func=cmd_cancel)

    # logs
    logs_parser = subparsers.add_parser("logs", help="Show job logs")
    logs_parser.add_argument("job_id", help="Job ID")
    logs_parser.add_argument(
        "--lines", "-n", type=int, default=50, help="Number of lines"
    )
    logs_parser.set_defaults(func=cmd_logs)

    # clear
    clear_parser = subparsers.add_parser("clear", help="Clear completed jobs")
    clear_parser.set_defaults(func=cmd_clear)

    # retry
    retry_parser = subparsers.add_parser("retry", help="Re-queue a completed job")
    retry_parser.add_argument("job_id", help="Job ID to retry")
    retry_parser.add_argument(
        "--front", "-f", action="store_true", help="Add to front of queue"
    )
    retry_parser.set_defaults(func=cmd_retry)

    # pause
    pause_parser = subparsers.add_parser(
        "pause", help="Pause a running job (re-queue at front)"
    )
    pause_parser.add_argument("job_id", help="Job ID to pause")
    pause_parser.set_defaults(func=cmd_pause)

    # watch
    watch_parser = subparsers.add_parser(
        "watch", help="Watch queue status continuously"
    )
    watch_parser.add_argument(
        "--interval", "-n", type=float, default=2.0, help="Update interval in seconds"
    )
    watch_parser.set_defaults(func=cmd_watch)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
