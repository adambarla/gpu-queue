#!/usr/bin/env python3
"""
GPU Job Queue Scheduler

A lightweight job queue for shared GPU servers without SLURM.
Automatically runs jobs when the required number of GPUs become available.

Usage:
    gpu-queue add --gpus 2 "uv run scripts/train.py model=deppo L=1"
    gpu-queue start
    gpu-queue status
    gpu-queue cancel <job_id>
    gpu-queue stop
    gpu-queue logs <job_id>
"""

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import textwrap
from typing import Any

# Configuration
QUEUE_DIR = Path.home() / ".gpu_queue"
QUEUE_FILE = QUEUE_DIR / "jobs.json"
PID_FILE = QUEUE_DIR / "daemon.pid"
DAEMON_LOG = QUEUE_DIR / "daemon.log"
LOG_DIR = QUEUE_DIR / "logs"
LOCK_FILE = QUEUE_DIR / "queue.lock"
DAEMON_LOCK_FILE = QUEUE_DIR / "daemon.lock"

POLL_INTERVAL = 2  # seconds between GPU checks
MIN_FREE_GPUS = 2  # Number of GPUs to always keep free for other users


def ensure_dirs():
    """Create queue directories if they don't exist."""
    QUEUE_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)


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
    queue["pending"].sort(
        key=lambda x: (-x.get("priority", 1), x["added"])
    )


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
                "--query-gpu=index,memory.used,memory.total",
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
                idx, used, total = int(parts[0]), int(parts[1]), int(parts[2])
                gpus[idx] = {
                    "index": idx,
                    "used_mb": used,
                    "total_mb": total,
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
                    gpu_uuid, pid_str, proc_name, mem_str = parts[0], parts[1], parts[2], parts[3]
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
                            
                            gpus[idx]["processes"].append({
                                "pid": pid_str,
                                "user": user,
                                "name": proc_name.split("/")[-1] if "/" in proc_name else proc_name,
                                "mem_mb": int(mem_str) if mem_str.isdigit() else 0,
                                "zombie": is_zombie,
                            })
                            
                            # Only mark as busy if NOT a zombie
                            if not is_zombie:
                                gpus[idx]["free"] = False

        return list(gpus.values())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


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
                
                # Process is dead. Give the shell wrapper a tiny bit of time to write use .exit file
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
                        except: pass
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
    log_file = LOG_DIR / f"{job['id']}.log"
    exit_file = QUEUE_DIR / f"{job['id']}.exit"
    gpu_str = ",".join(map(str, gpu_indices))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_str
    
    # Ensure ~/.local/bin is in PATH for uv (and other user tools)
    local_bin = str(Path.home() / ".local" / "bin")
    if local_bin not in env.get("PATH", ""):
        env["PATH"] = f"{local_bin}:{env.get('PATH', '')}"

    with open(log_file, "w") as f:
        f.write(f"=== Job {job['id']} ===\n")
        f.write(f"Command: {job['cmd']}\n")
        f.write(f"GPUs: {gpu_str}\n")
        f.write(f"Started: {datetime.now().isoformat()}\n")
        f.write(f"CWD: {job.get('cwd', 'default')}\n")
        f.write("=" * 40 + "\n\n")
    
    # Wrap in shell to capture exit code. Quote paths to be safe.
    q_log = f"'{log_file}'"
    q_exit = f"'{exit_file}'"
    wrapped_cmd = f"({job['cmd']}) >> {q_log} 2>&1; echo $? > {q_exit}"
    
    # Determine working directory
    # 1. Use job['cwd'] if available
    # 2. Fallback to ~/pepo for backward compatibility (hardcoded in original)
    # 3. Fallback to home
    cwd = Path.home()
    if job.get("cwd"):
        cwd = Path(job["cwd"])
    elif (Path.home() / "pepo").exists():
        cwd = Path.home() / "pepo"
    
    proc = subprocess.Popen(
        wrapped_cmd,
        shell=True,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=cwd,
        start_new_session=True 
    )

    return proc.pid


def log_msg(msg: str, verbose: bool = False):
    """Log a message to the daemon log."""
    if verbose:
        return  # Skip verbose messages for now
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    with open(DAEMON_LOG, "a") as f:
        f.write(line)


def cmd_serve(args):
    """Run the scheduler loop in the foreground."""
    ensure_dirs()
    
    # Try to acquire daemon lock to prevent multiple instances
    lock_fd = open(DAEMON_LOCK_FILE, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Error: Another instance of gpu-queue (serve/start) is already running!")
        sys.exit(1)

    print(f"✓ Scheduler started (keeping {args.min_free} GPUs reserved for other users)")
    print(f"  Polling every {POLL_INTERVAL}s")
    
    # Write PID to file for stop command
    PID_FILE.write_text(str(os.getpid()))

    # Store config
    config = {"min_free_gpus": args.min_free}
    
    # Parse excluded GPUs
    excluded = set()
    if args.exclude_gpus:
        try:
             for p in args.exclude_gpus.split(","):
                 if p.strip():
                     excluded.add(int(p.strip()))
        except ValueError:
            print(f"Error: Invalid format for --exclude-gpus. Use comma-separated integers.")
            sys.exit(1)
            
        print(f"  Excluding GPUs: {sorted(list(excluded))}")
        config["excluded_gpus"] = list(excluded)

    (QUEUE_DIR / "config.json").write_text(json.dumps(config))

    while True:
        try:
            cleanup_dead_jobs()
            
            with locked_queue() as queue:
                if not queue["pending"]:
                    time.sleep(POLL_INTERVAL)
                    continue

                # --- Quota and Availability Logic ---
                all_gpus = get_free_gpus()
                
                # Filter out excluded GPUs
                gpus = [g for g in all_gpus if g["index"] not in excluded]
                
                total_gpus = len(gpus)
                
                # Excluded GPUs count towards the reserved quota
                effective_min_free = max(0, args.min_free - len(excluded))
                
                quota = total_gpus - effective_min_free
                
                our_usage = sum(job.get("gpus", 1) for job in queue["running"])
                quota_remaining = quota - our_usage
                
                our_assigned = set()
                for j in queue["running"]:
                    for idx in j.get("assigned_gpus", []):
                        our_assigned.add(idx)
                
                free_indices = [g["index"] for g in gpus if g["free"] and g["index"] not in our_assigned]
                
                if quota_remaining <= 0:
                    time.sleep(POLL_INTERVAL)
                    continue

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
                        job["started"] = datetime.now().isoformat()
                        queue["running"].append(job)
                        
                        quota_remaining -= req
                        free_indices = free_indices[req:]
                        jobs_started = True
                    else:
                        remaining_pending.append(job)
                
                if jobs_started:
                    queue["pending"] = remaining_pending
                    # Saved automatically by locked_queue

            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            print("\nStopping scheduler.")
            break
        except Exception as e:
            log_msg(f"Error: {e}")
            time.sleep(POLL_INTERVAL)


# === CLI Commands ===


def cmd_add(args):
    """Add a job to the queue."""
    priorities = {"low": 0, "medium": 1, "high": 2}
    prio = priorities.get(args.priority, 1)
    if args.front: prio = 3

    job = {
        "id": generate_job_id(),
        "cmd": args.command,
        "gpus": args.gpus,
        "added": datetime.now().isoformat(),
        "priority": prio,
        "cwd": os.getcwd(), # Capture current working directory
    }
    
    with locked_queue() as queue:
        queue["pending"].append(job)

    print(f"✓ Added job {job['id']} (requires {args.gpus} GPUs)")
    print(f"  Command: {args.command}")
    print(f"  CWD: {os.getcwd()}")


def cmd_start(args):
    """Start the daemon."""
    ensure_dirs()
    
    # Check lock
    if DAEMON_LOCK_FILE.exists():
        # Try to acquire lock non-blocking to see if locked
        f = open(DAEMON_LOCK_FILE, "w")
        try:
             fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
             fcntl.flock(f, fcntl.LOCK_UN)
        except BlockingIOError:
             print("Daemon is already running (locked)!")
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

    # We need to pass args to cmd_serve-like logic
    # Just call cmd_serve, it handles locking too
    # But cmd_serve prints to stdout, which is redirected to log. That's fine.
    try:
        cmd_serve(args)
    except SystemExit:
        pass


def cmd_stop(args):
    """Stop the daemon."""
    if not PID_FILE.exists():
        print("Daemon PID file not found.")
        return

    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        print(f"✓ Stopped daemon (PID: {pid})")
    except ProcessLookupError:
        print("Daemon process not found (stale PID file?)")
        PID_FILE.unlink(missing_ok=True)
    except Exception as e:
        print(f"Error stopping daemon: {e}")


def get_terminal_width() -> int:
    """Get the current terminal width."""
    return shutil.get_terminal_size((80, 20)).columns


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
                        os.killpg(pid, signal.SIGKILL) # Aggressive kill
                    except: pass
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


def cmd_retry(args):
    """Re-queue a completed job."""
    with locked_queue() as queue:
        # Find job in completed
        for i, job in enumerate(queue["completed"]):
            if job["id"] == args.job_id:
                # Remove from completed
                queue["completed"].pop(i)
                
                # Reset job metadata
                new_job = {
                    "id": generate_job_id(),
                    "cmd": job["cmd"],
                    "gpus": job.get("gpus", 1),
                    "added": datetime.now().isoformat(),
                    "retry_of": job["id"],
                    "priority": 1,
                    "cwd": job.get("cwd") # Preserve CWD
                }
                
                # Add to front or back of pending
                if args.front:
                    queue["pending"].insert(0, new_job)
                    print(f"✓ Re-queued job {job['id']} as {new_job['id']} (front)")
                else:
                    queue["pending"].append(new_job)
                    print(f"✓ Re-queued job {job['id']} as {new_job['id']} (back)")
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
                    except: pass
                
                # Remove from running
                queue["running"].pop(i)
                
                # Reset metadata for re-queue
                new_job = {
                    "id": generate_job_id(),
                    "cmd": job["cmd"],
                    "gpus": job.get("gpus", 1),
                    "added": datetime.now().isoformat(),
                    "priority": 3, # Urgent
                    "paused_from": job["id"],
                    "cwd": job.get("cwd"),
                }
                
                # Add to front of pending queue
                queue["pending"].insert(0, new_job)
                print(f"✓ Paused job {job['id']} (Killed process group {pid})")
                print(f"✓ Re-queued as {new_job['id']} at front")
                return
                
    print(f"Job {args.job_id} not found in running jobs")


def cmd_status(args):
    """Check if the daemon is running."""
    # Check lock file
    running = False
    pid = "unknown"
    
    if DAEMON_LOCK_FILE.exists():
        # Try to acquire lock non-blocking
        f = open(DAEMON_LOCK_FILE, "w")
        try:
             fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
             fcntl.flock(f, fcntl.LOCK_UN)
             # If we can lock it, it means NO ONE else is holding it -> Not Running
             running = False
        except BlockingIOError:
             # Could not lock -> Someone else holds it -> Running
             running = True
    
    if running:
        if PID_FILE.exists():
            try:
                pid = PID_FILE.read_text().strip()
                # Verify process exists
                try:
                    os.kill(int(pid), 0)
                except OSError:
                    running = False # Stale lock? Or lock held but pid file wrong?
                    # Actually if lock is held, process exists. 
                    pass
            except: pass
            
        print(f"✓ Daemon is RUNNING (PID: {pid})")
        print(f"  Log: {DAEMON_LOG}")
    else:
        print("✗ Daemon is STOPPED")



class GPUQueueTUI:
    """Interactive Text User Interface for GPU Queue."""
    
    def __init__(self, interval: float):
        self.interval = interval
        self.lock = threading.Lock()
        self.stdscr = None
        self.stop_event = threading.Event()
        self.ui_rows = [] 
        self.data = {}
        self.data_ready = False
        
        self.scroll_offset = 0
        self.selected_idx = 0
        
        self.action_msg = ""
        self.msg_clear_time = 0

    def get_selected_item(self):
        """Get the currently selected item (job or header)."""
        if 0 <= self.selected_idx < len(self.ui_rows):
            return self.ui_rows[self.selected_idx]
        return None

    def background_update_loop(self):
        """Periodically fetch queue status."""
        while not self.stop_event.is_set():
            try:
                # Load queue safely
                if QUEUE_FILE.exists():
                    try:
                        raw = QUEUE_FILE.read_text()
                        data = json.loads(raw)
                    except:
                        data = {"running":[], "pending":[], "completed":[]}
                else:
                    data = {"running":[], "pending":[], "completed":[]}

                # Get GPU status
                gpu_info = get_free_gpus()
                
                with self.lock:
                    self.data = {
                        "queue": data,
                        "gpus": gpu_info,
                        "min_free": data.get("meta", {}).get("config", {}).get("min_free", 2)
                    }
                    
                    # Tag jobs with type
                    for j in data["running"]: j["_type"] = "running"
                    for j in data["pending"]: j["_type"] = "pending"
                    for j in data["completed"]: j["_type"] = "completed"
                    
                    # Sort completed by end time (descending)
                    def sort_key(j):
                        t = j.get("ended") or j.get("started") or j.get("added") or ""
                        return t
                    data["completed"].sort(key=sort_key, reverse=True)

                    # Build UI Rows with Headers
                    new_rows = []
                    if data["running"]:
                        new_rows.append("--- RUNNING ---")
                        new_rows.extend(data["running"])
                    
                    if data["pending"]:
                        new_rows.append("--- PENDING ---")
                        new_rows.extend(data["pending"])
                    
                    if data["completed"]:
                        new_rows.append("--- COMPLETED ---")
                        new_rows.extend(data["completed"])
                        
                    self.ui_rows = new_rows
                    self.data_ready = True
                    
                    # Adjust selection if out of bounds
                    if self.selected_idx >= len(self.ui_rows):
                        self.selected_idx = max(0, len(self.ui_rows) - 1)

            except Exception as e:
                pass
            
            time.sleep(self.interval)

    def start(self):
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.background_update_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1.0)

    def format_duration(self, start_str, end_str=None):
        if not start_str: return "-"
        try:
            start_dt = datetime.fromisoformat(start_str)
            if end_str:
                end_dt = datetime.fromisoformat(end_str)
            else:
                end_dt = datetime.now()
            
            delta = end_dt - start_dt
            total = int(delta.total_seconds())
            if total < 60: return f"{total}s"
            if total < 3600: return f"{total//60}m"
            return f"{total//3600}h {total%3600//60}m"
        except: return "-"

    def action_cancel(self):
        """Cancel/Kill selected job."""
        item = self.get_selected_item()
        if not item or isinstance(item, str): return
        job = item

        self.action_msg = f"Cancelling {job['id']}..."
        self.msg_clear_time = time.time() + 2
        
        with locked_queue() as queue:
            # Check pending
            for i, j in enumerate(queue["pending"]):
                if j["id"] == job["id"]:
                    queue["pending"].pop(i)
                    self.action_msg = f"Cancelled pending {job['id']}"
                    return

            # Check running
            for i, j in enumerate(queue["running"]):
                if j["id"] == job["id"]:
                    pid = j.get("pid")
                    if pid:
                        try: os.killpg(pid, signal.SIGTERM)
                        except: pass
                    queue["running"].pop(i)
                    j["status"] = "cancelled"
                    j["ended"] = datetime.now().isoformat()
                    queue["completed"].append(j)
                    self.action_msg = f"Killed running {job['id']}"
                    return
        
        self.action_msg = f"Job {job['id']} not found"

    def action_retry(self):
        """Retry completed job."""
        item = self.get_selected_item()
        if not item or isinstance(item, str): return
        job = item
        
        if job["_type"] != "completed":
            self.action_msg = "Can only retry COMPLETED jobs"
            self.msg_clear_time = time.time() + 2
            return

        self.action_msg = f"Retrying {job['id']}..."
        self.msg_clear_time = time.time() + 2
        
        with locked_queue() as queue:
             new_job = {
                "id": generate_job_id(),
                "cmd": job["cmd"],
                "gpus": job.get("gpus", 1),
                "added": datetime.now().isoformat(),
                "retry_of": job["id"],
                "priority": 1,
                "cwd": job.get("cwd")
             }
             queue["pending"].append(new_job)

    def action_pause(self):
        """Pause selected job."""
        item = self.get_selected_item()
        if not item or isinstance(item, str): return
        job = item

        if job["_type"] != "running": return
        
        self.action_msg = f"Pausing {job['id']}..."
        self.msg_clear_time = time.time() + 2
        
        with locked_queue() as queue:
            pid = job.get("pid")
            if pid:
                try: os.killpg(pid, signal.SIGKILL)
                except: pass
            
            queue["running"] = [j for j in queue["running"] if j["id"] != job["id"]]
            
            new_job = {
                "id": generate_job_id(),
                "cmd": job["cmd"],
                "gpus": job.get("gpus", 1),
                "added": datetime.now().isoformat(),
                "priority": 3,
                "paused_from": job["id"],
                "cwd": job.get("cwd")
            }
            queue["pending"].insert(0, new_job)

    def action_duplicate(self):
        """Duplicate selected job and edit."""
        item = self.get_selected_item()
        if not item or isinstance(item, str): return
        job = item
        
        self.action_msg = f"Duplicating {job['id']}..."
        self.msg_clear_time = time.time() + 2
        
        new_job_obj = None
        
        with locked_queue() as queue:
             new_id = generate_job_id()
             new_job = {
                "id": new_id,
                "cmd": job["cmd"],
                "gpus": job.get("gpus", 1),
                "added": datetime.now().isoformat(),
                "priority": job.get("priority", 1),
                "cwd": job.get("cwd")
             }
             
             inserted = False
             if job["_type"] == "pending":
                 for i, j in enumerate(queue["pending"]):
                     if j["id"] == job["id"]:
                         queue["pending"].insert(i+1, new_job)
                         inserted = True
                         break
             
             if not inserted:
                 queue["pending"].append(new_job)
                 
             new_job_obj = new_job
             new_job_obj["_type"] = "pending"

        self.action_msg = f"Duplicated as {new_job_obj['id']}"
        if new_job_obj:
            self.action_edit(target_job=new_job_obj)

    def action_edit(self, target_job=None):
        """Edit command for PENDING jobs."""
        if target_job:
            job = target_job
        else:
            item = self.get_selected_item()
            if not item or isinstance(item, str): return
            job = item
        
        if job.get("_type") != "pending":
            self.action_msg = "Only PENDING jobs can be edited"
            self.msg_clear_time = time.time() + 2
            return

        import curses
        curses.noecho()
        curses.curs_set(1)
        h, w = self.stdscr.getmaxyx()
        
        self.stdscr.move(h-2, 0)
        self.stdscr.clrtoeol()
        
        prompt = "Edit: "
        self.stdscr.addstr(h-2, 0, prompt, curses.color_pair(3) | curses.A_BOLD)
        
        self.stdscr.refresh()
        self.stdscr.nodelay(False)
        
        buffer = list(job["cmd"])
        cursor_x = len(buffer)
        
        start_col = len(prompt)
        max_width = w - start_col - 1
        view_offset = 0
        
        while True:
            if cursor_x < view_offset:
                view_offset = cursor_x
            elif cursor_x >= view_offset + max_width:
                view_offset = cursor_x - max_width + 1
            
            display_str = "".join(buffer[view_offset : view_offset + max_width])
            
            self.stdscr.move(h-2, start_col)
            self.stdscr.clrtoeol()
            self.stdscr.addstr(h-2, start_col, display_str)
            
            visual_cursor_x = start_col + (cursor_x - view_offset)
            self.stdscr.move(h-2, visual_cursor_x)
            
            ch = self.stdscr.getch()
            
            if ch == 10: # Enter
                break
            elif ch == 27: # Esc
                buffer = None
                break
            elif ch in (curses.KEY_BACKSPACE, 127, 8):
                if cursor_x > 0:
                    buffer.pop(cursor_x - 1)
                    cursor_x -= 1
            elif ch == curses.KEY_DC: # Delete
                if cursor_x < len(buffer):
                    buffer.pop(cursor_x)
            elif ch == curses.KEY_LEFT:
                if cursor_x > 0: cursor_x -= 1
            elif ch == curses.KEY_RIGHT:
                if cursor_x < len(buffer): cursor_x += 1
            elif ch in (curses.KEY_HOME, 1):
                cursor_x = 0
            elif ch in (curses.KEY_END, 5):
                cursor_x = len(buffer)
            elif ch == 21: # Ctrl+U
                buffer = []
                cursor_x = 0
            elif ch == 23: # Ctrl+W
                 while cursor_x > 0 and buffer[cursor_x-1] == ' ':
                     buffer.pop(cursor_x-1)
                     cursor_x -= 1
                 while cursor_x > 0 and buffer[cursor_x-1] != ' ':
                     buffer.pop(cursor_x-1)
                     cursor_x -= 1
            elif 32 <= ch <= 126:
                buffer.insert(cursor_x, chr(ch))
                cursor_x += 1
        
        self.stdscr.nodelay(True)
        curses.noecho()
        curses.curs_set(0)
        
        if buffer is None:
            self.action_msg = "Edit cancelled"
        else:
            new_cmd = "".join(buffer).strip()
            if new_cmd:
                with locked_queue() as queue:
                    for j in queue["pending"]:
                        if j["id"] == job["id"]:
                            j["cmd"] = new_cmd
                            break
                self.action_msg = f"Updated job {job['id']}"
            else:
                 self.action_msg = "Empty command ignored"
        
        self.msg_clear_time = time.time() + 2

    def action_view_logs(self):
        """Open logs in less."""
        item = self.get_selected_item()
        if not item or isinstance(item, str): return
        job = item
        
        log_file = LOG_DIR / f"{job['id']}.log"
        if not log_file.exists(): return
        
        import curses
        curses.endwin()
        os.system(f"less +F {log_file}")
        self.stdscr.refresh()

    def draw(self):
        import curses
        self.stdscr.erase()
        h, w = self.stdscr.getmaxyx()
        
        with self.lock:
            if not self.data_ready:
                self.stdscr.addstr(0, 0, "Loading data...", curses.color_pair(4))
                self.stdscr.refresh()
                return

            data = self.data
            gpus = data.get("gpus", [])
            queue_data = data.get("queue", {"running": [], "pending": [], "completed": []})
            
        row = 0
        total_gpus = len(gpus)
        min_free = data.get("min_free", 2)
        our_usage = sum(j.get("gpus", 1) for j in queue_data["running"])
        
        our_assigned_indices = set()
        for j in queue_data["running"]:
            for idx in j.get("assigned_gpus", []):
                our_assigned_indices.add(idx)
        
        others_busy_count = 0
        for g in gpus:
            if not g["free"] and g["index"] not in our_assigned_indices:
                others_busy_count += 1

        try:
            self.stdscr.addstr(row, 0, f"GPU Queue Monitor | Poll: {POLL_INTERVAL}s", curses.color_pair(4) | curses.A_BOLD)
        except: pass
        row += 1
        
        excluded_count = len(data.get("excluded", []))
        effective_min_free = max(0, min_free - excluded_count)
        quota = total_gpus - effective_min_free
        
        header_stats = [
            f"Total: {total_gpus}",
            f"Quota: {quota}",
            f"Usage: {our_usage}",
            f"Others: {others_busy_count}",
            f"Rsvd: {min_free}",
            f"Excl: {excluded_count}",
        ]
        stats_str = " | ".join(header_stats)
        try:
            self.stdscr.addstr(row, 0, stats_str[:w-1], curses.color_pair(4))
        except: pass
        
        if self.action_msg:
            if time.time() > self.msg_clear_time:
                self.action_msg = ""
            else:
                try:
                    self.stdscr.addstr(row, w - len(self.action_msg) - 2, self.action_msg, curses.color_pair(3) | curses.A_BOLD)
                except: pass
        
        row += 2
        
        list_start_row = row
        detail_h = 6
        footer_h = 1
        
        list_view_h = h - list_start_row - detail_h - footer_h
        if list_view_h < 1: list_view_h = 1 
        
        if self.selected_idx < self.scroll_offset:
            self.scroll_offset = self.selected_idx
        elif self.selected_idx >= self.scroll_offset + list_view_h:
            self.scroll_offset = self.selected_idx - list_view_h + 1
            
        visible_rows = self.ui_rows[self.scroll_offset : self.scroll_offset + list_view_h]
        
        col_header = f" {'ID':<8} | {'PID':<8} | {'GPUs':<4} | {'STATUS':<20} | {'TIME':<10} | {'COMMAND'}"
        self.stdscr.addstr(list_start_row, 0, col_header[:w-1], curses.A_UNDERLINE)
        list_start_row += 1
        list_view_h -= 1 

        selected_item = None
        
        for i, item in enumerate(visible_rows):
            actual_idx = self.scroll_offset + i
            is_selected = (actual_idx == self.selected_idx)
            if is_selected: selected_item = item
            
            style = curses.A_NORMAL
            if is_selected:
                style = curses.A_REVERSE
            
            if isinstance(item, str):
                try:
                    self.stdscr.addstr(list_start_row + i, 0, item[:w-1], curses.color_pair(4) | curses.A_BOLD)
                except: pass
                continue

            job = item    
            jid = job["id"]
            jtype = job["_type"]
            gpus_req = job.get("gpus", 1)
            
            color = curses.color_pair(0)
            status_str = jtype.upper()
            time_str = "--"
            pid_str = str(job.get("pid", ""))

            if jtype == "running":
                color = curses.color_pair(3)
                alloc = ",".join(map(str, job.get("assigned_gpus", [])))
                status_str = f"RUNNING ({alloc})"
                time_str = self.format_duration(job.get("started", ""))
            elif jtype == "pending":
                pid_str = ""
                prio = job.get("priority", 1)
                prio_s = ["LOW", "MED", "HIGH", "URGENT"][min(prio, 3)]
                status_str = f"PENDING [{prio_s}]"
                time_str = self.format_duration(job.get("added", ""))
            elif jtype == "completed":
                st = job.get("status", "unknown")
                if st == "success": color = curses.color_pair(1)
                elif st == "cancelled": color = curses.color_pair(3)
                else: color = curses.color_pair(2)
                status_str = st.upper()
                time_str = self.format_duration(job.get("started", ""), job.get("ended", ""))
                
            line = f" {jid:<8} | {pid_str:<8} | {gpus_req:<4} | {status_str:<20} | {time_str:<10} | {job['cmd']}"
            if len(line) > w: line = line[:w-1]
            try:
                self.stdscr.addstr(list_start_row + i, 0, line, color | style)
            except: pass

        detail_y = h - detail_h - 1

        try:
            self.stdscr.addstr(detail_y, 0, "─" * w, curses.color_pair(4))
            if selected_item and isinstance(selected_item, dict):
                job = selected_item
                jid = job['id']
                pid = job.get('pid', 'N/A')
                st_info = f" {job['_type'].upper()} INFO: {jid} | PID: {pid} "
                self.stdscr.addstr(detail_y, 2, st_info, curses.color_pair(4) | curses.A_BOLD)
                
                cmd_full = f"Command: {job['cmd']}"
                wrapped_lines = textwrap.wrap(cmd_full, width=w-2)
                for j, line_text in enumerate(wrapped_lines[:3]):
                    if j == 0 and line_text.startswith("Command: "):
                        self.stdscr.addstr(detail_y + 1 + j, 1, "Command: ", curses.A_DIM)
                        self.stdscr.addstr(detail_y + 1 + j, 10, line_text[9:w-2], curses.color_pair(4) | curses.A_BOLD)
                    else:
                        self.stdscr.addstr(detail_y + 1 + j, 1, line_text[:w-2], curses.color_pair(4) | curses.A_BOLD)
                
                meta = []
                if 'added' in job: meta.append(f"Added: {job['added']}")
                if 'started' in job: meta.append(f"Started: {job['started']}")
                if 'ended' in job: meta.append(f"Ended: {job['ended']}")
                if 'cwd' in job: meta.append(f"CWD: {job['cwd']}")
                
                meta_str = " | ".join(meta)
                self.stdscr.addstr(detail_y + 4, 1, meta_str[:w-2], curses.A_DIM)
            else:
                self.stdscr.addstr(detail_y + 1, 2, "Select a job for details.")
        except: pass

        help_str = "Q:Quit"
        if selected_item and isinstance(selected_item, dict):
            jtype = selected_item.get("_type")
            if jtype == "pending":
                help_str = "E:Edit  D:Dup  [/]:Prio  +/-:GPUs  c:Cancel  Q:Quit"
            elif jtype == "running":
                help_str = "P:Pause  D:Dup  c:Cancel  l/Ent:Log  Q:Quit"
            elif jtype == "completed":
                help_str = "R:Retry  D:Dup  l/Ent:Log  Q:Quit"
        else:
             help_str = "j/k:Nav  Q:Quit"
             
        try:
            self.stdscr.addstr(h-1, 0, help_str[:w-1], curses.color_pair(4) | curses.A_REVERSE)
        except: pass

    def adjust_priority(self, delta):
        item = self.get_selected_item()
        if not item or isinstance(item, str): return
        job = item
        
        if job["_type"] != "pending": return
        
        current = job.get("priority", 1)
        new_prio = max(0, min(3, current + delta))
        
        if new_prio != current:
            with locked_queue() as queue:
                 for j in queue["pending"]:
                    if j["id"] == job["id"]:
                        j["priority"] = new_prio
                        break
            
            prio_s = ["LOW", "MED", "HIGH", "URGENT"][new_prio]
            self.action_msg = f"Priority -> {prio_s}"
            self.msg_clear_time = time.time() + 1.5

    def adjust_gpus(self, delta):
        item = self.get_selected_item()
        if not item or isinstance(item, str): return
        job = item
        
        if job["_type"] != "pending": return
        
        current = job.get("gpus", 1)
        new_gpus = max(1, min(8, current + delta))
        
        if new_gpus != current:
            with locked_queue() as queue:
                 for j in queue["pending"]:
                    if j["id"] == job["id"]:
                        j["gpus"] = new_gpus
                        break
            
            self.action_msg = f"GPUs -> {new_gpus}"
            self.msg_clear_time = time.time() + 1.5

    def main(self, stdscr):
        import curses
        self.stdscr = stdscr
        
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)
        curses.init_pair(2, curses.COLOR_RED, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
        curses.init_pair(4, curses.COLOR_CYAN, -1)
        
        curses.curs_set(0)
        self.stdscr.nodelay(True)
        
        self.start()
        
        try:
            while True:
                self.draw() # Draws from shared state
                
                ch = self.stdscr.getch()
                
                if ch == -1:
                    time.sleep(0.05)
                    continue
                    
                if ch == ord('q'): break
                
                if ch == curses.KEY_UP or ch == ord('k'):
                    if self.selected_idx > 0: 
                        self.selected_idx -= 1
                        if isinstance(self.get_selected_item(), str) and self.selected_idx > 0:
                            self.selected_idx -= 1
                        
                elif ch == curses.KEY_DOWN or ch == ord('j'):
                     with self.lock:
                         curr_len = len(self.ui_rows)
                     if self.selected_idx < curr_len - 1: 
                        self.selected_idx += 1
                        if isinstance(self.get_selected_item(), str) and self.selected_idx < curr_len - 1:
                            self.selected_idx += 1
                
                elif ch in [ord(' '), 10, ord('l')]:
                    self.action_view_logs()
                
                elif ch in [ord('c'), curses.KEY_DC]:
                    self.action_cancel()
                elif ch == ord('d'):
                    self.action_duplicate()
                elif ch == ord('p'):
                    self.action_pause()
                elif ch == ord('r'):
                    self.action_retry()
                elif ch == ord('e'):
                    self.action_edit()
                elif ch == ord(']'):
                    self.adjust_priority(1)
                elif ch == ord('['):
                    self.adjust_priority(-1)
                elif ch == ord('+') or ch == ord('='): 
                    self.adjust_gpus(1)
                elif ch == ord('-') or ch == ord('_'):
                    self.adjust_gpus(-1)

        finally:
            self.stop()


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
    add_parser.add_argument("--gpus", "-g", type=int, default=2, help="Number of GPUs required")
    add_parser.add_argument("--priority", "-p", choices=["low", "medium", "high"], default="medium", help="Job priority")
    add_parser.add_argument("--front", "-f", action="store_true", help="Add to front of queue (Urgent priority)")
    add_parser.set_defaults(func=cmd_add)

    # serve
    serve_parser = subparsers.add_parser("serve", help="Run the queue scheduler (foreground)")
    serve_parser.add_argument("--min-free", type=int, default=2, help="Number of GPUs to always keep free")
    serve_parser.add_argument("--exclude-gpus", type=str, default="", help="Comma-separated list of GPU indices to ignore (e.g. '0,1')")
    serve_parser.set_defaults(func=cmd_serve)

    # start
    start_parser = subparsers.add_parser("start", help="Start the daemon (background)")
    start_parser.add_argument("--min-free", type=int, default=2, help="Number of GPUs to always keep free")
    start_parser.add_argument("--exclude-gpus", type=str, default="", help="Comma-separated list of GPU indices to ignore")
    start_parser.set_defaults(func=cmd_start)
    
    # stop
    stop_parser = subparsers.add_parser("stop", help="Stop the daemon")
    stop_parser.set_defaults(func=cmd_stop)

    # status
    status_parser = subparsers.add_parser("status", help="Check if daemon is running")
    status_parser.set_defaults(func=cmd_status)

    # cancel
    cancel_parser = subparsers.add_parser("cancel", help="Cancel a pending job")
    cancel_parser.add_argument("job_id", help="Job ID to cancel")
    cancel_parser.set_defaults(func=cmd_cancel)

    # logs
    logs_parser = subparsers.add_parser("logs", help="Show job logs")
    logs_parser.add_argument("job_id", help="Job ID")
    logs_parser.add_argument("--lines", "-n", type=int, default=50, help="Number of lines")
    logs_parser.set_defaults(func=cmd_logs)

    # clear
    clear_parser = subparsers.add_parser("clear", help="Clear completed jobs")
    clear_parser.set_defaults(func=cmd_clear)

    # retry
    retry_parser = subparsers.add_parser("retry", help="Re-queue a completed job")
    retry_parser.add_argument("job_id", help="Job ID to retry")
    retry_parser.add_argument("--front", "-f", action="store_true", help="Add to front of queue")
    retry_parser.set_defaults(func=cmd_retry)

    # pause
    pause_parser = subparsers.add_parser("pause", help="Pause a running job (re-queue at front)")
    pause_parser.add_argument("job_id", help="Job ID to pause")
    pause_parser.set_defaults(func=cmd_pause)

    # watch
    watch_parser = subparsers.add_parser("watch", help="Watch queue status continuously")
    watch_parser.add_argument("--interval", "-n", type=float, default=2.0, help="Update interval in seconds")
    watch_parser.set_defaults(func=cmd_watch)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
