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

from gpu_queue.cli import build_parser, cmd_watch, main
from gpu_queue.commands import (
    cmd_add,
    cmd_cancel,
    cmd_clear,
    cmd_logs,
    cmd_pause,
    cmd_retry,
    cmd_serve,
    cmd_start,
    cmd_stop,
)
from gpu_queue.gpu import get_available_gpu_indices, get_free_gpus
from gpu_queue.ids import generate_job_id
from gpu_queue.paths import (
    DAEMON_LOG,
    LOCK_FILE,
    LOG_DIR,
    MIN_FREE_GPUS,
    PID_FILE,
    POLL_INTERVAL,
    QUEUE_DIR,
    QUEUE_FILE,
    SERVER_PORT,
    get_server_url,
)
from gpu_queue.scheduler import (
    cleanup_dead_jobs,
    daemon_loop,
    is_daemon_running,
    run_job,
)
from gpu_queue.storage import load_queue_raw, save_queue, save_queue_raw
from gpu_queue.tui.app import GPUQueueTUI, Window, get_status_data, get_terminal_width

__all__ = [
    "DAEMON_LOG",
    "GPUQueueTUI",
    "LOCK_FILE",
    "LOG_DIR",
    "MIN_FREE_GPUS",
    "PID_FILE",
    "POLL_INTERVAL",
    "QUEUE_DIR",
    "QUEUE_FILE",
    "SERVER_PORT",
    "Window",
    "build_parser",
    "cleanup_dead_jobs",
    "cmd_add",
    "cmd_cancel",
    "cmd_clear",
    "cmd_logs",
    "cmd_pause",
    "cmd_retry",
    "cmd_serve",
    "cmd_start",
    "cmd_stop",
    "cmd_watch",
    "daemon_loop",
    "generate_job_id",
    "get_available_gpu_indices",
    "get_free_gpus",
    "get_server_url",
    "get_status_data",
    "get_terminal_width",
    "is_daemon_running",
    "load_queue_raw",
    "main",
    "run_job",
    "save_queue",
    "save_queue_raw",
]


if __name__ == "__main__":
    main()
