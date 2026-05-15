"""Command-line interface for gpu-queue."""

import argparse

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
from gpu_queue.tui.app import GPUQueueTUI


def cmd_watch(args: argparse.Namespace) -> None:
    """Interactive TUI for queue monitoring."""
    import curses

    tui = GPUQueueTUI(args.interval)
    try:
        curses.wrapper(tui.main)
    except KeyboardInterrupt:
        pass
    print("Exited TUI.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GPU Job Queue Scheduler")
    subparsers = parser.add_subparsers(dest="command", required=True)

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

    start_parser = subparsers.add_parser(
        "start", help="Start the queue scheduler (background)"
    )
    start_parser.add_argument(
        "--min-free",
        type=int,
        default=2,
        help="Number of physically idle GPUs to always keep free",
    )
    start_parser.add_argument(
        "--max-use",
        type=int,
        default=None,
        help="Maximum GPUs this queue may occupy at once",
    )
    start_parser.set_defaults(func=cmd_start)

    stop_parser = subparsers.add_parser("stop", help="Stop the background scheduler")
    stop_parser.set_defaults(func=cmd_stop)

    serve_parser = subparsers.add_parser(
        "serve", help="Run the queue scheduler (foreground)"
    )
    serve_parser.add_argument(
        "--min-free",
        type=int,
        default=2,
        help="Number of physically idle GPUs to always keep free",
    )
    serve_parser.add_argument(
        "--max-use",
        type=int,
        default=None,
        help="Maximum GPUs this queue may occupy at once",
    )
    serve_parser.add_argument(
        "--exclude-gpus",
        type=str,
        default="",
        help="Comma-separated list of GPU indices to ignore (e.g. '0,1')",
    )
    serve_parser.set_defaults(func=cmd_serve)

    cancel_parser = subparsers.add_parser("cancel", help="Cancel a pending job")
    cancel_parser.add_argument("job_id", help="Job ID to cancel")
    cancel_parser.set_defaults(func=cmd_cancel)

    logs_parser = subparsers.add_parser("logs", help="Show job logs")
    logs_parser.add_argument("job_id", help="Job ID")
    logs_parser.add_argument(
        "--lines", "-n", type=int, default=50, help="Number of lines"
    )
    logs_parser.set_defaults(func=cmd_logs)

    clear_parser = subparsers.add_parser("clear", help="Clear completed jobs")
    clear_parser.set_defaults(func=cmd_clear)

    retry_parser = subparsers.add_parser("retry", help="Re-queue a completed job")
    retry_parser.add_argument("job_id", help="Job ID to retry")
    retry_parser.add_argument(
        "--front", "-f", action="store_true", help="Add to front of queue"
    )
    retry_parser.set_defaults(func=cmd_retry)

    pause_parser = subparsers.add_parser(
        "pause", help="Pause a running job (re-queue at front)"
    )
    pause_parser.add_argument("job_id", help="Job ID to pause")
    pause_parser.set_defaults(func=cmd_pause)

    watch_parser = subparsers.add_parser(
        "watch", help="Watch queue status continuously"
    )
    watch_parser.add_argument(
        "--interval", "-n", type=float, default=2.0, help="Update interval in seconds"
    )
    watch_parser.set_defaults(func=cmd_watch)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
