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
import copy
import curses
import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, cast

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
from gpu_queue.queue_state import (
    cancel_staged_job,
    insert_staged_job,
    make_staged_job,
    move_pending_job,
    move_pending_job_to_staging,
    move_pending_jobs,
    send_staged_job_to_pending,
    stage_completed_retry,
)
from gpu_queue.scheduler import (
    cleanup_dead_jobs,
    daemon_loop,
    is_daemon_running,
    run_job,
)
from gpu_queue.storage import (
    load_queue,
    load_queue_raw,
    locked_queue,
    save_queue,
    save_queue_raw,
)

__all__ = [
    "DAEMON_LOG",
    "LOCK_FILE",
    "LOG_DIR",
    "MIN_FREE_GPUS",
    "PID_FILE",
    "QUEUE_DIR",
    "QUEUE_FILE",
    "SERVER_PORT",
    "cleanup_dead_jobs",
    "daemon_loop",
    "get_available_gpu_indices",
    "get_free_gpus",
    "get_server_url",
    "is_daemon_running",
    "load_queue_raw",
    "run_job",
    "save_queue",
    "save_queue_raw",
]

# Try importing requests. Env should have it. Fallback if needed.
try:
    import requests  # type: ignore
except ImportError:
    requests = None

CMD_FIELD_GHOST = "<press enter to edit command>"

# Sparkline for GPU util (U+2581..U+2588).
# Buffer >= max spark columns so full width scrolls.
_BLOCK_SPARK_CHARS = "▁▂▃▄▅▆▇█"
GPU_UTIL_HISTORY_MAX_SAMPLES = 512
# GPU STATUS table: fixed widths so header and rows align (content inside borders)
GPU_COL_IDX_W = 4
GPU_COL_UTIL_W = 4  # "100%"
GPU_COL_MEM_W = 14
GPU_MIN_PROC_W = 8


def get_terminal_width() -> int:
    """Get the current terminal width."""
    return shutil.get_terminal_size((80, 20)).columns


def sparkline_trailing(util_series: list[int], width: int) -> str:
    """Latest `width` samples as block chars, right-aligned (scrolls as time passes)."""
    if width <= 0:
        return ""
    if not util_series:
        return " " * width
    tail = util_series[-width:]
    parts: list[str] = []
    for u in tail:
        u = max(0, min(100, int(u)))
        bi = min(7, u * 7 // 100)
        parts.append(_BLOCK_SPARK_CHARS[bi])
    s = "".join(parts)
    return (" " * (width - len(s))) + s


def _gpu_status_column_widths(inner: int) -> tuple[int, int, int]:
    """Return (prefix_w, hist_w, proc_w) for one full-width status row."""
    prefix_w = GPU_COL_IDX_W + 1 + GPU_COL_UTIL_W + 1 + GPU_COL_MEM_W
    slack = inner - prefix_w
    if slack <= 0:
        return prefix_w, 0, 0
    if slack == 1:
        return prefix_w, 0, 1
    pair = slack - 2
    if pair < GPU_MIN_PROC_W:
        return prefix_w, 0, slack - 1
    proc_w = max(GPU_MIN_PROC_W, pair // 3)
    hist_w = pair - proc_w
    return prefix_w, hist_w, proc_w


def _fit_text_field(s: str, max_w: int) -> str:
    if max_w <= 0:
        return ""
    if len(s) <= max_w:
        return s.ljust(max_w)
    if max_w <= 3:
        return s[:max_w]
    return s[: max_w - 3] + "..."


def _format_gpu_history_span_seconds(total_sec: int) -> str:
    total_sec = max(0, int(total_sec))
    if total_sec < 60:
        return f"{total_sec}s"
    if total_sec < 3600:
        m, s = divmod(total_sec, 60)
        return f"{m}m" if s == 0 else f"{m}m{s}s"
    h, rem = divmod(total_sec, 3600)
    m, s = divmod(rem, 60)
    if m == 0 and s == 0:
        return f"{h}h"
    if s == 0:
        return f"{h}h{m}m"
    return f"{h}h{m}m{s}s"


def _gpu_history_header_label(hist_w: int, sample_interval_sec: float) -> str:
    """Visible window = one sample per column at daemon poll cadence."""
    if hist_w <= 0:
        return ""
    span_sec = int(round(hist_w * sample_interval_sec))
    return f"HISTORY {_format_gpu_history_span_seconds(span_sec)}"


def shorten_command(cmd: str, max_len: int) -> str:
    """Shorten a command string to max_len by removing the middle part."""
    if len(cmd) <= max_len:
        return cmd

    # Calculate lengths to keep
    head_len = (max_len - 3) // 2
    tail_len = max_len - 3 - head_len

    return f"{cmd[:head_len]}...{cmd[-tail_len:]}"


# cmd_status removed - functionality merged into watch TUI


def get_status_data():
    """Gather all status data for the queue and GPUs."""
    cleanup_dead_jobs()
    queue = load_queue()
    gpus = get_free_gpus()

    # Get config (min_free, max_use, and excluded)
    min_free = 2
    max_use = None
    excluded = set()
    config_file = QUEUE_DIR / "config.json"
    if config_file.exists():
        try:
            cfg = json.loads(config_file.read_text())
            min_free = cfg.get("min_free_gpus", 2)
            max_use = cfg.get("max_use_gpus")
            excluded = set(cfg.get("excluded_gpus", []))
        except Exception:
            pass

    # Filter GPUs (hiding excluded ones entirely from the monitor view)
    gpus = [g for g in gpus if g["index"] not in excluded]

    return {
        "queue": queue,
        "gpus": gpus,
        "min_free": min_free,
        "max_use": max_use,
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
        self.height: Optional[int] = None
        self.collapsed = False

    def update_items(self, items):
        self.items = items
        # Clamp selection
        if self.selected_idx >= len(self.items):
            self.selected_idx = max(0, len(self.items) - 1)
        self.clamp_scroll()

    def visible_item_count(self, h=None):
        """Return how many list rows are visible at this window height."""
        effective_h = h if h is not None else getattr(self, "height", None)
        if effective_h is None:
            effective_h = 10

        visible_h = max(1, effective_h - 2)
        if self.key in ["running", "staging", "pending", "completed"]:
            visible_h = max(1, visible_h - 1)
        return visible_h

    def clamp_scroll(self, h=None):
        """Keep the scroll offset valid for the current item count and height."""
        if not self.items:
            self.selected_idx = 0
            self.scroll_offset = 0
            return

        self.selected_idx = max(0, min(len(self.items) - 1, self.selected_idx))
        visible_h = self.visible_item_count(h)
        max_offset = max(0, len(self.items) - visible_h)
        self.scroll_offset = max(0, min(max_offset, self.scroll_offset))

    def ensure_selected_visible(self, h=None):
        """Adjust scroll offset so the selected row remains on screen."""
        self.clamp_scroll(h)
        if not self.items:
            return

        visible_h = self.visible_item_count(h)
        if self.selected_idx < self.scroll_offset:
            self.scroll_offset = self.selected_idx
        elif self.selected_idx >= self.scroll_offset + visible_h:
            self.scroll_offset = self.selected_idx - visible_h + 1
        self.clamp_scroll(h)

    def scroll(self, delta, h=None):
        """Scroll selection by delta."""
        if not self.items:
            return

        new_idx = self.selected_idx + delta
        self.selected_idx = max(0, min(len(self.items) - 1, new_idx))
        self.ensure_selected_visible(h)

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
        self.data = {"staging": [], "running": [], "pending": [], "completed": []}
        self.data = {"staging": [], "running": [], "pending": [], "completed": []}
        self.gpu_status = []
        self.min_free = 2
        self.max_use: Optional[int] = None
        self.excluded = []
        self.server_status = "UNKNOWN"
        self.last_updated = 0
        self.action_msg = ""
        self.msg_clear_time = 0
        self.action_msg = ""
        self.msg_clear_time = 0
        self.modal: Optional[Dict[str, Any]] = None  # { type, title, text, val... }
        self._gpu_util_history: dict[int, list[int]] = {}
        self._last_util_status_ts: Optional[str] = None

        # Windows
        self.windows = [
            Window("RUNNING", "running", 0.2),
            Window("PENDING", "pending", 0.2),
            Window("STAGING", "staging", 0.2),
            Window("COMPLETED", "completed", 0.2),
            Window("GPU STATUS", "gpu_status", 0.2),
            Window("SELECTED JOB", "job_details", 0.2),
        ]
        self.active_win_idx = 0  # Index in self.windows
        self.mode = "NAV"  # "NAV" (select window) or "ACTION" (interact with window)
        self.has_selected_job_context = False
        self.selected_job_ids: dict[str, set[str]] = {
            "running": set(),
            "pending": set(),
            "staging": set(),
            "completed": set(),
        }
        self.select_mode_active = False
        self.select_mode_window_key: Optional[str] = None
        self.select_mode_anchor_idx: Optional[int] = None

        # Log view state
        self.viewing_logs = False
        self.log_job_id = None
        self.log_content = []
        self.log_scroll = 0

        # Edit Mode State
        self.edit_mode_active = False
        self.edit_job = None
        self.edit_field_idx = 0  # 0: GPUs, 1: Command
        self.edit_is_new = False

        # Dynamic column widths for tables
        self.col_widths = {
            "running": {"id": 8, "pid": 6, "gpus": 4, "elapsed": 7},
            "staging": {"id": 8, "gpus": 4, "waiting": 7},
            "pending": {"id": 8, "gpus": 4, "waiting": 7},
            "completed": {"id": 8, "runtime": 7, "ago": 7},
        }

    def _calc_col_widths(self):
        """Calculate column widths based on current data for all tables."""
        try:

            def calc_queue_widths(items: list[dict[str, Any]]) -> dict[str, int]:
                widths = {"id": 3, "gpus": 4, "waiting": 7}
                for job in items:
                    jid = job.get("id", "")[:8]
                    widths["id"] = max(widths["id"], len(jid))
                    widths["gpus"] = max(widths["gpus"], len(str(job.get("gpus", 1))))
                    add_dt = self._parse_iso(job.get("added"))
                    if add_dt:
                        waiting = self._fmt_delta(datetime.now() - add_dt)
                        widths["waiting"] = max(widths["waiting"], len(waiting))
                for key in widths:
                    widths[key] += 1
                return widths

            # Running table
            running_w = {"id": 3, "pid": 3, "gpus": 4, "elapsed": 7}
            for job in self.data.get("running", []):
                jid = job.get("id", "")[:8]
                running_w["id"] = max(running_w["id"], len(jid))
                running_w["pid"] = max(running_w["pid"], len(str(job.get("pid", ""))))
                gpus = ",".join(map(str, job.get("assigned_gpus", [])))
                running_w["gpus"] = max(running_w["gpus"], len(gpus) if gpus else 1)
                start_dt = self._parse_iso(job.get("started"))
                if start_dt:
                    elapsed = self._fmt_delta(datetime.now() - start_dt)
                    running_w["elapsed"] = max(running_w["elapsed"], len(elapsed))

            # Staging + Pending tables
            staging_w = calc_queue_widths(self.data.get("staging", []))
            pending_w = calc_queue_widths(self.data.get("pending", []))

            # Completed table
            completed_w = {"id": 3, "runtime": 7, "ago": 7}
            for job in self.data.get("completed", []):
                jid = job.get("id", "")[:8]
                completed_w["id"] = max(completed_w["id"], len(jid))
                start_dt = self._parse_iso(job.get("started"))
                end_dt = self._parse_iso(job.get("ended"))
                if start_dt and end_dt:
                    run_s = self._fmt_delta(end_dt - start_dt)
                    completed_w["runtime"] = max(completed_w["runtime"], len(run_s))
                if end_dt:
                    ago_s = self._fmt_delta(datetime.now() - end_dt)
                    completed_w["ago"] = max(completed_w["ago"], len(ago_s))

            # Add padding
            for key in running_w:
                running_w[key] += 1
            for key in completed_w:
                completed_w[key] += 1

            self.col_widths = {
                "running": running_w,
                "staging": staging_w,
                "pending": pending_w,
                "completed": completed_w,
            }
        except Exception:
            pass  # Keep existing widths on error

    def _set_data_snapshot(self, queue: dict[str, list]):
        """Replace local queue data from a freshly mutated queue snapshot."""
        self.data = copy.deepcopy(queue)
        self.data["completed"].sort(key=lambda x: x.get("ended", ""), reverse=True)
        self.last_updated = time.time()
        self._sync_windows_from_data()
        self._calc_col_widths()

    def _sync_windows_from_data(self):
        for w in self.windows:
            items = self.data.get(w.key, [])
            type_label = w.key
            for item in items:
                item["_type"] = type_label
            w.update_items(items)
            if w.key in self.selected_job_ids:
                live_ids = {str(item.get("id")) for item in items}
                self.selected_job_ids[w.key].intersection_update(live_ids)

        if self.edit_mode_active and self.edit_job is not None:
            edit_id = self.edit_job.get("id")
            for w in self.windows:
                if w.key == "staging":
                    for idx, item in enumerate(w.items):
                        if item.get("id") == edit_id:
                            w.selected_idx = idx
                            break
                    break

    def _selected_job(self):
        win = self.windows[self.active_win_idx]
        if win.key in ["running", "pending", "staging", "completed"]:
            return win.get_selected()
        return getattr(self, "last_selected_job", None)

    def _selected_ids_for_window(self, win) -> list[str]:
        """Return checked row IDs in display order for a queue window."""
        selected = self.selected_job_ids.get(win.key, set())
        if not selected:
            return []
        return [
            str(item["id"])
            for item in win.items
            if item.get("id") is not None and str(item["id"]) in selected
        ]

    def _action_ids_for_window(self, win) -> list[str]:
        checked = self._selected_ids_for_window(win)
        if checked:
            return checked
        job = win.get_selected()
        if not job:
            return []
        return [str(job["id"])]

    def _select_current_row(self, win) -> None:
        if win.key not in self.selected_job_ids:
            return
        job = win.get_selected()
        if not job:
            return
        self.selected_job_ids[win.key].add(str(job["id"]))

    def _select_range_to_cursor(self, win) -> None:
        if win.key not in self.selected_job_ids or not win.items:
            return
        if self.select_mode_anchor_idx is None:
            self.select_mode_anchor_idx = win.selected_idx
        anchor = max(0, min(len(win.items) - 1, self.select_mode_anchor_idx))
        cursor = max(0, min(len(win.items) - 1, win.selected_idx))
        lo = min(anchor, cursor)
        hi = max(anchor, cursor)
        self.selected_job_ids[win.key] = {
            str(win.items[idx]["id"]) for idx in range(lo, hi + 1)
        }

    def _enter_select_mode(self, win) -> None:
        if win.key not in self.selected_job_ids or not win.get_selected():
            return
        self.selected_job_ids[win.key] = set()
        self.select_mode_active = True
        self.select_mode_window_key = win.key
        self.select_mode_anchor_idx = win.selected_idx
        self._select_range_to_cursor(win)
        self.action_msg = "Select mode"
        self.msg_clear_time = time.time() + 2.0

    def _exit_select_mode(self) -> None:
        self.select_mode_active = False
        self.select_mode_window_key = None
        self.select_mode_anchor_idx = None

    def _has_selected_ids(self) -> bool:
        return any(self.selected_job_ids.values())

    def _clear_all_selected_ids(self) -> None:
        for key in self.selected_job_ids:
            self.selected_job_ids[key].clear()

    def _cancel_select_mode(self) -> None:
        self._clear_all_selected_ids()
        self._exit_select_mode()

    def _select_mode_move(self, win, delta: int, h: Optional[int] = None) -> None:
        if not self.select_mode_active or self.select_mode_window_key != win.key:
            return
        previous_idx = win.selected_idx
        win.scroll(delta, h)
        if win.selected_idx == previous_idx:
            return
        self._select_range_to_cursor(win)
        selected_count = len(self._selected_ids_for_window(win))
        self.action_msg = f"Selected {selected_count} jobs"
        self.msg_clear_time = time.time() + 2.0

    def _clear_selected_ids(self, jids: Sequence[str]) -> None:
        ids = {str(jid) for jid in jids}
        for selected in self.selected_job_ids.values():
            selected.difference_update(ids)
        if self.select_mode_window_key is not None and not self.selected_job_ids.get(
            self.select_mode_window_key
        ):
            self._exit_select_mode()

    def _enter_action_window(self, win):
        self.mode = "ACTION"
        win.collapsed = False
        if win.key in ["running", "pending", "staging", "completed"]:
            self.has_selected_job_context = True

    def _exit_action_window(self, win):
        self.mode = "NAV"
        self.has_selected_job_context = False
        self.last_selected_job = None
        win.scroll_offset = 0
        win.selected_idx = 0

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
                        self._set_data_snapshot(q)

                time.sleep(0.5)  # Fast update
            except Exception:
                # with self.lock: self.action_msg = f"Q Poll Error: {str(e)}"
                time.sleep(1)

    def _append_gpu_util_sample(self, status: dict[str, Any]) -> None:
        ts = status.get("ts")
        if not ts or ts == self._last_util_status_ts:
            return
        self._last_util_status_ts = ts
        for g in status.get("gpus", []):
            idx = g.get("index")
            if idx is None:
                continue
            util = int(g.get("util", 0))
            buf = self._gpu_util_history.setdefault(int(idx), [])
            buf.append(max(0, min(100, util)))
            while len(buf) > GPU_UTIL_HISTORY_MAX_SAMPLES:
                buf.pop(0)

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
                                self.max_use = s.get("max_use")
                                self.excluded = s.get("excluded", [])
                                self.server_status = "DAEMON: ON"
                                self._append_gpu_util_sample(s)
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
        if self.stdscr is None:
            return
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
            # Formt: ID(8) | PID(10) | GPUS(8) | ELAPSED(9) | CMD
            color = curses.color_pair(5)  # Blue

            gpus = ",".join(map(str, job.get("assigned_gpus", [])))
            if not gpus:
                gpus = "?"

            pid = str(job.get("pid", "?"))

            elapsed = "-"
            start_dt = self._parse_iso(job.get("started"))
            if start_dt:
                elapsed = self._fmt_delta(datetime.now() - start_dt)

            # Use dynamic column widths
            cw = self.col_widths["running"]
            prefix = (
                f" {jid:<{cw['id']}} {pid:<{cw['pid']}} "
                f"{gpus:<{cw['gpus']}} {elapsed:<{cw['elapsed']}} "
            )

        elif job["_type"] in ["staging", "pending"]:
            gpus = str(job.get("gpus", 1))
            waiting = "-"
            add_dt = self._parse_iso(job.get("added"))
            if add_dt:
                waiting = self._fmt_delta(datetime.now() - add_dt)

            # Use dynamic column widths
            cw = self.col_widths["staging" if job["_type"] == "staging" else "pending"]
            prefix = (
                f" {jid:<{cw['id']}} {gpus:<{cw['gpus']}} {waiting:<{cw['waiting']}} "
            )

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

            # Use dynamic column widths
            cw = self.col_widths["completed"]
            prefix = (
                f" {jid:<{cw['id']}} {run_s:<{cw['runtime']}} {ago_s:<{cw['ago']}} "
            )

        cmd = job.get("cmd", "") or ""
        avail_cmd = w - len(prefix)
        if len(cmd) > avail_cmd:
            cmd = cmd[: (avail_cmd - 1)] + "…"

        full_line = prefix + cmd

        if is_editing and job["_type"] == "staging":
            cw = self.col_widths["staging"]

            # Field 0: GPUS
            if edit_idx == 0:
                s_val = f"[{gpus}]"
                s_gpus = f"{s_val:<{cw['gpus'] + 1}}"
            else:
                s_gpus = f"{gpus:<{cw['gpus']}} "

            head = f" {jid:<{cw['id']}} " + s_gpus + f"{waiting:<{cw['waiting']}} "
            cmd_avail = max(1, w - len(head))

            cmd_stripped = (job.get("cmd", "") or "").strip()
            if edit_idx == 1:
                disp = cmd_stripped if cmd_stripped else CMD_FIELD_GHOST
                if len(disp) > cmd_avail:
                    disp = disp[: max(0, cmd_avail - 1)] + "…"
                disp = disp.ljust(cmd_avail)[:cmd_avail]
                cmd_attr = curses.A_REVERSE
            else:
                disp = cmd_stripped
                if len(disp) > cmd_avail:
                    disp = disp[: max(0, cmd_avail - 1)] + "…"
                disp = disp.ljust(cmd_avail)[:cmd_avail]
                cmd_attr = curses.A_NORMAL

            return {
                "type": "rich",
                "segments": [
                    (f" {jid:<{cw['id']}} ", curses.A_NORMAL),
                    (s_gpus, curses.A_REVERSE if edit_idx == 0 else curses.A_NORMAL),
                    (f"{waiting:<{cw['waiting']}} ", curses.A_NORMAL),
                    (disp, cmd_attr),
                ],
                "base_color": color,
            }, color

        return full_line, color

    def draw(self):
        if self.stdscr is None:
            return
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
            if self.max_use is not None:
                info_str += f" | Max: {self.max_use}"
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

            win_by_key = {win.key: win for win in self.windows}

            # --- Dynamic Sizing Logic ---
            gpu_h = 1
            if not win_by_key["gpu_status"].collapsed:
                gpu_content_len = len(self.gpu_status) if self.gpu_status else 1
                gpu_h = min(gpu_content_len + 3, max(3, avail_h // 3))
                gpu_h = max(3, gpu_h)

            running_h = 1
            if not win_by_key["running"].collapsed:
                running_items = len(self.data.get("running", []))
                running_content_len = running_items + 1  # +1 for header
                running_h = min(running_content_len + 2, max(3, avail_h // 3))
                running_h = max(3, running_h)

            job_h = 1 if win_by_key["job_details"].collapsed else 8

            queue_keys = ["staging", "pending", "completed"]
            queue_min_h = len(queue_keys)
            nonqueue_min_heights = {
                "running": 1 if win_by_key["running"].collapsed else 3,
                "gpu_status": 1 if win_by_key["gpu_status"].collapsed else 3,
                "job_details": 1 if win_by_key["job_details"].collapsed else 3,
            }
            nonqueue_heights = {
                "running": running_h,
                "gpu_status": gpu_h,
                "job_details": job_h,
            }
            while sum(nonqueue_heights.values()) + queue_min_h > avail_h:
                shrinkable = [
                    key
                    for key, height in nonqueue_heights.items()
                    if height > nonqueue_min_heights[key]
                ]
                if not shrinkable:
                    break
                key = max(shrinkable, key=lambda k: nonqueue_heights[k])
                nonqueue_heights[key] -= 1

            running_h = nonqueue_heights["running"]
            gpu_h = nonqueue_heights["gpu_status"]
            job_h = nonqueue_heights["job_details"]
            remaining_h = max(0, avail_h - sum(nonqueue_heights.values()))
            visible_queue_keys = [k for k in queue_keys if not win_by_key[k].collapsed]
            heights_by_key = {
                "running": running_h,
                "gpu_status": gpu_h,
                "job_details": job_h,
            }
            if visible_queue_keys:
                collapsed_queue_h = sum(
                    1 for k in queue_keys if win_by_key[k].collapsed
                )
                visible_h = max(0, remaining_h - collapsed_queue_h)
                base_h = max(1, visible_h // len(visible_queue_keys))
                extra = max(0, visible_h - (base_h * len(visible_queue_keys)))
                for k in queue_keys:
                    if win_by_key[k].collapsed:
                        heights_by_key[k] = 1
                    else:
                        add = 1 if extra > 0 else 0
                        heights_by_key[k] = base_h + add
                        if extra > 0:
                            extra -= 1
            else:
                for k in queue_keys:
                    heights_by_key[k] = 1

            heights = [heights_by_key[win.key] for win in self.windows]

            current_y = 1
            for i, win in enumerate(self.windows):
                wh = heights[i]
                win.height = wh  # Store actual height for scrolling
                if wh <= 0:
                    continue  # Skip hidden windows
                if not win.collapsed:
                    win.ensure_selected_visible(wh)

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
        if self.stdscr is None:
            return
        if self.modal is None:
            return
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
        if self.stdscr is None:
            return
        """Draw compact nvidia-smi style info."""
        try:
            left = x + 2
            inner = max(0, w - 4)
            prefix_w, hist_w, proc_w = _gpu_status_column_widths(inner)
            hdr_attr = curses.A_BOLD
            row_attr = curses.A_NORMAL

            if not self.gpu_status:
                self.stdscr.addstr(
                    y + 1, left, "No GPU info available", curses.A_NORMAL
                )
                return

            hdr_idx = "IDX".ljust(GPU_COL_IDX_W)[:GPU_COL_IDX_W]
            hdr_util = "UTIL".ljust(GPU_COL_UTIL_W)[:GPU_COL_UTIL_W]
            hdr_mem = "MEM".ljust(GPU_COL_MEM_W)[:GPU_COL_MEM_W]
            prefix_hdr = f"{hdr_idx} {hdr_util} {hdr_mem}"
            if len(prefix_hdr) > inner:
                self.stdscr.addstr(y, left, prefix_hdr[:inner], hdr_attr)
            else:
                self.stdscr.addstr(y, left, prefix_hdr, hdr_attr)
                col = left + prefix_w + 1
                if hist_w > 0:
                    h_hist = _fit_text_field(
                        _gpu_history_header_label(hist_w, POLL_INTERVAL),
                        hist_w,
                    ).ljust(hist_w)[:hist_w]
                    self.stdscr.addstr(y, col, h_hist, hdr_attr)
                    col += hist_w + 1
                h_proc = _fit_text_field("PROCESSES (USER:PID)", proc_w).ljust(proc_w)[
                    :proc_w
                ]
                if proc_w > 0:
                    self.stdscr.addstr(y, col, h_proc, hdr_attr)

            for i, g in enumerate(self.gpu_status[: h - 1]):
                idx = g.get("index", "?")
                used_mb = g.get("used_mb", 0)
                total_mb = g.get("total_mb", 0)
                util = g.get("util", 0)

                used_gb = used_mb / 1024.0
                total_gb = total_mb / 1024.0
                mem_s = f"{used_gb:.1f}/{total_gb:.0f}G"

                line_y = y + 1 + i
                idx_s = (
                    str(int(idx))[:GPU_COL_IDX_W]
                    if isinstance(idx, int)
                    else str(idx)[:GPU_COL_IDX_W]
                ).ljust(GPU_COL_IDX_W)[:GPU_COL_IDX_W]
                u = max(0, min(100, int(util)))
                util_s = f"{u:>3}%".ljust(GPU_COL_UTIL_W)[:GPU_COL_UTIL_W]
                mem_col = mem_s[:GPU_COL_MEM_W].ljust(GPU_COL_MEM_W)[:GPU_COL_MEM_W]
                prefix_row = f"{idx_s} {util_s} {mem_col}"
                if len(prefix_row) > inner:
                    self.stdscr.addstr(line_y, left, prefix_row[:inner], row_attr)
                    continue

                self.stdscr.addstr(line_y, left, prefix_row, row_attr)
                col = left + prefix_w + 1
                if hist_w > 0:
                    hist = (
                        self._gpu_util_history.get(int(idx), []) if idx != "?" else []
                    )
                    spark = sparkline_trailing(hist, hist_w)
                    self.stdscr.addstr(line_y, col, spark, row_attr)
                    col += hist_w + 1
                proc_strs = [
                    f"{p.get('user', '?')}:{p.get('pid', '?')}"
                    for p in g.get("processes", [])
                    if not p.get("zombie")
                ]
                proc_line = _fit_text_field(", ".join(proc_strs), proc_w).ljust(proc_w)[
                    :proc_w
                ]
                if proc_w > 0:
                    self.stdscr.addstr(line_y, col, proc_line, curses.A_NORMAL)
        except Exception:
            pass

    def draw_job_details(self, y, x, h, w):
        if self.stdscr is None:
            return
        """Draw detailed job information for the selected job across all windows."""
        try:
            if not self.has_selected_job_context:
                self.stdscr.addstr(y + 1, x + 2, "No job selected.", curses.A_NORMAL)
                return

            # Find which job is "selected" across the 3 main windows
            # Or just use the one from the active window if it's a queue
            job = self._selected_job()

            if not job:
                self.stdscr.addstr(y + 1, x + 2, "No job selected.", curses.A_NORMAL)
                return

            self.last_selected_job = job  # Keep it

            jid = job["id"]
            queue_s = str(job.get("_type", "unk")).upper()
            st = str(job.get("status", "-")).upper()
            gpu_s = str(job.get("gpus", "-"))
            meta_str = f"ID: {jid} | Queue: {queue_s} | Status: {st} | GPUs: {gpu_s}"
            inner_w = max(1, w - 4)
            self.stdscr.addstr(
                y, x + 2, _fit_text_field(meta_str, inner_w), curses.A_BOLD
            )

            cmd = job.get("cmd", "")
            # Normalize command (remove newlines)
            cmd = cmd.replace("\n", " ").replace("\r", " ")

            prefix = "Cmd: "
            # Width calculation checks
            # w - 5 - len(prefix) ?
            # w is Full Width.
            # Draws at x+2.
            # Max index w-2 (border at w-1).
            # So length available = w-4.
            # Subtract prefix.
            # Subtract 2 more for safety.
            safe_width = max(1, inner_w - len(prefix))

            import textwrap

            lines = textwrap.wrap(cmd, width=safe_width) or ["-"]

            for i, line in enumerate(lines[: h - 1]):
                self.stdscr.addstr(
                    y + 1 + i,
                    x + 2,
                    prefix if i == 0 else " " * len(prefix),
                    curses.A_NORMAL,
                )
                self.stdscr.addstr(
                    y + 1 + i,
                    x + 2 + len(prefix),
                    _fit_text_field(line, safe_width),
                )
        except Exception:
            pass

    def draw_window(self, win, y, x, h, w, active, focused):
        if self.stdscr is None:
            return
        # Draw Box
        try:
            # Border Color logic: Blue when focused, white when selected.
            # If unselected, don't draw borders (they'll blend with background).
            if focused:
                border_color = curses.color_pair(5)  # Blue
                title_color = curses.color_pair(5) | curses.A_BOLD  # Blue bold
                draw_border = True
            elif active:
                border_color = curses.A_NORMAL
                title_color = curses.A_BOLD
                draw_border = True
            else:
                # Unselected: don't draw border; title should use normal text.
                border_color = None
                title_color = curses.A_NORMAL
                draw_border = False

            if draw_border:
                self.stdscr.attron(border_color)
                self.stdscr.hline(y, x, curses.ACS_HLINE, w)
                if not win.collapsed:
                    self.stdscr.hline(y + h - 1, x, curses.ACS_HLINE, w)
                    self.stdscr.vline(y, x, curses.ACS_VLINE, h)
                    self.stdscr.vline(y, x + w - 1, curses.ACS_VLINE, h)
                    self.stdscr.addch(y, x, curses.ACS_ULCORNER)
                    self.stdscr.addch(y, x + w - 1, curses.ACS_URCORNER)
                    self.stdscr.addch(y + h - 1, x, curses.ACS_LLCORNER)
                    self.stdscr.addch(y + h - 1, x + w - 1, curses.ACS_LRCORNER)
                self.stdscr.attroff(border_color)

            # Title
            count_str = ""
            if win.key in ["running", "staging", "pending", "completed"]:
                selected_count = len(self._selected_ids_for_window(win))
                count_str = f"[{len(win.items)}"
                if selected_count:
                    count_str += f"/{selected_count}"
                count_str += "]"

            title_s = f" {win.title} {count_str} "
            if win.collapsed:
                title_s = f" [+] {win.title} {count_str} "
            elif focused:
                title_s = f" [ {win.title} {count_str} ] "

            # Title uses matching color scheme
            self.stdscr.addstr(y, 2, title_s, title_color)

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
                # Dynamic header based on column widths
                cw = self.col_widths["running"]
                hdr = (
                    f" {'ID':<{cw['id']}} {'PID':<{cw['pid']}} "
                    f"{'GPUS':<{cw['gpus']}} {'ELAPSED':<{cw['elapsed']}} CMD"
                )
            elif win.key == "staging":
                cw = self.col_widths["staging"]
                hdr = (
                    f" {'ID':<{cw['id']}} {'GPUS':<{cw['gpus']}} "
                    f"{'WAITING':<{cw['waiting']}} CMD"
                )
            elif win.key == "pending":
                # Dynamic header based on column widths
                cw = self.col_widths["pending"]
                hdr = (
                    f" {'ID':<{cw['id']}} {'GPUS':<{cw['gpus']}} "
                    f"{'WAITING':<{cw['waiting']}} CMD"
                )
            elif win.key == "completed":
                # Dynamic header based on column widths
                cw = self.col_widths["completed"]
                hdr = (
                    f" {'ID':<{cw['id']}} {'RUNTIME':<{cw['runtime']}} "
                    f"{'AGO':<{cw['ago']}} CMD"
                )

            # List items
            display_items = list(win.items)

            # Edit-mode row swap in staging only
            if (
                active
                and self.edit_mode_active
                and self.edit_job
                and win.key == "staging"
            ):
                edit_copy = copy.deepcopy(self.edit_job)
                edit_copy["_edit_field_idx"] = self.edit_field_idx
                for idx, it in enumerate(display_items):
                    if it["id"] == edit_copy["id"]:
                        display_items[idx] = edit_copy
                        break

            # Recalculate list height or just use what we have
            list_h = h - 2 - header_offset
            if list_h < 1:
                return

            start_y = y + 1 + header_offset
            has_scrollbar = len(display_items) > list_h
            row_text_w = max(0, w - (4 if has_scrollbar else 3))
            scrollbar_x = max(x + 1, x + w - 2)

            if hdr:
                self.stdscr.addstr(
                    y + 1,
                    x + 1,
                    _fit_text_field(hdr, row_text_w),
                    curses.A_BOLD,
                )

            # We need to handle scroll offset carefully if we injected an item
            # If we injected, the list is 1 longer.
            visible_items = display_items[
                win.scroll_offset : win.scroll_offset + list_h
            ]

            for i, item in enumerate(visible_items):
                abs_idx = win.scroll_offset + i

                is_sel = False
                if active:
                    if abs_idx == win.selected_idx:
                        is_sel = True

                line_res, line_col = self.format_job_line(item, row_text_w)

                draw_style = curses.A_NORMAL
                is_bulk_selected = (
                    win.key in self.selected_job_ids
                    and str(item.get("id")) in self.selected_job_ids[win.key]
                )
                if is_bulk_selected:
                    draw_style |= curses.A_REVERSE | curses.A_BOLD
                if is_sel:
                    if focused:
                        draw_style |= curses.A_REVERSE
                    else:
                        # In NAV mode, pass or different style
                        pass

                try:
                    self.stdscr.addstr(
                        start_y + i, x + 1, " " * row_text_w, curses.A_NORMAL
                    )
                except Exception:
                    pass

                # Handle Rich Text (Dictionary)
                if isinstance(line_res, dict) and line_res.get("type") == "rich":
                    current_x = 1
                    segments = line_res["segments"]
                    base_attr = line_res.get("base_color", curses.A_NORMAL)
                    if is_bulk_selected:
                        base_attr |= curses.A_REVERSE | curses.A_BOLD

                    # Clear line with base attr first?
                    # self.stdscr.addstr(start_y + i, 1, " " * (w-2), base_attr)

                    for text, attr in segments:
                        try:
                            self.stdscr.addstr(
                                start_y + i, x + current_x, text, attr | base_attr
                            )
                            current_x += len(text)
                        except Exception:
                            pass
                else:
                    self.stdscr.addstr(
                        start_y + i, x + 1, line_res, line_col | draw_style
                    )

            # Scroll bar indicator?
            if len(display_items) > list_h:
                sb_h = max(1, int(list_h * (list_h / len(display_items))))
                sb_pos = int((win.scroll_offset / len(display_items)) * list_h)
                for k in range(list_h):
                    char = "│"
                    if k >= sb_pos and k < sb_pos + sb_h:
                        char = "█"
                    try:
                        self.stdscr.addstr(
                            start_y + k, scrollbar_x, char, curses.A_NORMAL
                        )
                    except Exception:
                        pass
        except Exception:
            # self.stdscr.addstr(y+1, 1, str(e))
            pass

    def draw_footer(self, y, w):
        if self.stdscr is None:
            return
        try:
            if self.modal:
                return  # Don't draw footer over modal or distract

            help_str = " Q:Quit "
            if self.edit_mode_active:
                help_str += (
                    "e:Save Staging  Esc:Cancel  h/l:Field  "
                    "j/k:GPUs  Enter:command editor"
                )
            elif self.mode == "NAV":
                help_str += "j/k:Select  l:Focus  n:New Job  Tab:Collapse"
            else:
                # Context-aware help based on active window
                win = self.windows[self.active_win_idx]
                if self.select_mode_active and self.select_mode_window_key == win.key:
                    help_str += "j/k:Extend Selection  v:Done  Esc:Cancel"
                elif win.key == "staging":
                    help_str += "h:Back  v:Select  c:Discard  e:Edit  s:Send  d:Dup"
                elif win.key == "pending":
                    help_str += (
                        "h:Back  v:Select  b:Stage  c:Cancel  J/K:Reorder  Space:Log"
                    )
                elif win.key == "running":
                    help_str += "h:Back  v:Select  Space:Log  c:Cancel  p:Pause  d:Dup"
                elif win.key == "completed":
                    help_str += "h:Back  v:Select  Space:Log  r:Retry  x:Delete  d:Dup"
                else:
                    help_str += "h:Back  Space:Log  d:Dup  n:New"

            # Mode display on the right
            mode_label = self.mode
            if self.edit_mode_active:
                mode_label = "EDIT"
            elif self.select_mode_active:
                mode_label = "SELECT"

            mode_s = f" MODE: {mode_label} "
            if len(mode_s) >= w:
                full_str = _fit_text_field(mode_s, w)
            else:
                help_str = _fit_text_field(help_str, w - len(mode_s)).rstrip()
                padding = " " * max(0, w - len(help_str) - len(mode_s))
                full_str = help_str + padding + mode_s

            # Blue background (Pair 5)
            self.stdscr.addstr(y, 0, full_str, curses.color_pair(5) | curses.A_REVERSE)
        except Exception:
            pass

    def draw_log_overlay(self, h, w):
        if self.stdscr is None:
            return
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
        if self.stdscr:
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
            if self.stdscr:
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
                            if ch == 27 and self._has_selected_ids():
                                self._cancel_select_mode()
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
                            cpos_int = cast(int, cpos)
                            m["cursor_pos"] = max(0, cpos_int - width)
                        elif ch == curses.KEY_DOWN:
                            width = 80 - 6
                            cpos_int = cast(int, cpos)
                            m["cursor_pos"] = min(len(str(val)), cpos_int + width)
                        elif ch == curses.KEY_HOME:
                            m["cursor_pos"] = 0
                        elif ch == curses.KEY_END:
                            m["cursor_pos"] = len(val)
                        elif ch >= 32 and ch <= 126:  # Printable
                            cpos_int = cast(int, cpos)
                            m["value"] = (
                                str(val)[:cpos_int] + chr(ch) + str(val)[cpos_int:]
                            )
                            m["cursor_pos"] = cpos_int + 1
                    continue

                if self.edit_mode_active:
                    # Force redraw of footer/status
                    # self.stdscr.touchwin()

                    if ch == 27:  # Esc -> Cancel
                        self.edit_mode_active = False
                        self.edit_job = None
                    elif ch == ord("e"):  # Confirm
                        if self.edit_job is None:
                            continue
                        self.execute_action(
                            "update_staging",
                            self.edit_job["id"],
                            cmd=self.edit_job["cmd"],
                            gpus=self.edit_job["gpus"],
                        )
                        self.edit_mode_active = False
                        self.edit_job = None

                    elif ch == ord("h"):  # Cycle Left
                        self.edit_field_idx = max(0, self.edit_field_idx - 1)
                    elif ch == ord("l"):  # Cycle Right
                        self.edit_field_idx = min(1, self.edit_field_idx + 1)

                    elif ch == ord("j"):  # Decrease Value
                        if self.edit_job is None:
                            continue
                        if self.edit_field_idx == 0:  # GPUS
                            self.edit_job["gpus"] = max(
                                1, int(self.edit_job["gpus"]) - 1
                            )

                    elif ch == ord("k"):  # Increase Value
                        if self.edit_job is None:
                            continue
                        if self.edit_field_idx == 0:  # GPUS
                            self.edit_job["gpus"] = int(self.edit_job["gpus"]) + 1

                    elif ch == 10 or ch == ord("i"):  # Enter/Edit Text
                        if self.edit_field_idx == 1:  # Command
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
                            self._enter_action_window(curr_win)
                    elif ch == 10:  # Enter
                        curr_win = self.windows[self.active_win_idx]
                        if curr_win.key not in ["gpu_status", "job_details"]:
                            self._enter_action_window(curr_win)
                    elif ch == 9:  # Tab
                        self.windows[self.active_win_idx].collapsed = not self.windows[
                            self.active_win_idx
                        ].collapsed
                    elif ch == ord("n"):  # New Job (Global context)
                        self.prompt_new_job()

                elif self.mode == "ACTION":
                    win = self.windows[self.active_win_idx]
                    window_h = getattr(win, "height", None)

                    if ch == ord("h") or ch == 27:  # h or Esc
                        if self.select_mode_active:
                            self._cancel_select_mode()
                            continue
                        if ch == 27 and self._has_selected_ids():
                            self._cancel_select_mode()
                            continue
                        self._exit_action_window(win)
                    elif ch == ord("k"):
                        if self.select_mode_active:
                            self._select_mode_move(win, -1, window_h)
                        else:
                            win.scroll(-1, window_h)
                    elif ch == ord("j"):
                        if self.select_mode_active:
                            self._select_mode_move(win, 1, window_h)
                        else:
                            win.scroll(1, window_h)
                    elif ch == ord("v"):
                        if self.select_mode_active:
                            self._exit_select_mode()
                        else:
                            self._enter_select_mode(win)
                    elif ch == ord(" "):
                        self.action_view_logs()
                    elif ch == ord("L"):
                        self.action_open_external_logs()

                    # Actions (context-aware per window type)
                    elif ch == ord("c"):
                        if win.key in ["staging", "pending", "running"]:
                            self.do_action("cancel")
                    elif ch == ord("x"):  # Remove/Delete (completed only)
                        if win.key == "completed":
                            self.do_action("remove")
                    elif ch == ord("d"):  # Dup (all windows)
                        self.do_action("dup")
                    elif ch == ord("n"):  # New
                        self.prompt_new_job()
                    elif ch == ord("p"):  # Pause (running only)
                        if win.key == "running":
                            self.do_action("pause")
                    elif ch == ord("e"):  # Edit (staging only)
                        if win.key == "staging":
                            self.do_action("edit")
                    elif ch == ord("r"):  # Retry (completed only)
                        if win.key == "completed":
                            self.do_action("retry")
                    elif ch == ord("s"):  # Send staged job to pending
                        if win.key == "staging":
                            self.do_action("send_to_pending")
                    elif ch == ord("b"):  # Move pending job back to staging
                        if win.key == "pending":
                            self.do_action("back_to_staging")
                    elif ch == ord("J"):  # Move pending job down
                        if win.key == "pending":
                            self.do_action("move_pending_down")
                    elif ch == ord("K"):  # Move pending job up
                        if win.key == "pending":
                            self.do_action("move_pending_up")
                    elif ch == 10:  # Enter sends staged job to pending (with confirm)
                        if win.key == "staging":
                            self.do_action("send_to_pending")

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
        if self.edit_job is None:
            return

        # Use external editor
        # 1. Write current cmd to temp file
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".sh") as tf:
            tf.write(str(self.edit_job["cmd"]))
            tf_path = tf.name

        if self.stdscr:
            curses.def_shell_mode()
            self.stdscr.clear()
            self.stdscr.refresh()
            curses.endwin()

        try:
            editor = os.environ.get("EDITOR", "nano")
            subprocess.run([editor, tf_path])

            # 3. Read back (collapse newlines/whitespace to single line)
            with open(tf_path, "r") as f:
                new_cmd = " ".join(f.read().split())
                self._update_edit_cmd(new_cmd)

            os.unlink(tf_path)
        except Exception:
            pass

        finally:
            if self.stdscr:
                self.stdscr.refresh()
                curses.doupdate()
                curses.reset_shell_mode()
                self.stdscr.keypad(True)
                self.stdscr.nodelay(True)

    def _update_edit_cmd(self, val):
        if self.edit_job is None:
            return
        if self.edit_job:
            self.edit_job["cmd"] = val

    def prompt_new_job(self):
        job = make_staged_job(generate_job_id())
        with locked_queue() as q:
            insert_staged_job(q, job)

        self.edit_job = copy.deepcopy(job)
        self.edit_job["_type"] = "staging"
        self.edit_is_new = True
        self.edit_mode_active = True
        self.edit_field_idx = 0

        for i, w in enumerate(self.windows):
            if w.key == "staging":
                self.active_win_idx = i
                w.selected_idx = 0
                w.scroll_offset = 0
                self.has_selected_job_context = True
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
        cursor_only_actions = {"edit"}
        focus_jid = str(job["id"]) if job else None
        if action in cursor_only_actions:
            if not job:
                return
            jids = [str(job["id"])]
        elif (
            action in {"move_pending_up", "move_pending_down"} and win.key == "pending"
        ):
            selected = self._selected_ids_for_window(win)
            if selected:
                jids = selected
                if focus_jid not in set(jids):
                    focus_jid = jids[0]
            else:
                if not job:
                    return
                jids = [str(job["id"])]
        else:
            jids = self._action_ids_for_window(win)
        if not jids:
            return

        jid = jids[0]
        count = len(jids)
        target = f"{count} jobs" if count > 1 else f"job {jid}"

        # Actions that require modals
        if action == "cancel":
            if win.key not in ["staging", "pending", "running"]:
                return
            if win.key == "staging":
                self.modal = {
                    "type": "CONFIRM",
                    "title": "Cancel Staged Job",
                    "text": f"Cancel staged {target}?",
                    "on_confirm": lambda jids=jids: self.execute_bulk_action(
                        "cancel", jids
                    ),
                    "on_cancel": None,
                }
                return

            self.modal = {
                "type": "CONFIRM",
                "title": "Cancel Job",
                "text": f"Cancel {target}?",
                "on_confirm": lambda jids=jids: self.execute_bulk_action(
                    "cancel", jids
                ),
                "on_cancel": None,
            }
            return

        elif action == "remove":
            # Remove only works for completed (already checked in keybinding)
            self.modal = {
                "type": "CONFIRM",
                "title": "Delete Job",
                "text": f"Permanently delete {target}?",
                "on_confirm": lambda jids=jids: self.execute_bulk_action(
                    "delete", jids
                ),
                "on_cancel": None,
            }
            return

        elif action == "edit":
            # Edit only works for staging (already checked in keybinding)
            if not job:
                return
            self.edit_job = copy.deepcopy(job)
            self.edit_job["_type"] = "staging"
            self.edit_is_new = False
            self.edit_mode_active = True
            self.edit_field_idx = 0
            return

        elif action == "dup":
            self.execute_bulk_action("dup", jids)
            return
        elif action == "send_to_pending":
            if win.key != "staging":
                return
            self.modal = {
                "type": "CONFIRM",
                "title": "Send To Pending",
                "text": f"Send {target} to pending queue?",
                "on_confirm": lambda jids=jids: self.execute_bulk_action(
                    "send_to_pending", jids
                ),
                "on_cancel": None,
            }
            return
        elif action == "back_to_staging":
            if win.key != "pending":
                return
            self.execute_bulk_action("back_to_staging", jids)
            return
        elif action == "retry":
            if win.key != "completed":
                return
            self.modal = {
                "type": "CONFIRM",
                "title": "Retry Completed Job",
                "text": (
                    f"Retry {target}? This may replace or overwrite access to old logs."
                ),
                "on_confirm": lambda jids=jids: self.execute_bulk_action("retry", jids),
                "on_cancel": None,
            }
            return

        # Immediate actions
        self.execute_bulk_action(action, jids, focus_jid=focus_jid)

    def execute_bulk_action(self, action, jids, **kwargs):
        jids = list(dict.fromkeys(str(jid) for jid in jids))
        if not jids:
            return
        if len(jids) == 1:
            self.execute_action(action, jids[0], **kwargs)
            return

        if action in ["edit", "change_gpus"]:
            self.action_msg = "Action only supports the cursor row"
            self.msg_clear_time = time.time() + 2.0
            return

        if action in ["move_pending_up", "move_pending_down"]:
            offset = -1 if action == "move_pending_up" else 1
            focus_jid = (
                str(kwargs.get("focus_jid")) if kwargs.get("focus_jid") else None
            )
            queue_snapshot = None
            with locked_queue() as q:
                if move_pending_jobs(q, jids, offset):
                    queue_snapshot = copy.deepcopy(q)
                else:
                    self.action_msg = "Cannot move further"
                    self.msg_clear_time = time.time() + 2.0
                    return
            if queue_snapshot is not None:
                with self.lock:
                    self._set_data_snapshot(queue_snapshot)
                    for w in self.windows:
                        if w.key != "pending":
                            continue
                        target_idx = None
                        if focus_jid is not None:
                            for idx, pending_job in enumerate(w.items):
                                if str(pending_job.get("id")) == focus_jid:
                                    target_idx = idx
                                    break
                        if target_idx is not None:
                            w.selected_idx = target_idx
                        w.ensure_selected_visible()
                        break
            direction = "up" if offset < 0 else "down"
            self.action_msg = (
                f"Moved {len(jids)} selected jobs {direction}"
                if len(jids) > 1
                else f"Moved {jids[0]} {direction}"
            )
            self.msg_clear_time = time.time() + 2.0
            return

        if action == "dup":
            created = []
            with locked_queue() as q:
                jobs_by_id = {
                    str(job.get("id")): job
                    for key in ["running", "pending", "staging", "completed"]
                    for job in q.get(key, [])
                }
                for jid in jids:
                    job = jobs_by_id.get(jid)
                    if job is None:
                        continue
                    dup_job = make_staged_job(
                        generate_job_id(), job.get("cmd", ""), job.get("gpus", 1)
                    )
                    insert_staged_job(q, dup_job)
                    created.append(dup_job["id"])
            self.action_msg = f"Duplicated {len(created)} jobs"
            self.msg_clear_time = time.time() + 2.0
            return

        if action == "back_to_staging":
            moved = []
            queue_snapshot = None
            with locked_queue() as q:
                for jid in reversed(jids):
                    if move_pending_job_to_staging(q, jid):
                        moved.append(jid)
                if moved:
                    queue_snapshot = copy.deepcopy(q)
            if queue_snapshot is not None:
                with self.lock:
                    self._set_data_snapshot(queue_snapshot)
            self._clear_selected_ids(jids)
            self.action_msg = f"Moved {len(moved)} jobs to staging"
            self.msg_clear_time = time.time() + 2.0
            return

        messages = []
        for jid in jids:
            self.execute_action(action, jid, **kwargs)
            if self.action_msg:
                messages.append(self.action_msg)

        action_labels = {
            "cancel": "Cancelled",
            "delete": "Deleted",
            "pause": "Paused",
            "retry": "Staged retries for",
            "dup": "Duplicated",
            "send_to_pending": "Sent",
            "back_to_staging": "Moved to staging",
            "discard_staging": "Discarded",
        }
        label = action_labels.get(action, "Updated")
        self.action_msg = f"{label} {len(messages) or len(jids)} jobs"
        self._clear_selected_ids(jids)
        self.msg_clear_time = time.time() + 2.0

    def execute_action(self, action, jid, **kwargs):
        # Re-use existing cmd functions if possible, or call logic directly
        msg = ""
        try:
            if action == "cancel":
                # If it's a pending job, move to completed with status 'cancelled'
                # If it's running, call the external tool
                is_pending = False
                is_staging = False
                with locked_queue() as q:
                    for j in q["staging"]:
                        if j["id"] == jid:
                            is_staging = True
                            break
                    for j in q["pending"]:
                        if j["id"] == jid:
                            is_pending = True
                            break

                if is_staging:
                    with locked_queue() as q:
                        if cancel_staged_job(q, jid):
                            msg = f"Cancelled staged {jid}"
                elif is_pending:
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

            elif action == "dup":
                with locked_queue() as q:
                    jobs_by_id = {
                        str(job.get("id")): job
                        for key in ["running", "pending", "staging", "completed"]
                        for job in q.get(key, [])
                    }
                    job = jobs_by_id.get(str(jid))
                    if job is not None:
                        dup_job = make_staged_job(
                            generate_job_id(), job.get("cmd", ""), job.get("gpus", 1)
                        )
                        insert_staged_job(q, dup_job)
                        msg = f"Duplicated {jid} to {dup_job['id']}"

            elif action == "retry":
                with locked_queue() as q:
                    for job in q["completed"]:
                        if job["id"] == jid:
                            new_job = make_staged_job(
                                generate_job_id(), job["cmd"], job.get("gpus", 1)
                            )
                            if stage_completed_retry(q, jid, new_job):
                                msg = f"Staged retry for {jid}"
                            break

            elif action == "update_staging":
                cmd = kwargs.get("cmd")
                gpus = kwargs.get("gpus")

                with locked_queue() as q:
                    for job in q["staging"]:
                        if job["id"] == jid:
                            if cmd is not None:
                                job["cmd"] = cmd
                            if gpus is not None:
                                job["gpus"] = gpus
                            msg = f"Updated staged job {jid}"
                            break

            elif action == "send_to_pending":
                with locked_queue() as q:
                    if send_staged_job_to_pending(q, jid):
                        msg = f"Sent {jid} to pending"

            elif action == "back_to_staging":
                queue_snapshot = None
                with locked_queue() as q:
                    if move_pending_job_to_staging(q, jid):
                        msg = f"Moved {jid} to staging"
                        queue_snapshot = copy.deepcopy(q)
                if queue_snapshot is not None:
                    with self.lock:
                        self._set_data_snapshot(queue_snapshot)

            elif action == "discard_staging":
                with locked_queue() as q:
                    if cancel_staged_job(q, jid):
                        msg = f"Discarded staged job {jid}"
                        if self.edit_mode_active and self.edit_job is not None:
                            if self.edit_job.get("id") == jid:
                                self.edit_mode_active = False
                                self.edit_job = None

            elif action in ["move_pending_up", "move_pending_down"]:
                offset = -1 if action == "move_pending_up" else 1
                new_idx = None
                queue_snapshot = None
                with locked_queue() as q:
                    if move_pending_job(q, jid, offset):
                        msg = f"Moved {jid} {'up' if offset < 0 else 'down'}"
                        for idx, job in enumerate(q["pending"]):
                            if job.get("id") == jid:
                                new_idx = idx
                                break
                        queue_snapshot = copy.deepcopy(q)
                    else:
                        msg = "Cannot move further"

                if queue_snapshot is not None:
                    with self.lock:
                        self._set_data_snapshot(queue_snapshot)
                        for w in self.windows:
                            if w.key == "pending":
                                if new_idx is not None:
                                    w.selected_idx = new_idx
                                w.ensure_selected_visible()
                                break

        except Exception as e:
            msg = f"Err: {str(e)}"

        if action not in [
            "dup",
            "move_pending_up",
            "move_pending_down",
            "update_staging",
        ]:
            self._clear_selected_ids([jid])
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

    # stop
    stop_parser = subparsers.add_parser("stop", help="Stop the background scheduler")
    stop_parser.set_defaults(func=cmd_stop)

    # serve
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
