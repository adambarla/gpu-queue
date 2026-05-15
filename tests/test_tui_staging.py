import copy
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from typing import Optional
from unittest.mock import patch

from gpu_queue import main as mod


def _job(job_id: str, *, cmd: str = "echo hi", gpus: int = 1, added: str = "2026-01-01T00:00:00", status: Optional[str] = None) -> dict:
    job = {"id": job_id, "cmd": cmd, "gpus": gpus, "added": added}
    if status is not None:
        job["status"] = status
    return job


class TuiStagingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.queue_dir = Path(self.tmp.name) / ".gpu_queue"
        self.stack = ExitStack()
        for name, value in {
            "QUEUE_DIR": self.queue_dir,
            "QUEUE_FILE": self.queue_dir / "jobs.json",
            "PID_FILE": self.queue_dir / "daemon.pid",
            "DAEMON_LOG": self.queue_dir / "daemon.log",
            "LOG_DIR": self.queue_dir / "logs",
            "LOCK_FILE": self.queue_dir / "queue.lock",
        }.items():
            self.stack.enter_context(patch.object(mod, name, value))
        self.addCleanup(self.stack.close)

    def write_queue(self, data: dict) -> None:
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        mod.save_queue_raw(data)

    def load_tui(self, data: dict) -> mod.GPUQueueTUI:
        self.write_queue(data)
        tui = mod.GPUQueueTUI(interval=0.1)
        current = mod.load_queue_raw()
        tui.data = copy.deepcopy(current)
        for win in tui.windows:
            items = copy.deepcopy(current.get(win.key, []))
            for item in items:
                item["_type"] = win.key
            win.update_items(items)
        return tui

    def staging_idx(self, tui: mod.GPUQueueTUI) -> int:
        for i, win in enumerate(tui.windows):
            if win.key == "staging":
                return i
        raise AssertionError("staging window not found")

    def pending_idx(self, tui: mod.GPUQueueTUI) -> int:
        for i, win in enumerate(tui.windows):
            if win.key == "pending":
                return i
        raise AssertionError("pending window not found")

    def completed_idx(self, tui: mod.GPUQueueTUI) -> int:
        for i, win in enumerate(tui.windows):
            if win.key == "completed":
                return i
        raise AssertionError("completed window not found")

    def test_prompt_new_job_inserts_on_top(self):
        tui = self.load_tui(
            {
                "staging": [_job("s1")],
                "pending": [],
                "running": [],
                "completed": [],
            }
        )
        idx = self.staging_idx(tui)
        tui.active_win_idx = idx
        tui.prompt_new_job()

        data = mod.load_queue_raw()
        self.assertEqual(data["staging"][0]["cmd"], "")
        self.assertEqual(tui.windows[idx].selected_idx, 0)
        self.assertTrue(tui.edit_mode_active)

    def test_edit_staged_job_enters_edit_mode(self):
        tui = self.load_tui(
            {
                "staging": [_job("s1", cmd="python train.py")],
                "pending": [],
                "running": [],
                "completed": [],
            }
        )
        idx = self.staging_idx(tui)
        tui.active_win_idx = idx
        tui.windows[idx].selected_idx = 0

        tui.do_action("edit")

        self.assertTrue(tui.edit_mode_active)
        self.assertEqual(tui.edit_job["id"], "s1")
        self.assertEqual(tui.edit_job["cmd"], "python train.py")

    def test_cancel_staged_job_moves_to_completed(self):
        tui = self.load_tui(
            {
                "staging": [_job("s1", cmd="python train.py")],
                "pending": [],
                "running": [],
                "completed": [],
            }
        )
        idx = self.staging_idx(tui)
        tui.active_win_idx = idx
        tui.windows[idx].selected_idx = 0

        tui.do_action("cancel")
        tui.modal["on_confirm"]()

        data = mod.load_queue_raw()
        self.assertEqual(data["staging"], [])
        self.assertEqual(data["completed"][0]["id"], "s1")
        self.assertEqual(data["completed"][0]["status"], "cancelled")

    def test_retry_stages_completed_job_on_top(self):
        tui = self.load_tui(
            {
                "staging": [],
                "pending": [],
                "running": [],
                "completed": [_job("c1", cmd="python eval.py", status="failed")],
            }
        )
        completed_idx = next(i for i, win in enumerate(tui.windows) if win.key == "completed")
        tui.active_win_idx = completed_idx
        tui.windows[completed_idx].selected_idx = 0

        tui.do_action("retry")
        self.assertEqual(tui.modal["type"], "CONFIRM")
        self.assertIn("old logs", tui.modal["text"])
        tui.modal["on_confirm"]()

        data = mod.load_queue_raw()
        self.assertEqual(data["completed"], [])
        self.assertEqual(data["staging"][0]["cmd"], "python eval.py")
        self.assertNotEqual(data["staging"][0]["id"], "c1")

    def test_duplicate_does_not_enter_edit_mode(self):
        tui = self.load_tui(
            {
                "staging": [],
                "pending": [_job("p1", cmd="python train.py", gpus=2)],
                "running": [],
                "completed": [],
            }
        )
        pending_idx = self.pending_idx(tui)
        tui.active_win_idx = pending_idx
        tui.windows[pending_idx].selected_idx = 0

        tui.do_action("dup")

        data = mod.load_queue_raw()
        self.assertEqual(len(data["staging"]), 1)
        self.assertEqual(data["staging"][0]["cmd"], "python train.py")
        self.assertEqual(data["staging"][0]["gpus"], 2)
        self.assertFalse(tui.edit_mode_active)
        self.assertIsNone(tui.edit_job)

    def test_duplicate_keeps_current_window_and_cursor(self):
        tui = self.load_tui(
            {
                "staging": [],
                "pending": [_job("p1"), _job("p2"), _job("p3")],
                "running": [],
                "completed": [],
            }
        )
        pending_idx = self.pending_idx(tui)
        win = tui.windows[pending_idx]
        tui.active_win_idx = pending_idx
        win.selected_idx = 1
        win.scroll_offset = 1

        tui.do_action("dup")

        self.assertEqual(tui.active_win_idx, pending_idx)
        self.assertEqual(win.selected_idx, 1)
        self.assertEqual(win.scroll_offset, 1)

    def test_send_staged_job_to_pending_requires_confirmation(self):
        tui = self.load_tui(
            {
                "staging": [_job("s1", cmd="python train.py")],
                "pending": [],
                "running": [],
                "completed": [],
            }
        )
        idx = self.staging_idx(tui)
        tui.active_win_idx = idx
        tui.windows[idx].selected_idx = 0

        tui.do_action("send_to_pending")
        self.assertEqual(tui.modal["type"], "CONFIRM")
        tui.modal["on_confirm"]()

        data = mod.load_queue_raw()
        self.assertEqual(data["staging"], [])
        self.assertEqual(data["pending"][0]["id"], "s1")

    def test_pending_reorder_moves_selected_job(self):
        tui = self.load_tui(
            {
                "staging": [],
                "pending": [_job("p1"), _job("p2"), _job("p3")],
                "running": [],
                "completed": [],
            }
        )
        pending_idx = next(i for i, win in enumerate(tui.windows) if win.key == "pending")
        tui.active_win_idx = pending_idx
        tui.windows[pending_idx].selected_idx = 1

        tui.execute_action("move_pending_up", "p2")

        data = mod.load_queue_raw()
        self.assertEqual([j["id"] for j in data["pending"]], ["p2", "p1", "p3"])
        self.assertEqual(tui.windows[pending_idx].selected_idx, 0)

    def test_pending_reorder_keeps_moved_job_visible_when_moving_up(self):
        tui = self.load_tui(
            {
                "staging": [],
                "pending": [_job(f"p{i}") for i in range(8)],
                "running": [],
                "completed": [],
            }
        )
        pending_idx = self.pending_idx(tui)
        win = tui.windows[pending_idx]
        tui.active_win_idx = pending_idx
        win.height = 5
        win.selected_idx = 3
        win.scroll_offset = 3

        tui.execute_action("move_pending_up", "p3")

        data = mod.load_queue_raw()
        self.assertEqual([j["id"] for j in data["pending"][:4]], ["p0", "p1", "p3", "p2"])
        self.assertEqual(win.selected_idx, 2)
        self.assertEqual(win.scroll_offset, 2)

    def test_pending_reorder_updates_window_items_for_rapid_repeated_moves(self):
        tui = self.load_tui(
            {
                "staging": [],
                "pending": [_job("p1"), _job("p2"), _job("p3"), _job("p4")],
                "running": [],
                "completed": [],
            }
        )
        pending_idx = self.pending_idx(tui)
        tui.active_win_idx = pending_idx
        tui.windows[pending_idx].selected_idx = 1

        tui.do_action("move_pending_down")
        tui.do_action("move_pending_down")

        data = mod.load_queue_raw()
        self.assertEqual([j["id"] for j in data["pending"]], ["p1", "p3", "p4", "p2"])
        self.assertEqual(tui.windows[pending_idx].selected_idx, 3)

    def test_select_mode_selects_rows_while_moving(self):
        tui = self.load_tui(
            {
                "staging": [],
                "pending": [_job("p1"), _job("p2"), _job("p3")],
                "running": [],
                "completed": [],
            }
        )
        pending_idx = self.pending_idx(tui)
        win = tui.windows[pending_idx]
        tui.active_win_idx = pending_idx
        win.selected_idx = 0

        tui._enter_select_mode(win)
        tui._select_mode_move(win, 1)
        tui._select_mode_move(win, 1)

        self.assertTrue(tui.select_mode_active)
        self.assertEqual(win.selected_idx, 2)
        self.assertEqual(tui._selected_ids_for_window(win), ["p1", "p2", "p3"])

        tui._exit_select_mode()
        self.assertFalse(tui.select_mode_active)
        self.assertEqual(tui._selected_ids_for_window(win), ["p1", "p2", "p3"])

    def test_select_mode_move_up_deselects_rows(self):
        tui = self.load_tui(
            {
                "staging": [],
                "pending": [_job("p1"), _job("p2"), _job("p3"), _job("p4")],
                "running": [],
                "completed": [],
            }
        )
        pending_idx = self.pending_idx(tui)
        win = tui.windows[pending_idx]
        tui.active_win_idx = pending_idx
        win.selected_idx = 0

        tui._enter_select_mode(win)
        tui._select_mode_move(win, 1)
        tui._select_mode_move(win, 1)
        self.assertEqual(tui._selected_ids_for_window(win), ["p1", "p2", "p3"])

        tui._select_mode_move(win, -1)
        self.assertEqual(win.selected_idx, 1)
        self.assertEqual(tui._selected_ids_for_window(win), ["p1", "p2"])

    def test_escape_clears_select_mode_selection(self):
        tui = self.load_tui(
            {
                "staging": [],
                "pending": [_job("p1"), _job("p2")],
                "running": [],
                "completed": [],
            }
        )
        pending_idx = self.pending_idx(tui)
        win = tui.windows[pending_idx]
        tui.active_win_idx = pending_idx
        win.selected_idx = 0

        tui._enter_select_mode(win)
        tui._select_mode_move(win, 1)
        tui._cancel_select_mode()

        self.assertFalse(tui.select_mode_active)
        self.assertEqual(tui._selected_ids_for_window(win), [])

    def test_escape_clears_selection_even_after_leaving_select_mode(self):
        tui = self.load_tui(
            {
                "staging": [_job("s1")],
                "pending": [_job("p1")],
                "running": [],
                "completed": [],
            }
        )
        pending_idx = self.pending_idx(tui)
        win = tui.windows[pending_idx]
        tui.active_win_idx = pending_idx
        tui._enter_select_mode(win)
        tui._exit_select_mode()
        tui.selected_job_ids["staging"] = {"s1"}

        tui._cancel_select_mode()

        self.assertFalse(tui.select_mode_active)
        self.assertEqual(tui.selected_job_ids["pending"], set())
        self.assertEqual(tui.selected_job_ids["staging"], set())

    def test_bulk_cancel_pending_uses_selected_rows(self):
        tui = self.load_tui(
            {
                "staging": [],
                "pending": [_job("p1"), _job("p2"), _job("p3")],
                "running": [],
                "completed": [],
            }
        )
        pending_idx = self.pending_idx(tui)
        tui.active_win_idx = pending_idx
        tui.selected_job_ids["pending"] = {"p1", "p3"}

        tui.do_action("cancel")
        self.assertEqual(tui.modal["type"], "CONFIRM")
        tui.modal["on_confirm"]()

        data = mod.load_queue_raw()
        self.assertEqual([j["id"] for j in data["pending"]], ["p2"])
        self.assertEqual([j["id"] for j in data["completed"]], ["p3", "p1"])
        self.assertEqual(tui.selected_job_ids["pending"], set())

    def test_move_pending_job_back_to_staging_top(self):
        tui = self.load_tui(
            {
                "staging": [_job("s1")],
                "pending": [_job("p1"), _job("p2")],
                "running": [],
                "completed": [],
            }
        )
        pending_idx = self.pending_idx(tui)
        win = tui.windows[pending_idx]
        tui.active_win_idx = pending_idx
        win.selected_idx = 1

        tui.do_action("back_to_staging")

        data = mod.load_queue_raw()
        self.assertEqual([j["id"] for j in data["staging"]], ["p2", "s1"])
        self.assertEqual([j["id"] for j in data["pending"]], ["p1"])
        self.assertEqual(win.selected_idx, 0)

    def test_bulk_move_pending_jobs_back_to_staging_preserves_order(self):
        tui = self.load_tui(
            {
                "staging": [_job("s1")],
                "pending": [_job("p1"), _job("p2"), _job("p3"), _job("p4")],
                "running": [],
                "completed": [],
            }
        )
        pending_idx = self.pending_idx(tui)
        tui.active_win_idx = pending_idx
        tui.selected_job_ids["pending"] = {"p2", "p4"}

        tui.do_action("back_to_staging")

        data = mod.load_queue_raw()
        self.assertEqual([j["id"] for j in data["staging"]], ["p2", "p4", "s1"])
        self.assertEqual([j["id"] for j in data["pending"]], ["p1", "p3"])
        self.assertEqual(tui.selected_job_ids["pending"], set())

    def test_bulk_send_staged_jobs_to_pending(self):
        tui = self.load_tui(
            {
                "staging": [_job("s1"), _job("s2"), _job("s3")],
                "pending": [_job("p1")],
                "running": [],
                "completed": [],
            }
        )
        staging_idx = self.staging_idx(tui)
        tui.active_win_idx = staging_idx
        tui.selected_job_ids["staging"] = {"s1", "s3"}

        tui.do_action("send_to_pending")
        tui.modal["on_confirm"]()

        data = mod.load_queue_raw()
        self.assertEqual([j["id"] for j in data["staging"]], ["s2"])
        self.assertEqual([j["id"] for j in data["pending"]], ["p1", "s1", "s3"])

    def test_bulk_retry_completed_jobs_to_staging(self):
        tui = self.load_tui(
            {
                "staging": [],
                "pending": [],
                "running": [],
                "completed": [
                    _job("c1", cmd="python a.py", status="failed"),
                    _job("c2", cmd="python b.py", status="failed"),
                ],
            }
        )
        completed_idx = self.completed_idx(tui)
        tui.active_win_idx = completed_idx
        tui.selected_job_ids["completed"] = {"c1", "c2"}

        tui.do_action("retry")
        self.assertEqual(tui.modal["type"], "CONFIRM")
        tui.modal["on_confirm"]()

        data = mod.load_queue_raw()
        self.assertEqual(data["completed"], [])
        self.assertEqual(
            [j["cmd"] for j in data["staging"]],
            ["python b.py", "python a.py"],
        )

    def test_reorder_moves_bulk_selection_when_present(self):
        tui = self.load_tui(
            {
                "staging": [],
                "pending": [_job("p1"), _job("p2"), _job("p3")],
                "running": [],
                "completed": [],
            }
        )
        pending_idx = self.pending_idx(tui)
        win = tui.windows[pending_idx]
        tui.active_win_idx = pending_idx
        win.selected_idx = 1
        tui.selected_job_ids["pending"] = {"p2", "p3"}

        tui.do_action("move_pending_up")

        data = mod.load_queue_raw()
        self.assertEqual([j["id"] for j in data["pending"]], ["p2", "p3", "p1"])
        self.assertEqual(win.selected_idx, 0)

    def test_scroll_offset_clamps_to_viewport_after_height_change(self):
        win = mod.Window("PENDING", "pending")
        win.height = 5
        win.update_items([_job(f"p{i}") for i in range(5)])
        win.scroll_offset = 4
        win.selected_idx = 4

        win.ensure_selected_visible(10)

        self.assertEqual(win.scroll_offset, 0)
        self.assertEqual(win.selected_idx, 4)

    def test_selected_job_panel_renders_current_job_in_action_mode(self):
        tui = self.load_tui(
            {
                "staging": [],
                "pending": [_job("p1", cmd="python train.py")],
                "running": [],
                "completed": [],
            }
        )
        pending_idx = self.pending_idx(tui)
        tui.active_win_idx = pending_idx
        tui._enter_action_window(tui.windows[pending_idx])
        screen = FakeScreen(10, 80)
        tui.stdscr = screen

        tui.draw_job_details(1, 0, 6, 80)

        text = screen.text()
        self.assertIn("ID: p1", text)
        self.assertIn("python train.py", text)

    def test_selected_job_panel_clears_after_exiting_action_window(self):
        tui = self.load_tui(
            {
                "staging": [],
                "pending": [_job("p1", cmd="python train.py")],
                "running": [],
                "completed": [],
            }
        )
        pending_idx = self.pending_idx(tui)
        tui.active_win_idx = pending_idx
        tui._enter_action_window(tui.windows[pending_idx])
        screen = FakeScreen(10, 80)
        tui.stdscr = screen

        tui.draw_job_details(1, 0, 6, 80)
        self.assertIn("ID: p1", screen.text())

        screen.calls = []
        tui._exit_action_window(tui.windows[pending_idx])
        tui.draw_job_details(1, 0, 6, 80)

        text = screen.text()
        self.assertIn("No job selected.", text)
        self.assertNotIn("ID: p1", text)

    def test_selected_job_panel_is_empty_until_a_panel_is_entered(self):
        tui = self.load_tui(
            {
                "staging": [],
                "pending": [_job("p1", cmd="python train.py")],
                "running": [],
                "completed": [],
            }
        )
        tui.active_win_idx = self.pending_idx(tui)
        tui.mode = "NAV"
        screen = FakeScreen(10, 80)
        tui.stdscr = screen

        tui.draw_job_details(1, 0, 6, 80)

        text = screen.text()
        self.assertIn("No job selected.", text)
        self.assertNotIn("ID: p1", text)

    def test_draw_layout_stays_onscreen_with_collapsed_queue_window(self):
        tui = self.load_tui(
            {
                "staging": [_job(f"s{i}") for i in range(8)],
                "pending": [_job(f"p{i}") for i in range(8)],
                "running": [
                    {
                        **_job(f"r{i}"),
                        "pid": 1000 + i,
                        "started": "2026-01-01T00:00:00",
                        "assigned_gpus": [i],
                    }
                    for i in range(8)
                ],
                "completed": [_job(f"c{i}", status="success") for i in range(8)],
            }
        )
        tui.mode = "ACTION"
        tui.active_win_idx = self.pending_idx(tui)
        tui.gpu_status = [
            {"index": i, "used_mb": 1000, "total_mb": 24000, "util": 50, "processes": []}
            for i in range(8)
        ]
        next(win for win in tui.windows if win.key == "staging").collapsed = True
        tui.stdscr = FakeScreen(20, 80)

        with patch.object(mod.curses, "color_pair", side_effect=lambda n: n):
            tui.draw()

        self.assertEqual(tui.stdscr.errors, [])

class FakeScreen:
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
        self.calls = []
        self.errors = []

    def getmaxyx(self):
        return self.height, self.width

    def erase(self):
        pass

    def refresh(self):
        pass

    def attron(self, *_args):
        pass

    def attroff(self, *_args):
        pass

    def _check(self, y: int, x: int, length: int):
        if y < 0 or y >= self.height or x < 0 or x + length > self.width:
            err = (y, x, length)
            self.errors.append(err)
            raise RuntimeError(f"out of bounds: {err}")

    def addstr(self, y: int, x: int, value, *_args):
        text = str(value)
        self._check(y, x, len(text))
        self.calls.append((y, x, text))

    def hline(self, y: int, x: int, _ch, n: int, *_args):
        self._check(y, x, n)

    def vline(self, y: int, x: int, _ch, n: int, *_args):
        self._check(y, x, 1)
        self._check(y + n - 1, x, 1)

    def addch(self, y: int, x: int, _ch, *_args):
        self._check(y, x, 1)

    def text(self) -> str:
        return "\n".join(text for _y, _x, text in self.calls)
