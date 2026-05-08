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

        data = mod.load_queue_raw()
        self.assertEqual(data["completed"], [])
        self.assertEqual(data["staging"][0]["cmd"], "python eval.py")
        self.assertNotEqual(data["staging"][0]["id"], "c1")

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
