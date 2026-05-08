import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from gpu_queue import main as mod


def _job(job_id: str, *, gpus: int = 1, cmd: str = "echo hi", added: str = "2026-01-01T00:00:00") -> dict:
    return {"id": job_id, "cmd": cmd, "gpus": gpus, "added": added}


class QueueCoreTest(unittest.TestCase):
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
        self.queue_file.write_text(json.dumps(data))

    @property
    def queue_file(self) -> Path:
        return mod.QUEUE_FILE

    def read_queue(self) -> dict:
        return json.loads(self.queue_file.read_text())

    def test_load_queue_backfills_staging(self):
        self.write_queue({"pending": [_job("p1")], "running": [], "completed": []})

        data = mod.load_queue_raw()

        self.assertEqual(list(data.keys()), ["staging", "pending", "running", "completed"])
        self.assertEqual(data["staging"], [])
        self.assertEqual([j["id"] for j in data["pending"]], ["p1"])

    def test_save_queue_preserves_pending_order(self):
        queue = {
            "staging": [_job("s1")],
            "pending": [_job("p2"), _job("p1")],
            "running": [],
            "completed": [],
        }

        mod.save_queue_raw(queue)

        saved = self.read_queue()
        self.assertEqual([j["id"] for j in saved["staging"]], ["s1"])
        self.assertEqual([j["id"] for j in saved["pending"]], ["p2", "p1"])

    def test_daemon_loop_starts_eligible_job(self):
        self.write_queue(
            {
                "staging": [],
                "pending": [_job("p1", gpus=1), _job("p2", gpus=1)],
                "running": [],
                "completed": [],
            }
        )

        with (
            patch.object(mod, "cleanup_dead_jobs"),
            patch.object(mod, "get_free_gpus", return_value=[{"index": 0, "free": True}, {"index": 1, "free": True}]),
            patch.object(mod, "run_job", return_value=4321),
            patch.object(mod.time, "sleep", side_effect=KeyboardInterrupt),
        ):
            mod.daemon_loop(min_free=1)

        data = mod.load_queue_raw()
        self.assertEqual([j["id"] for j in data["running"]], ["p1"])
        self.assertEqual(data["running"][0]["pid"], 4321)
        self.assertEqual(data["pending"][0]["id"], "p2")
        self.assertEqual(data["running"][0]["assigned_gpus"], [0])

    def test_cmd_add_front_inserts_at_head(self):
        args = type("Args", (), {"command": "cmd-a", "gpus": 2, "priority": "medium", "front": False})()
        mod.cmd_add(args)
        front_args = type("Args", (), {"command": "cmd-b", "gpus": 1, "priority": "medium", "front": True})()
        mod.cmd_add(front_args)

        data = mod.load_queue_raw()
        self.assertEqual([j["cmd"] for j in data["pending"]], ["cmd-b", "cmd-a"])

