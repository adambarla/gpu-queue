from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gpu_queue.runner import SubprocessJobRunner


class _Proc:
    pid = 4242


class SubprocessJobRunnerTest(unittest.TestCase):
    def test_start_preserves_command_and_uses_job_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            captured = {}

            def fake_popen(cmd, **kwargs):
                captured["cmd"] = cmd
                captured["kwargs"] = kwargs
                return _Proc()

            runner = SubprocessJobRunner(
                log_dir=root / "logs",
                queue_dir=root,
                popen=fake_popen,
            )

            pid = runner.start(
                {
                    "id": "job1",
                    "cmd": "python  train.py --name 'two spaces'",
                    "cwd": str(root / "work"),
                },
                [2, 3],
            )

            self.assertEqual(pid, 4242)
            self.assertIn("python  train.py --name 'two spaces'", captured["cmd"])
            self.assertEqual(captured["kwargs"]["cwd"], root / "work")
            self.assertEqual(captured["kwargs"]["env"]["CUDA_VISIBLE_DEVICES"], "2,3")
            log_text = (root / "logs" / "job1.log").read_text()
            self.assertIn("Command: python  train.py --name 'two spaces'", log_text)
