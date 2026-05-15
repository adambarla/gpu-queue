from __future__ import annotations

import subprocess
import unittest

from gpu_queue.gpu import NvidiaSmiGpuProvider


def _completed(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["nvidia-smi"], returncode=0, stdout=stdout)


class NvidiaSmiGpuProviderTest(unittest.TestCase):
    def test_snapshot_marks_gpus_with_live_processes_busy(self):
        def fake_run(args, **kwargs):
            query = args[1]
            if query == "--query-gpu=index,memory.used,memory.total,utilization.gpu":
                return _completed("0, 1024, 81920, 75\n1, 0, 81920, 0\n")
            if query == "--query-gpu=index,uuid":
                return _completed("0, GPU-0\n1, GPU-1\n")
            if query == "--query-compute-apps=gpu_uuid,pid,process_name,used_memory":
                return _completed("GPU-0, 111, /home/alice/train.py, 1000\n")
            raise AssertionError(f"unexpected nvidia-smi args: {args}")

        provider = NvidiaSmiGpuProvider(run=fake_run, pid_exists=lambda pid: pid == 111)

        gpus = provider.snapshot()

        self.assertEqual(len(gpus), 2)
        self.assertFalse(gpus[0]["free"])
        self.assertEqual(gpus[0]["processes"][0]["user"], "alice")
        self.assertEqual(gpus[0]["processes"][0]["name"], "train.py")
        self.assertTrue(gpus[1]["free"])

    def test_snapshot_ignores_zombie_processes_for_free_status(self):
        def fake_run(args, **kwargs):
            query = args[1]
            if query == "--query-gpu=index,memory.used,memory.total,utilization.gpu":
                return _completed("0, 1024, 81920, 75\n")
            if query == "--query-gpu=index,uuid":
                return _completed("0, GPU-0\n")
            if query == "--query-compute-apps=gpu_uuid,pid,process_name,used_memory":
                return _completed("GPU-0, 111, [Not Found], 1000\n")
            raise AssertionError(f"unexpected nvidia-smi args: {args}")

        provider = NvidiaSmiGpuProvider(run=fake_run, pid_exists=lambda pid: False)

        gpus = provider.snapshot()

        self.assertTrue(gpus[0]["free"])
        self.assertTrue(gpus[0]["processes"][0]["zombie"])
        self.assertEqual(gpus[0]["processes"][0]["user"], "zombie")
