from __future__ import annotations

import os
import shlex
import subprocess
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from gpu_queue import paths
from gpu_queue.domain import Job


class SubprocessJobRunner:
    def __init__(
        self,
        log_dir: Path | None = None,
        queue_dir: Path | None = None,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    ) -> None:
        self.log_dir = log_dir or paths.LOG_DIR
        self.queue_dir = queue_dir or paths.QUEUE_DIR
        self.popen = popen

    def start(self, job: Job, gpu_indices: list[int]) -> int:
        """Run a job with the specified GPUs and return the wrapper process PID."""
        log_file = self.log_dir / f"{job['id']}.log"
        exit_file = self.queue_dir / f"{job['id']}.exit"
        gpu_str = ",".join(map(str, gpu_indices))
        cmd = str(job["cmd"])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_str
        local_bin = str(Path.home() / ".local" / "bin")
        if local_bin not in env.get("PATH", ""):
            env["PATH"] = f"{local_bin}:{env.get('PATH', '')}"

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w") as f:
            f.write(f"=== Job {job['id']} ===\n")
            f.write(f"Command: {cmd}\n")
            f.write(f"GPUs: {gpu_str}\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write("=" * 40 + "\n\n")

        q_log = shlex.quote(str(log_file))
        q_exit = shlex.quote(str(exit_file))
        wrapped_cmd = f"({cmd}) >> {q_log} 2>&1; echo $? > {q_exit}"
        cwd = Path(str(job.get("cwd", Path.home() / "pepo"))).expanduser()

        proc = self.popen(
            wrapped_cmd,
            shell=True,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=cwd,
            start_new_session=True,
        )

        return int(proc.pid)
