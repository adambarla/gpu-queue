from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from gpu_queue.storage import load_queue


def get_free_gpus() -> list[dict[str, Any]]:
    """Get list of GPUs with their status (free = no processes running)."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
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
                idx = int(parts[0])
                used = int(parts[1])
                total = int(parts[2])
                util = 0
                if len(parts) >= 4 and parts[3].strip().isdigit():
                    util = int(parts[3].strip())

                gpus[idx] = {
                    "index": idx,
                    "used_mb": used,
                    "total_mb": total,
                    "util": util,
                    "free": True,
                    "processes": [],
                }

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

        for line in proc_result.stdout.strip().split("\n"):
            if "," in line:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    gpu_uuid, pid_str, proc_name, mem_str = (
                        parts[0],
                        parts[1],
                        parts[2],
                        parts[3],
                    )
                    if gpu_uuid in uuid_to_idx:
                        idx = uuid_to_idx[gpu_uuid]
                        if idx in gpus:
                            is_zombie = proc_name == "[Not Found]"
                            if not is_zombie:
                                try:
                                    pid = int(pid_str)
                                    if not Path(f"/proc/{pid}").exists():
                                        is_zombie = True
                                except ValueError:
                                    is_zombie = True

                            user = "unknown"
                            if "/home/" in proc_name:
                                user = proc_name.split("/home/")[1].split("/")[0]
                            elif is_zombie:
                                user = "zombie"

                            gpus[idx]["processes"].append(
                                {
                                    "pid": pid_str,
                                    "user": user,
                                    "name": proc_name.split("/")[-1]
                                    if "/" in proc_name
                                    else proc_name,
                                    "mem_mb": int(mem_str) if mem_str.isdigit() else 0,
                                    "zombie": is_zombie,
                                }
                            )

                            if not is_zombie:
                                gpus[idx]["free"] = False

        return list(gpus.values())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def get_available_gpu_indices() -> list[int]:
    """Get indices of free GPUs (excluding running jobs)."""
    gpus = get_free_gpus()
    free_indices = [g["index"] for g in gpus if g["free"]]

    queue = load_queue()
    reserved_gpus = set()
    for job in queue.get("running", []):
        for gpu_idx in job.get("assigned_gpus", []):
            reserved_gpus.add(gpu_idx)

    return [idx for idx in free_indices if idx not in reserved_gpus]
