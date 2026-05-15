import json
import tempfile
import unittest
from pathlib import Path

from gpu_queue.storage import JsonQueueStore


def _store(root: Path) -> JsonQueueStore:
    return JsonQueueStore(
        queue_file=root / "jobs.json",
        lock_file=root / "queue.lock",
        queue_dir=root,
        log_dir=root / "logs",
    )


class JsonQueueStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.queue_dir = Path(self.tmp.name) / ".gpu_queue"

    def test_transaction_persists_mutations(self):
        store = _store(self.queue_dir)

        with store.transaction() as queue:
            queue["pending"].append(
                {"id": "j1", "cmd": "echo hi", "gpus": 1, "added": "now"}
            )

        saved = json.loads((self.queue_dir / "jobs.json").read_text())
        self.assertEqual(saved["pending"][0]["id"], "j1")
        self.assertTrue((self.queue_dir / "queue.lock").exists())
        self.assertTrue((self.queue_dir / "logs").exists())

    def test_load_normalizes_missing_sections(self):
        store = _store(self.queue_dir)
        self.queue_dir.mkdir(parents=True)
        (self.queue_dir / "jobs.json").write_text(
            json.dumps({"pending": [{"id": "p1"}], "completed": "bad"})
        )

        queue = store.load()

        self.assertEqual(
            list(queue.keys()), ["staging", "pending", "running", "completed"]
        )
        self.assertEqual(queue["pending"], [{"id": "p1"}])
        self.assertEqual(queue["completed"], [])

    def test_load_bad_json_returns_empty_queue(self):
        store = _store(self.queue_dir)
        self.queue_dir.mkdir(parents=True)
        (self.queue_dir / "jobs.json").write_text("{bad json")

        queue = store.load()

        self.assertEqual(
            queue,
            {"staging": [], "pending": [], "running": [], "completed": []},
        )
