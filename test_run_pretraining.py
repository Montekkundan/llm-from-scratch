"""The launcher waits for data and preserves completed optimizer updates."""
from array import array
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from gpu_checkpoint import atomic_json, sha256


class RunnerTests(unittest.TestCase):
    def test_wait_then_append_preserves_training_cursor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.json"
            atomic_json(config, {"model": {"vocab_size": 32, "width": 16, "heads": 2, "layers": 1,
                "context": 8, "ff_width": 32}, "training": {"steps": 4, "batch_size": 2,
                "grad_accum": 2, "lr": 0.01, "warmup_steps": 1, "dtype": "float32",
                "validation_every": 2, "validation_batches": 2, "loss_chunk_size": 3,
                "log_every": 100, "checkpoint_keep": 1}})
            def shard(name, count):
                path = root / name
                path.write_bytes(array("H", [index % 32 for index in range(count)]).tobytes())
                return {"path": name, "tokens": count, "sha256": sha256(path)}
            data = {"format_version": 1, "dtype": "<u2", "tokenizer": {"model": "fixture",
                "revision": "fixed", "vocab_size": 32}, "train": [], "val": [shard("val.u16", 65)]}
            manifest, output = root / "manifest.json", root / "run"
            log = root / "runner.log"
            with log.open("w") as handle:
                child = subprocess.Popen([sys.executable, str(Path(__file__).with_name("run_pretraining.py")),
                    "--config", str(config), "--data", str(manifest), "--output", str(output),
                    "--device", "cpu", "--poll-seconds", "0.05", "--save-every", "1steps"],
                    stdout=handle, stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"})
                def wait_for(path, field, value):
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline:
                        if path.exists() and json.loads(path.read_text()).get(field) == value:
                            return json.loads(path.read_text())
                        if child.poll() is not None:
                            self.fail(log.read_text())
                        time.sleep(0.05)
                    self.fail(log.read_text())
                try:
                    wait_for(output / "runner.json", "status", "waiting_data")
                    data["train"] = [shard("first.u16", 65)]
                    atomic_json(manifest, data)
                    first = wait_for(output / "run-report.json", "status", "paused_data")
                    self.assertEqual((first["global_step"], first["tokens"]), (2, 64))
                    data["train"].append(shard("second.u16", 64))
                    atomic_json(manifest, data)
                    self.assertEqual(child.wait(timeout=30), 0, log.read_text())
                    result = json.loads((output / "run-report.json").read_text())
                    self.assertEqual((result["status"], result["global_step"], result["tokens"]), ("complete", 4, 128))
                    events = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
                    self.assertEqual([event["global_step"] for event in events if event["event"] == "train"], [1, 2, 3, 4])
                finally:
                    if child.poll() is None:
                        child.terminate()
                        child.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
