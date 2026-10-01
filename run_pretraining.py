"""Wait for immutable prepared shards and resume at optimizer boundaries."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

from gpu_checkpoint import atomic_json


def ready_tokens(manifest):
    if not manifest.exists():
        return 0
    value = json.loads(manifest.read_text())
    return sum(row["tokens"] for row in value.get("train", []))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-every", default="300s")
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    config = json.loads(Path(args.config).read_text())
    training = config["training"]
    per_update = training["batch_size"] * training["grad_accum"] * config["model"]["context"]
    output, manifest = Path(args.output), Path(args.data)
    output.mkdir(parents=True, exist_ok=True)
    stopping = False
    child = None

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not stopping:
        previous = json.loads((output / "run-report.json").read_text()) if (output / "run-report.json").exists() else {}
        if previous.get("status") == "complete":
            return
        consumed = int(previous.get("tokens", 0))
        available = ready_tokens(manifest)
        if available < consumed + per_update + 1:
            atomic_json(output / "runner.json", {"status": "waiting_data", "consumed_tokens": consumed,
                "available_tokens": available, "required_tokens": consumed + per_update + 1})
            time.sleep(args.poll_seconds)
            continue
        command = [sys.executable, "-u", str(Path(__file__).with_name("gpu_train.py")),
            "--config", args.config, "--data", args.data, "--output", args.output,
            "--device", args.device, "--save-every", args.save_every]
        if (output / "latest.json").exists():
            command += ["--resume", "latest"]
        atomic_json(output / "runner.json", {"status": "training", "command": command})
        child = subprocess.Popen(command)
        code = child.wait()
        child = None
        if stopping:
            atomic_json(output / "runner.json", {"status": "stopped", "exit_code": code})
            return
        if code:
            atomic_json(output / "runner.json", {"status": "error", "exit_code": code})
            raise SystemExit(code)
        if not (output / "run-report.json").exists():
            raise RuntimeError("Trainer returned without its report; refusing an automatic restart")
        result = json.loads((output / "run-report.json").read_text())
        if result["status"] == "complete":
            atomic_json(output / "runner.json", {"status": "complete", "tokens": result["tokens"]})
            return
        if result["status"] != "paused_data":
            raise RuntimeError(f"Trainer stopped with {result['status']}; inspect before resuming")


if __name__ == "__main__":
    main()
