"""Atomic, CPU-portable optimizer-boundary checkpoints."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid

import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def cpu_state(value, memo=None):
    memo = {} if memo is None else memo
    if isinstance(value, torch.Tensor):
        key = (str(value.device), str(value.dtype), value.untyped_storage().data_ptr(),
               value.storage_offset(), tuple(value.shape), tuple(value.stride()))
        if key not in memo:
            memo[key] = value.detach().cpu().clone()
        return memo[key]
    if isinstance(value, dict):
        return {key: cpu_state(item, memo) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_state(item, memo) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(item, memo) for item in value)
    return value


def tensor_bytes(value, seen=None):
    seen = set() if seen is None else seen
    if isinstance(value, torch.Tensor):
        key = (str(value.device), value.untyped_storage().data_ptr())
        if key in seen:
            return 0
        seen.add(key)
        return value.untyped_storage().nbytes()
    if isinstance(value, dict):
        return sum(tensor_bytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(item, seen) for item in value)
    return 0


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def capture_rng(device):
    result = {"cpu": torch.get_rng_state(), "cuda": None, "cuda_device": None, "mps": None}
    if device.type == "cuda":
        result["cuda"] = torch.cuda.get_rng_state_all()
        result["cuda_device"] = device.index if device.index is not None else torch.cuda.current_device()
    if device.type == "mps":
        result["mps"] = torch.mps.get_rng_state()
    return cpu_state(result)


def restore_rng(saved, device):
    torch.set_rng_state(saved["cpu"])
    if device.type == "cuda" and saved.get("cuda") is not None:
        states = saved["cuda"]
        if len(states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(states)
        torch.cuda.set_rng_state(states[saved["cuda_device"]], device=device)
    if device.type == "mps" and saved.get("mps") is not None:
        torch.mps.set_rng_state(saved["mps"])


def compatible_data(saved, current):
    if saved["tokenizer"] != current["tokenizer"]:
        return False
    for split in ("train", "validation"):
        previous, present = saved[split], current[split]
        if len(present) < len(previous) or present[:len(previous)] != previous:
            return False
    return True


def resolve_checkpoint(output, resume):
    output = Path(output)
    if str(resume) == "latest":
        pointer = json.loads((output / "latest.json").read_text())
        return output / pointer["checkpoint"]
    path = Path(resume)
    if path.is_file() and path.name == "checkpoint.pt":
        return path.parent
    return path


def load_checkpoint(directory):
    directory = Path(directory)
    marker = json.loads((directory / "complete.json").read_text())
    if marker.get("format_version") != 1 or sha256(directory / "checkpoint.pt") != marker["sha256"]:
        raise ValueError("Checkpoint is incomplete or its checksum differs")
    return torch.load(directory / "checkpoint.pt", weights_only=True, map_location="cpu")


def remove_tree(path):
    def ignore_missing(function, missing_path, error):
        if not isinstance(error[1], FileNotFoundError):
            raise error[1]
    shutil.rmtree(path, onerror=ignore_missing)


def save_checkpoint(output, state, best=False, keep=3):
    if type(keep) is not int or keep < 1:
        raise ValueError("Checkpoint retention must be a positive integer")
    output = Path(output)
    root = output / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    step = state["global_step"]
    saved_at = time.time_ns()
    destination = root / f"step-{step:012d}-{uuid.uuid4().hex[:12]}"
    required = int(tensor_bytes(state) * 1.05) + 1048576
    if shutil.disk_usage(root).free < required:
        raise OSError(f"Insufficient checkpoint space: require approximately {required} free bytes")
    temporary = root / (".incomplete-" + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        with (temporary / "checkpoint.pt").open("xb") as stream:
            torch.save(cpu_state(state), stream)
            stream.flush()
            os.fsync(stream.fileno())
        atomic_json(temporary / "complete.json",
                    {"format_version": 1, "global_step": step, "tokens": state["tokens"],
                     "status": state["status"], "config_sha256": state["config_sha256"],
                     "data_sha256": state["data_sha256"], "saved_at_ns": saved_at,
                     "sha256": sha256(temporary / "checkpoint.pt")})
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            remove_tree(temporary)
    pointer = {"checkpoint": str(destination.relative_to(output)), "global_step": step}
    atomic_json(output / "latest.json", pointer)
    if best:
        atomic_json(output / "best.json", pointer)
    best_path = None
    if (output / "best.json").exists():
        best_path = output / json.loads((output / "best.json").read_text())["checkpoint"]
    completed = [path for path in root.glob("step-*") if (path / "complete.json").exists()]
    completed.sort(key=lambda path: json.loads((path / "complete.json").read_text())["saved_at_ns"])
    retained = set(completed[-keep:])
    if best_path is not None:
        retained.add(best_path)
    for path in completed:
        if path not in retained:
            remove_tree(path)
    return destination
