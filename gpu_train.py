"""Portable PicoLLM pretraining on immutable packed token shards."""
import argparse
from bisect import bisect_right
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import hashlib
import math
import mmap
from pathlib import Path
import re
import signal
import sys
import time

import torch

from course_model import ModelConfig
from gpu_model import GPUPicoLLM
from gpu_checkpoint import (atomic_json, capture_rng, compatible_data, fingerprint,
                            load_checkpoint, resolve_checkpoint, restore_rng,
                            save_checkpoint, sha256)


@dataclass
class TrainingConfig:
    steps: int = 1000
    batch_size: int = 1
    grad_accum: int = 1
    lr: float = 0.0003
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    seed: int = 7
    dtype: str = "bfloat16"
    grad_clip: float = 1.0
    validation_every: int = 100
    validation_batches: int = 8
    loss_chunk_size: int = 128
    log_every: int = 10
    checkpoint_keep: int = 3
    deterministic: bool = False

    def __post_init__(self):
        for name in ("steps", "batch_size", "grad_accum", "validation_every",
                     "validation_batches", "loss_chunk_size", "log_every", "checkpoint_keep"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.warmup_steps) is not int or not 0 <= self.warmup_steps <= self.steps:
            raise ValueError("warmup_steps must be between zero and steps")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("lr", "grad_clip"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not math.isfinite(self.min_lr_ratio) or not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be between zero and one")
        if self.dtype not in ("float32", "bfloat16"):
            raise ValueError("dtype must be float32 or bfloat16")
        if type(self.deterministic) is not bool:
            raise ValueError("deterministic must be a boolean")


DTYPES = {"uint16": (torch.uint16, 2), "<u2": (torch.uint16, 2),
          "uint32": (torch.uint32, 4), "<u4": (torch.uint32, 4),
          "int32": (torch.int32, 4), "<i4": (torch.int32, 4)}


class TokenStream:
    def __init__(self, records, directory, dtype, vocab_size):
        if not isinstance(records, list) or not records:
            raise ValueError("Each split requires at least one completed token shard")
        if sys.byteorder != "little":
            raise ValueError("Raw little-endian token shards require a little-endian host")
        self.shards, self.ends, self.fingerprints = [], [], []
        total = 0
        try:
            for record in records:
                if not isinstance(record, dict):
                    raise ValueError("Token shard records must be objects")
                count = record.get("tokens")
                if type(count) is not int or count < 1:
                    raise ValueError("Shard tokens must be a positive integer")
                token_dtype = record.get("dtype", dtype)
                if token_dtype not in DTYPES:
                    raise ValueError("Require uint16, uint32 or int32 little-endian token files")
                tensor_dtype, item_size = DTYPES[token_dtype]
                raw_path, digest = record.get("path"), record.get("sha256")
                if not isinstance(raw_path, str) or not raw_path:
                    raise ValueError("Shard path must be a nonempty string")
                if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
                    raise ValueError("Shard requires its lowercase SHA256")
                path = Path(raw_path)
                path = path if path.is_absolute() else directory / path
                if path.stat().st_size != count * item_size or sha256(path) != digest:
                    raise ValueError(f"Token file size or checksum differs: {path}")
                stream = path.open("rb")
                try:
                    mapped = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_COPY)
                except BaseException:
                    stream.close()
                    raise
                self.shards.append((stream, mapped, tensor_dtype, item_size))
                for start in range(0, count, 1048576):
                    values = torch.frombuffer(mapped, dtype=tensor_dtype,
                                              count=min(1048576, count - start),
                                              offset=start * item_size).long()
                    if int(values.min()) < 0 or int(values.max()) >= vocab_size:
                        raise ValueError(f"Token ID outside model vocabulary: {path}")
                    del values
                total += count
                self.ends.append(total)
                self.fingerprints.append({"tokens": count, "sha256": digest,
                                          "dtype": str(tensor_dtype).removeprefix("torch.")})
        except BaseException:
            self.close()
            raise
        self.tokens = total

    def read(self, start, length):
        if start < 0 or length < 1 or start + length > self.tokens:
            raise ValueError("Packed token stream exhausted; wrapping is forbidden")
        result = torch.empty(length, dtype=torch.long)
        filled = 0
        while filled < length:
            index = bisect_right(self.ends, start)
            previous = self.ends[index - 1] if index else 0
            count = min(length - filled, self.ends[index] - start)
            _, mapped, dtype, size = self.shards[index]
            values = torch.frombuffer(mapped, dtype=dtype, count=count,
                                      offset=(start - previous) * size).long()
            result[filled:filled + count] = values
            del values
            start += count
            filled += count
        return result

    def batch(self, cursor, batch_size, context, device):
        values = self.read(cursor, batch_size * context + 1)
        inputs = torch.stack([values[index * context:index * context + context]
                              for index in range(batch_size)])
        targets = torch.stack([values[index * context + 1:index * context + context + 1]
                               for index in range(batch_size)])
        return inputs.to(device), targets.to(device), cursor + batch_size * context

    def close(self):
        for stream, mapped, _, _ in getattr(self, "shards", []):
            mapped.close()
            stream.close()
        self.shards = []


def read_data(path, vocab_size):
    path = Path(path)
    manifest_bytes = path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("format_version") != 1 or not isinstance(manifest.get("tokenizer"), dict):
        raise ValueError("Require format_version 1 and tokenizer metadata")
    train_records = manifest.get("train")
    validation_records = manifest.get("val", manifest.get("validation"))
    if isinstance(train_records, dict):
        train_records = [train_records]
    if isinstance(validation_records, dict):
        validation_records = [validation_records]
    dtype = manifest.get("dtype", "uint16")
    train = TokenStream(train_records, path.parent, dtype, vocab_size)
    try:
        validation = TokenStream(validation_records, path.parent, dtype, vocab_size)
    except BaseException:
        train.close()
        raise
    identities = {"tokenizer": manifest["tokenizer"], "train": train.fingerprints,
                  "validation": validation.fingerprints}
    return train, validation, identities, hashlib.sha256(manifest_bytes).hexdigest()


def learning_rate(step, config):
    if step < config.warmup_steps:
        return config.lr * (step + 1) / config.warmup_steps
    fraction = (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps - 1)
    return config.lr * (config.min_lr_ratio + (1 - config.min_lr_ratio) *
                        0.5 * (1 + math.cos(math.pi * min(1.0, fraction))))


def autocast(device, dtype):
    if dtype == "float32":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def validate(model, stream, config, device):
    count = min(config.validation_batches,
                (stream.tokens - 1) // (config.batch_size * model.config.context))
    if count < 1:
        raise ValueError("Validation needs at least one complete packed batch")
    model.eval()
    losses, cursor = 0.0, 0
    for _ in range(count):
        inputs, targets, cursor = stream.batch(cursor, config.batch_size, model.config.context, device)
        with autocast(device, config.dtype):
            loss = model.loss(inputs, targets, config.loss_chunk_size)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite validation loss")
        losses += float(loss)
    return {"nll": losses / count, "batches": count,
            "tokens": count * config.batch_size * model.config.context}


def optimizer_for(model, config, device):
    decay, scales = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else scales).append(parameter)
    kwargs = {"lr": config.lr, "betas": (0.9, 0.95), "foreach": False, "fused": False}
    if device.type == "cuda":
        kwargs["fused"] = True
    return torch.optim.AdamW([{"params": decay, "weight_decay": config.weight_decay},
                              {"params": scales, "weight_decay": 0.0}], **kwargs)


def save_interval(value):
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(s|seconds|steps)?", value)
    if not match or float(match[1]) <= 0:
        raise ValueError("save-every requires positive seconds (300s) or integer steps (100steps)")
    amount, unit = float(match[1]), match[2] or "steps"
    if unit == "steps" and not amount.is_integer():
        raise ValueError("Checkpoint step interval must be an integer")
    return amount, "seconds" if unit in ("s", "seconds") else "steps"


def train(config_path, data_path, output, device_name="cuda", resume=None,
          max_steps=None, save_every="300s", stop_requested=None):
    config_bytes = Path(config_path).read_bytes()
    document = json.loads(config_bytes)
    config_file_sha = hashlib.sha256(config_bytes).hexdigest()
    model_config = ModelConfig(**document["model"])
    training = TrainingConfig(**document.get("training", {}))
    limit = training.steps if max_steps is None else min(max_steps, training.steps)
    if type(limit) is not int or limit < 1:
        raise ValueError("max-steps must be a positive global optimizer-step limit")
    interval, interval_unit = save_interval(save_every)
    device = torch.device(device_name)
    if device.type not in ("cuda", "cpu", "mps"):
        raise ValueError("Device must be cuda, cpu or mps")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device unavailable")
        if training.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Selected CUDA runtime does not support bfloat16")
        device = torch.device("cuda", 0 if device.index is None else device.index)
        torch.cuda.set_device(device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS device unavailable")
    if device.type == "cpu":
        torch.set_num_threads(1)
    torch.use_deterministic_algorithms(device.type == "cpu" or training.deterministic)
    torch.manual_seed(training.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(training.seed)
    if device.type == "mps":
        torch.mps.manual_seed(training.seed)
    train_stream, validation_stream, data_identity, manifest_sha = read_data(data_path, model_config.vocab_size)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if resume is None and (output / "latest.json").exists():
        train_stream.close(); validation_stream.close()
        raise ValueError("Existing experiment requires --resume latest or a new output")
    config_sha, data_sha = fingerprint(document), fingerprint(data_identity)
    try:
        model = GPUPicoLLM(model_config).to(device)
        optimizer = optimizer_for(model, training, device)
        step, tokens, cursor, best_validation = 0, 0, 0, None
        scheduler = {"steps": training.steps, "warmup_steps": training.warmup_steps,
                     "lr": training.lr, "min_lr_ratio": training.min_lr_ratio,
                     "completed_steps": 0, "next_lr": learning_rate(0, training)}
        if resume is not None:
            saved = load_checkpoint(resolve_checkpoint(output, resume))
            if saved["config_sha256"] != config_sha or not compatible_data(saved["data_identity"], data_identity):
                raise ValueError("Resume config, tokenizer or immutable token-shard prefix differs")
            model.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            for group in optimizer.param_groups:
                group["fused"] = device.type == "cuda"
                group["foreach"] = False
            for values in optimizer.state.values():
                if isinstance(values.get("step"), torch.Tensor):
                    values["step"] = values["step"].to(device if device.type == "cuda" else "cpu")
            step, tokens, cursor = saved["global_step"], saved["tokens"], saved["cursor"]["train"]
            scheduler = saved["scheduler"]
            best_validation = saved["best_validation"]
            if limit < step:
                raise ValueError("max-steps precedes the saved global step")
            if scheduler["completed_steps"] != step or tokens != cursor:
                raise ValueError("Checkpoint optimizer boundary counters disagree")
            restore_rng(saved["rng"], device)
        initial_step = step
        started = time.perf_counter()
        last_saved, saved_time = step, started
        validation = None
        status = "running"

        with (output / "metrics.jsonl").open("a", encoding="utf-8") as log:
            def emit(value):
                value["time_unix"] = time.time()
                log.write(json.dumps(value, allow_nan=False, sort_keys=True) + "\n")
                log.flush()
                return value

            def checkpoint(current_status, is_best=False):
                state = {"format_version": 1, "model": model.state_dict(),
                         "optimizer": optimizer.state_dict(), "scheduler": scheduler,
                         "global_step": step, "tokens": tokens, "cursor": {"train": cursor},
                         "rng": capture_rng(device), "config": document,
                         "resolved_config": {"model": asdict(model_config), "training": asdict(training)},
                         "config_sha256": config_sha, "config_file_sha256": config_file_sha,
                         "data_sha256": data_sha, "data_manifest_sha256": manifest_sha,
                         "data_identity": data_identity, "best_validation": best_validation,
                         "validation": validation, "status": current_status,
                         "source_sha256": {name: sha256(Path(__file__).with_name(name)) for name in
                                           ("gpu_train.py", "gpu_model.py", "gpu_checkpoint.py", "course_model.py")},
                         "environment": {"torch": str(torch.__version__), "device": str(device),
                                         "dtype": training.dtype}}
                directory = save_checkpoint(output, state, best=is_best, keep=training.checkpoint_keep)
                emit({"event": "checkpoint", "global_step": step, "tokens": tokens,
                      "status": current_status, "path": str(directory)})
                return directory

            emit({"event": "resume" if resume else "start", "global_step": step,
                  "tokens": tokens, "config_sha256": config_sha, "data_sha256": data_sha,
                  "train_tokens_available": train_stream.tokens,
                  "parameters": sum(parameter.numel() for parameter in model.parameters()),
                  "device": str(device), "dtype": training.dtype})
            tokens_per_step = training.batch_size * training.grad_accum * model_config.context
            try:
                while step < limit:
                    if stop_requested is not None and stop_requested():
                        status = "paused_signal"
                        break
                    if cursor + tokens_per_step + 1 > train_stream.tokens:
                        status = "paused_data"
                        break
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    for group in optimizer.param_groups:
                        group["lr"] = learning_rate(step, training)
                    began = time.perf_counter()
                    accumulated_loss = 0.0
                    for _ in range(training.grad_accum):
                        inputs, targets, cursor = train_stream.batch(
                            cursor, training.batch_size, model_config.context, device)
                        with autocast(device, training.dtype):
                            loss = model.loss(inputs, targets, training.loss_chunk_size)
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError("Nonfinite training loss; optimizer update refused")
                        (loss / training.grad_accum).backward()
                        accumulated_loss += float(loss.detach()) / training.grad_accum
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training.grad_clip,
                                                         error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    tokens += tokens_per_step
                    scheduler["completed_steps"] = step
                    scheduler["next_lr"] = learning_rate(step, training)
                    synchronize(device)
                    elapsed = time.perf_counter() - began
                    record = emit({"event": "train", "global_step": step, "loss": accumulated_loss,
                                   "tokens": tokens, "tokens_per_second": tokens_per_step / elapsed,
                                   "step_seconds": elapsed, "lr": optimizer.param_groups[0]["lr"],
                                   "gradient_norm": float(norm)})
                    is_best = False
                    if step % training.validation_every == 0:
                        validation = validate(model, validation_stream, training, device)
                        emit({"event": "validation", "global_step": step, "tokens": tokens, **validation})
                        is_best = best_validation is None or validation["nll"] < best_validation
                        if is_best:
                            best_validation = validation["nll"]
                    due = (step - last_saved >= interval if interval_unit == "steps"
                           else time.perf_counter() - saved_time >= interval)
                    if due or is_best:
                        checkpoint("running", is_best)
                        last_saved, saved_time = step, time.perf_counter()
                    if step % training.log_every == 0:
                        print(json.dumps(record, sort_keys=True), flush=True)
                if status == "running":
                    status = "complete" if step >= training.steps else "paused_limit"
                validation = validate(model, validation_stream, training, device)
                is_best = best_validation is None or validation["nll"] < best_validation
                if is_best:
                    best_validation = validation["nll"]
                emit({"event": "validation", "global_step": step, "tokens": tokens, **validation})
                directory = checkpoint(status, is_best)
            except (FloatingPointError, RuntimeError, OSError) as error:
                emit({"event": "failure", "global_step": step, "tokens": tokens, "error": str(error)})
                raise
        report = {"status": status, "global_step": step, "tokens": tokens,
                  "updates_this_run": step - initial_step,
                  "run_seconds": time.perf_counter() - started, "validation": validation,
                  "checkpoint": str(directory), "config_sha256": config_sha,
                  "data_sha256": data_sha, "train_tokens_available": train_stream.tokens}
        atomic_json(output / "run-report.json", report)
        return report
    finally:
        train_stream.close()
        validation_stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--save-every", default="300s", help="300s or 100steps")
    args = parser.parse_args()
    requested = [False]

    def request_stop(signum, frame):
        requested[0] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    report = train(args.config, args.data, args.output, args.device, args.resume,
                   args.max_steps, args.save_every, lambda: requested[0])
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
