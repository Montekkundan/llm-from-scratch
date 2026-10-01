"""Assistant-only full-parameter SFT of a scratch PicoLLM checkpoint."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import signal
import time

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from chat_data import NEUTRAL_SYSTEM, SYSTEM
from course_model import ModelConfig
from gpu_checkpoint import (atomic_json, capture_rng, fingerprint, load_checkpoint,
                            resolve_checkpoint, restore_rng, save_checkpoint, sha256)
from gpu_model import GPUPicoLLM
from gpu_train import TrainingConfig, autocast, learning_rate, optimizer_for, save_interval, synchronize
from scratch_chat import checkpoint_directory, resolve_device


def load_prepared(path, tokenizer_metadata, model_context, vocab_size):
    path = Path(path)
    path = path / "manifest.json" if path.is_dir() else path
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema") != 1 or manifest.get("experiment_type") != "scratch":
        raise ValueError("Require a scratch-specific chat_data manifest, not foundation identity data")
    if manifest.get("model_kind") != "scratch" or manifest.get("pretrained_model_weights") is not False:
        raise ValueError("Prepared provenance must identify scratch weights")
    if manifest.get("system") != SYSTEM or manifest.get("identity_training_system") != NEUTRAL_SYSTEM:
        raise ValueError("Prepared course and neutral identity systems differ")
    if (manifest.get("model_id"), manifest.get("tokenizer_revision")) != (
            tokenizer_metadata["model"], tokenizer_metadata["revision"]):
        raise ValueError("Prepared tokenizer differs from the scratch pretraining tokenizer")
    if manifest.get("model_revision") != tokenizer_metadata["revision"]:
        raise ValueError("Prepared tokenizer model revision differs")
    context = manifest.get("context")
    if type(context) is not int or not 2 <= context <= model_context:
        raise ValueError("Prepared context must fit the scratch model; clip during chat_data preparation")
    template = manifest.get("template_sha256")
    if not isinstance(template, str) or len(template) != 64:
        raise ValueError("Prepared data requires a native chat-template SHA256")
    rows = {}
    for split in ("train", "heldout"):
        file = path.parent / (split + ".jsonl")
        metadata = manifest["splits"][split]
        if sha256(file) != metadata["sha256"]:
            raise ValueError(f"Prepared {split} checksum differs")
        values = [json.loads(line) for line in file.read_text().splitlines() if line.strip()]
        if not values:
            raise ValueError(f"Prepared {split} is empty")
        for row in values:
            ids, labels, mask = row.get("input_ids"), row.get("labels"), row.get("attention_mask")
            if not isinstance(ids, list) or not 2 <= len(ids) <= context:
                raise ValueError("Every conversation must fit the prepared context")
            if not isinstance(labels, list) or len(labels) != len(ids) or mask != [1] * len(ids):
                raise ValueError("Require unpadded equal-length IDs, labels and all-one attention masks")
            if any(type(value) is not int or not 0 <= value < vocab_size for value in ids):
                raise ValueError("Prepared token IDs must fit the scratch vocabulary")
            if any(type(label) is not int or label not in (-100, token) for label, token in zip(labels, ids)):
                raise ValueError("Assistant labels must be -100 or the original token ID")
            if labels[0] != -100 or not any(label != -100 for label in labels[1:]):
                raise ValueError("Each conversation requires assistant targets after the causal shift")
        counts = {"examples": len(values), "input_tokens": sum(len(row["input_ids"]) for row in values),
                  "supervised_tokens": sum(sum(label != -100 for label in row["labels"][1:]) for row in values)}
        if any(metadata.get(name) != count for name, count in counts.items()):
            raise ValueError(f"Prepared {split} counts differ")
        rows[split] = values
    training_hashes = {fingerprint(row) for row in rows["train"]}
    if any(fingerprint(row) in training_hashes for row in rows["heldout"]):
        raise ValueError("Training and held-out token conversations overlap")
    identity = {"tokenizer": tokenizer_metadata, "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                "train": manifest["splits"]["train"], "validation": manifest["splits"]["heldout"],
                "template_sha256": template, "experiment_type": "scratch"}
    return rows["train"], rows["heldout"], manifest, identity


def collate(rows, device):
    width = max(len(row["input_ids"]) for row in rows)
    # Causal right padding only changes ignored future positions.
    ids = [row["input_ids"] + [0] * (width - len(row["input_ids"])) for row in rows]
    labels = [row["labels"] + [-100] * (width - len(row["labels"])) for row in rows]
    return torch.tensor(ids, dtype=torch.long, device=device), torch.tensor(labels, dtype=torch.long, device=device)


def assistant_loss_sum(model, ids, labels, chunk_size=128):
    if ids.shape != labels.shape or ids.dtype != torch.long or labels.dtype != torch.long:
        raise ValueError("Require equal-shaped torch.long conversation IDs and labels")
    targets = labels[:, 1:]
    count = int(targets.ne(-100).sum())
    if count < 1 or chunk_size < 1:
        raise ValueError("Require assistant targets and a positive loss chunk size")
    hidden = model.hidden_states(ids[:, :-1])

    def loss_chunk(values, expected):
        logits = model.lm_head(values).float()
        return F.cross_entropy(logits.reshape(-1, model.config.vocab_size), expected.reshape(-1),
                               ignore_index=-100, reduction="sum")

    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, targets.shape[1], chunk_size):
        values, expected = hidden[:, start:start + chunk_size], targets[:, start:start + chunk_size]
        value = checkpoint(loss_chunk, values, expected, use_reentrant=False,
                           preserve_rng_state=False) if torch.is_grad_enabled() else loss_chunk(values, expected)
        total = total + value
    return total, count


@torch.inference_mode()
def validate_assistant(model, rows, config, device):
    model.eval()
    numerator, denominator, inputs, batches = 0.0, 0, 0, 0
    for start in range(0, min(len(rows), config.batch_size * config.validation_batches), config.batch_size):
        batch = rows[start:start + config.batch_size]
        ids, labels = collate(batch, device)
        with autocast(device, config.dtype):
            loss, count = assistant_loss_sum(model, ids, labels, config.loss_chunk_size)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite held-out assistant loss")
        numerator += float(loss)
        denominator += count
        inputs += sum(len(row["input_ids"]) for row in batch)
        batches += 1
    return {"assistant_token_loss": numerator / denominator, "supervised_tokens": denominator,
            "input_tokens": inputs, "batches": batches,
            "scope": "Token-weighted fixed held-out prefix, including assistant EOS; not a human chat-quality score"}


def train_sft(config_path, data_path, output, base_checkpoint=None, device_name="cuda",
              resume=None, max_steps=None, save_every="300s", stop_requested=None):
    document = json.loads(Path(config_path).read_text())
    epochs = document.get("epochs", 1)
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    output = Path(output)
    if resume is None and (output / "latest.json").exists():
        raise ValueError("Existing SFT experiment requires --resume latest or a fresh output")
    saved = load_checkpoint(resolve_checkpoint(output, resume)) if resume is not None else None
    if saved is not None:
        if saved.get("stage") != "scratch_sft":
            raise ValueError("Resume requires a scratch SFT checkpoint")
        base_sha = saved["base_checkpoint_sha256"]
        model_config = ModelConfig(**saved["resolved_config"]["model"])
        tokenizer_metadata = saved["data_identity"]["tokenizer"]
        if base_checkpoint is not None:
            directory = checkpoint_directory(base_checkpoint)
            if sha256(directory / "checkpoint.pt") != base_sha:
                raise ValueError("Resume base checkpoint differs")
    else:
        if base_checkpoint is None:
            raise ValueError("Fresh SFT requires --base-checkpoint")
        directory = checkpoint_directory(base_checkpoint)
        base = load_checkpoint(directory)
        base_sha = sha256(directory / "checkpoint.pt")
        model_config = ModelConfig(**base["resolved_config"]["model"])
        tokenizer_metadata = base["data_identity"]["tokenizer"]
    rows, heldout, manifest, identity = load_prepared(data_path, tokenizer_metadata,
                                                     model_config.context, model_config.vocab_size)
    values = dict(document.get("training", {}))
    for name in ("batch_size", "grad_accum"):
        if type(values.get(name, 1)) is not int or values.get(name, 1) < 1:
            raise ValueError("batch_size and grad_accum must be positive integers")
    examples_per_update = values.get("batch_size", 1) * values.get("grad_accum", 1)
    values.setdefault("steps", math.ceil(len(rows) * epochs / examples_per_update))
    values.setdefault("warmup_steps", min(100, values["steps"]))
    training = TrainingConfig(**values)
    limit = training.steps if max_steps is None else min(training.steps, max_steps)
    if type(limit) is not int or limit < 1:
        raise ValueError("max-steps must be a positive global optimizer-step cap")
    interval, unit = save_interval(save_every)
    device = resolve_device(device_name, training.dtype)
    torch.use_deterministic_algorithms(device.type == "cpu" or training.deterministic)
    torch.manual_seed(training.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(training.seed)
    elif device.type == "mps":
        torch.mps.manual_seed(training.seed)
    model = GPUPicoLLM(model_config).to(device)
    model.load_state_dict((saved if saved is not None else base)["model"], strict=True)
    if saved is None:
        del base
    optimizer = optimizer_for(model, training, device)
    step, cursor, input_tokens, supervised_tokens, best_validation = 0, 0, 0, 0, None
    config_sha, data_sha = fingerprint(document), fingerprint(identity)
    scheduler = {"steps": training.steps, "warmup_steps": training.warmup_steps,
                 "lr": training.lr, "min_lr_ratio": training.min_lr_ratio,
                 "completed_steps": 0, "next_lr": learning_rate(0, training)}
    if saved is not None:
        if saved["config_sha256"] != config_sha or saved["data_sha256"] != data_sha:
            raise ValueError("Resume SFT configuration or prepared conversations differ")
        optimizer.load_state_dict(saved["optimizer"])
        for group in optimizer.param_groups:
            group["fused"], group["foreach"] = device.type == "cuda", False
        for state in optimizer.state.values():
            if isinstance(state.get("step"), torch.Tensor):
                state["step"] = state["step"].to(device if device.type == "cuda" else "cpu")
        step, cursor = saved["global_step"], saved["cursor"]["examples"]
        input_tokens, supervised_tokens = saved["tokens"], saved["supervised_tokens"]
        best_validation, scheduler = saved["best_validation"], saved["scheduler"]
        if limit < step or scheduler["completed_steps"] != step:
            raise ValueError("Resume optimizer counters or global step cap differ")
        restore_rng(saved["rng"], device)
        del saved
    total_examples = len(rows) * epochs
    if not 0 <= cursor <= total_examples or step > training.steps:
        raise ValueError("SFT checkpoint cursor is outside the declared budget")
    initial_step, started = step, time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    last_saved, saved_time, validation, status = step, started, None, "running"
    with (output / "metrics.jsonl").open("a") as log:
        def emit(value):
            value["time_unix"] = time.time()
            log.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
            log.flush()
            return value

        def save(current_status, is_best=False):
            state = {"format_version": 1, "stage": "scratch_sft", "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler, "global_step": step,
                     "tokens": input_tokens, "supervised_tokens": supervised_tokens,
                     "cursor": {"examples": cursor, "epoch": cursor // len(rows), "row": cursor % len(rows)},
                     "rng": capture_rng(device), "config": {"model": asdict(model_config),
                         "training": asdict(training), "epochs": epochs}, "raw_config": document,
                     "resolved_config": {"model": asdict(model_config), "training": asdict(training)},
                     "config_sha256": config_sha, "data_sha256": data_sha, "data_identity": identity,
                     "base_checkpoint_sha256": base_sha, "optimizer_initialization": "fresh_for_sft",
                     "chat_template_sha256": manifest["template_sha256"], "best_validation": best_validation,
                     "validation": validation, "status": current_status,
                     "source_sha256": {name: sha256(Path(__file__).with_name(name)) for name in
                         ("scratch_sft.py", "scratch_chat.py", "gpu_model.py", "gpu_train.py", "gpu_checkpoint.py", "course_model.py", "chat_data.py")},
                     "environment": {"torch": str(torch.__version__), "device": str(device), "dtype": training.dtype}}
            directory = save_checkpoint(output, state, best=is_best, keep=training.checkpoint_keep)
            emit({"event": "checkpoint", "global_step": step, "tokens": input_tokens,
                  "status": current_status, "path": str(directory)})
            return directory

        emit({"event": "resume" if resume else "start", "global_step": step, "tokens": input_tokens,
              "supervised_tokens": supervised_tokens, "base_checkpoint_sha256": base_sha,
              "config_sha256": config_sha, "data_sha256": data_sha, "epochs": epochs,
              "training_examples": len(rows), "optimizer_initialization": "restored" if resume else "fresh_for_sft"})
        try:
            while step < limit and cursor < total_examples:
                if stop_requested is not None and stop_requested():
                    status = "paused_signal"
                    break
                count = min(training.batch_size * training.grad_accum, total_examples - cursor)
                update_rows = [rows[index % len(rows)] for index in range(cursor, cursor + count)]
                denominator = sum(sum(label != -100 for label in row["labels"][1:]) for row in update_rows)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate(step, training)
                began, numerator = time.perf_counter(), 0.0
                for start in range(0, count, training.batch_size):
                    ids, labels = collate(update_rows[start:start + training.batch_size], device)
                    with autocast(device, training.dtype):
                        loss, _ = assistant_loss_sum(model, ids, labels, training.loss_chunk_size)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Nonfinite SFT loss; optimizer update refused")
                    (loss / denominator).backward()
                    numerator += float(loss.detach())
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training.grad_clip, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                cursor += count
                input_tokens += sum(len(row["input_ids"]) for row in update_rows)
                supervised_tokens += denominator
                scheduler["completed_steps"], scheduler["next_lr"] = step, learning_rate(step, training)
                synchronize(device)
                record = emit({"event": "train", "global_step": step, "assistant_token_loss": numerator / denominator,
                    "tokens": input_tokens, "supervised_tokens": supervised_tokens, "examples": cursor,
                    "supervised_tokens_per_second": denominator / (time.perf_counter() - began),
                    "gradient_norm": float(norm), "lr": optimizer.param_groups[0]["lr"]})
                is_best = False
                if step % training.validation_every == 0:
                    validation = validate_assistant(model, heldout, training, device)
                    emit({"event": "validation", "global_step": step, **validation})
                    is_best = best_validation is None or validation["assistant_token_loss"] < best_validation
                    if is_best:
                        best_validation = validation["assistant_token_loss"]
                due = step - last_saved >= interval if unit == "steps" else time.perf_counter() - saved_time >= interval
                if due or is_best:
                    save("running", is_best)
                    last_saved, saved_time = step, time.perf_counter()
                if step % training.log_every == 0:
                    print(json.dumps(record, sort_keys=True), flush=True)
            if status == "running":
                status = "complete" if cursor >= total_examples or step >= training.steps else "paused_limit"
            validation = validate_assistant(model, heldout, training, device)
            is_best = best_validation is None or validation["assistant_token_loss"] < best_validation
            if is_best:
                best_validation = validation["assistant_token_loss"]
            emit({"event": "validation", "global_step": step, **validation})
            directory = save(status, is_best)
        except (RuntimeError, FloatingPointError, OSError) as error:
            emit({"event": "failure", "global_step": step, "tokens": input_tokens, "error": str(error)})
            raise
    report = {"status": status, "global_step": step, "input_tokens": input_tokens,
              "supervised_tokens": supervised_tokens, "examples_seen": cursor,
              "updates_this_run": step - initial_step, "run_seconds": time.perf_counter() - started,
              "checkpoint": str(directory), "base_checkpoint_sha256": base_sha,
              "validation": validation, "config_sha256": config_sha, "data_sha256": data_sha,
              "finished_epochs": cursor // len(rows), "declared_epochs": epochs,
              "completion_reason": "epoch_budget" if cursor >= total_examples else
                                   "step_budget" if step >= training.steps else "paused"}
    atomic_json(output / "run-report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True, help="Scratch chat_data manifest or directory")
    parser.add_argument("--base-checkpoint", help="Complete pretraining checkpoint; required for fresh SFT")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--save-every", default="300s")
    args = parser.parse_args()
    requested = [False]

    def stop(signum, frame):
        requested[0] = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    report = train_sft(args.config, args.data, args.output, args.base_checkpoint,
                       args.device, args.resume, args.max_steps, args.save_every, lambda: requested[0])
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
