"""PicoLLM base training from random weights: synthetic or explicit JSONL documents.

Run from this directory: python train.py --output runs/picollm-base --steps 160
This synthetic distribution is not a language-competence benchmark.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import torch
from torch.nn import functional as F

from course_model import PicoLLM, ModelConfig
from tokenizer import BOS, EOS, PAD, IGNORE, TOKENIZER, encode

def corpus():
    """A declared combinatorial holdout: all words occur in both splits."""
    rows = []
    for i, color in enumerate(("red", "blue", "green", "gold")):
        for j, animal in enumerate(("fox", "owl", "cat", "elk")):
            for k, action in enumerate(("rests", "runs", "waits", "jumps")):
                text = f"{color} {animal} {action}.\n"
                rows.append({"id": f"{i}-{j}-{k}", "text": text,
                             "split": "validation" if (i + j + k) % 5 == 0 else "train",
                             "sha256": hashlib.sha256(text.encode()).hexdigest()})
    assert len({r["sha256"] for r in rows}) == len(rows)
    return rows


def read_documents(train_file, validation_file, test_file=None):
    """Require caller-assigned document splits; reject duplicate content/groups."""
    rows, ids, hashes, groups = [], set(), set(), {}
    paths = [("train", train_file), ("validation", validation_file)]
    if test_file is not None:
        paths.append(("test", test_file))
    for split, path in paths:
        count = 0
        for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError(f"{path}:{line_number}: require nonempty text")
            identifier = row.get("id")
            group = row.get("group", identifier)
            if not isinstance(identifier, str) or not identifier or identifier in ids:
                raise ValueError("Document IDs must be nonempty and globally unique")
            if not isinstance(group, str) or not group:
                raise ValueError("Document group must be a nonempty string")
            digest = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
            if digest in hashes or (group in groups and groups[group] != split):
                raise ValueError("Duplicate document content or group crosses the declared splits")
            ids.add(identifier); hashes.add(digest); groups[group] = split
            rows.append({**row, "id": identifier, "group": group, "split": split, "sha256": digest})
            count += 1
        if not count:
            raise ValueError(f"{split} must contain documents")
    return rows


def document_windows(texts, context):
    # Each target occurs once. A fresh window retains its immediately preceding
    # token, resets position to zero, and never crosses a document boundary.
    return [(i, start) for i, text in enumerate(texts)
            for start in range(0, len(encode(text)) - 1, context)]


def batch(texts, context=128, starts=None):
    sequences = [encode(text) for text in texts]
    if starts is not None:
        if len(starts) != len(texts):
            raise ValueError("One window start per document is required")
        sequences = [seq[start:start + context + 1] for seq, start in zip(sequences, starts)]
        if any(start < 0 for start in starts) or any(len(seq) < 2 for seq in sequences):
            raise ValueError("Window must contain an input and a target")
    width = max(len(seq) - 1 for seq in sequences)
    if width > context:
        raise ValueError("Use explicit document window starts for long text")
    inputs = torch.full((len(texts), width), PAD, dtype=torch.long)
    targets = torch.full_like(inputs, IGNORE)
    for row, seq in enumerate(sequences):
        inputs[row, :len(seq) - 1] = torch.tensor(seq[:-1])
        targets[row, :len(seq) - 1] = torch.tensor(seq[1:])
    return inputs, targets


@torch.inference_mode()
def measure(model, texts, batch_size=8):
    """Score each byte/EOS once; reset context only at explicit window boundaries."""
    was_training = model.training
    model.eval()
    summed, targets, byte_nll, byte_count = 0.0, 0, 0.0, 0
    documents = [{"nll_sum": 0.0, "targets": 0} for _ in texts]
    windows = document_windows(texts, model.config.context)
    try:
        for start in range(0, len(windows), batch_size):
            selected = windows[start:start + batch_size]
            x, y = batch([texts[i] for i, _ in selected], model.config.context,
                         [offset for _, offset in selected])
            losses = F.cross_entropy(model(x).reshape(-1, model.config.vocab_size),
                                     y.reshape(-1), ignore_index=IGNORE,
                                     reduction="none").reshape_as(y)
            valid, bytes_mask = y != IGNORE, (y >= 0) & (y < 256)
            summed += losses[valid].sum().item()
            targets += int(valid.sum())
            byte_nll += losses[bytes_mask].sum().item()
            byte_count += int(bytes_mask.sum())
            for r, (document, _) in enumerate(selected):
                documents[document]["nll_sum"] += losses[r][valid[r]].sum().item()
                documents[document]["targets"] += int(valid[r].sum())
    finally:
        model.train(was_training)
    if not targets or not byte_count:
        raise ValueError("evaluation requires nonempty text and targets")
    return {"nll": summed / targets, "nll_sum": summed, "targets": targets,
            "perplexity": math.exp(summed / targets),
            "bits_per_byte": byte_nll / (byte_count * math.log(2)),
            "byte_nll_sum": byte_nll, "bytes": byte_count, "documents": documents}


def optimizer_for(model, lr):
    matrices, scales = [], []
    for parameter in model.parameters():
        (matrices if parameter.ndim >= 2 else scales).append(parameter)
    return torch.optim.AdamW([
        {"params": matrices, "weight_decay": 0.01},
        {"params": scales, "weight_decay": 0.0},
    ], lr=lr, betas=(0.9, 0.999), foreach=False)


def learning_rate(step, total, peak=0.003, warmup=10):
    warmup = min(warmup, total)
    if step < warmup:
        return peak * (step + 1) / warmup
    fraction = (step - warmup) / max(1, total - warmup - 1)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * fraction)))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-file", type=Path, help="JSONL: one original document per row")
    parser.add_argument("--validation-file", type=Path)
    parser.add_argument("--test-file", type=Path, help="untouched final JSONL holdout; never scored during training")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--ff-width", type=int, default=176)
    args = parser.parse_args()
    if bool(args.train_file) != bool(args.validation_file):
        parser.error("Provide both --train-file and --validation-file")
    if args.test_file and not args.train_file:
        parser.error("--test-file requires --train-file and --validation-file")
    if args.steps < 1 or args.batch_size < 1:
        parser.error("steps and batch size must be positive")
    stop = args.stop_after if args.stop_after is not None else args.steps
    if not 0 <= stop <= args.steps:
        parser.error("stop-after must be between 0 and steps")
    # Refuse to overwrite a previous experiment.
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed + 1)
    cfg = ModelConfig(width=args.width, heads=args.heads, layers=args.layers,
                      context=args.context, ff_width=args.ff_width)
    model = PicoLLM(cfg)
    optimizer = optimizer_for(model, 0.003)
    rows = read_documents(args.train_file, args.validation_file, args.test_file) if args.train_file else corpus()
    train = [r["text"] for r in rows if r["split"] == "train"]
    valid = [r["text"] for r in rows if r["split"] == "validation"]
    assert train and valid and not set(train) & set(valid)
    windows = document_windows(train, cfg.context)
    signature = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    settings = {"steps": args.steps, "seed": args.seed, "batch_size": args.batch_size,
                "corpus_sha256": signature, "model": asdict(cfg)}
    start, history = 0, []
    if args.resume:
        saved = torch.load(args.resume, weights_only=True, map_location="cpu")
        if saved["settings"] != settings:
            raise ValueError("Resume settings differ from the saved experiment")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        rng.set_state(saved["sampler_rng"])
        start, history = saved["completed_steps"], saved["history"]
        if stop <= start:
            raise ValueError("Resume must perform at least one additional step")
    initial = {"train": measure(model, train), "validation": measure(model, valid)}
    trained_targets = sum(item["targets"] for item in history)
    began = time.perf_counter()
    for step in range(start, stop):
        model.train()
        indices = torch.randint(len(windows), (args.batch_size,), generator=rng).tolist()
        selected = [windows[index] for index in indices]
        x, y = batch([train[i] for i, _ in selected], cfg.context, [start for _, start in selected])
        n = int((y != IGNORE).sum())
        lr = learning_rate(step, args.steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x).reshape(-1, cfg.vocab_size), y.reshape(-1),
                               ignore_index=IGNORE, reduction="sum") / n
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at update {step + 1}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                             error_if_nonfinite=True)
        optimizer.step()
        trained_targets += n
        history.append({"step": step + 1, "loss": loss.item(), "targets": n,
                        "lr": lr, "gradient_norm": norm.item(), "batch_indices": indices})
    elapsed = time.perf_counter() - began
    final = {"train": measure(model, train), "validation": measure(model, valid)}
    probe, _ = batch([train[0]], cfg.context, [0])
    model.eval()
    with torch.inference_mode():
        before = model(probe).clone()
    training_rng = torch.get_rng_state()
    torch.save(model.state_dict(), args.output / "model.pt")
    restored = PicoLLM(cfg).eval()
    restored.load_state_dict(torch.load(args.output / "model.pt", weights_only=True,
                                        map_location="cpu"), strict=True)
    with torch.inference_mode():
        reload_error = (before - restored(probe)).abs().max().item()
    assert reload_error == 0.0
    torch.save({"settings": settings, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "completed_steps": stop,
                "torch_rng": training_rng, "sampler_rng": rng.get_state(),
                "history": history}, args.output / "resume.pt")
    write_json(args.output / "config.json", asdict(cfg))
    write_json(args.output / "tokenizer.json", TOKENIZER)
    write_json(args.output / "corpus.json", rows)
    write_json(args.output / "history.json", history)
    report = {"purpose": "user-supplied document training; quality requires evaluation" if args.train_file else "synthetic mechanics test; not general language evaluation",
              "model_name": "PicoLLM", "training_windows": len(windows),
              "window_policy": "nonoverlapping targets; one-token input boundary; reset RoPE; no cross-document context",
              "settings": settings, "start_step": start, "completed_steps": stop,
              "train_documents": len(train), "validation_documents": len(valid),
              "test_documents": sum(r["split"] == "test" for r in rows),
              "unique_parameters": sum(p.numel() for p in model.parameters()),
              "initial": initial, "final": final, "trained_targets": trained_targets,
              "segment_seconds": elapsed, "reload_max_abs_logit_error": reload_error,
              "environment": {"python": platform.python_version(),
                              "torch": str(torch.__version__), "device": "cpu",
                              "dtype": "float32", "threads": 1},
              "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ("course_model.py", "tokenizer.py", "train.py", "evaluate.py")}}
    write_json(args.output / "run-report.json", report)
    files = {name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
             for name in ("config.json", "tokenizer.json", "model.pt")}
    write_json(args.output / "manifest.json", {"format_version": 1, "files": files})
    print(json.dumps({"completed_steps": stop, "train_nll": final["train"]["nll"],
                      "validation_nll": final["validation"]["nll"],
                      "reload_max_abs_logit_error": reload_error,
                      "segment_seconds": elapsed}, indent=2))


if __name__ == "__main__":
    main()
