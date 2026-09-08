"""Continue the same PicoLLM base weights on a tiny, declared echo task.

This checks assistant-only training and artifact lineage. It is not a broad
instruction-following or reasoning benchmark. All data are generated here.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from tokenizer import TOKENIZER, PAD, IGNORE
from chat import CHAT_TEMPLATE, serialize
from train import optimizer_for, measure, write_json
from evaluate import load_artifact
from generate import generate_ids


def task_data():
    groups = {"train": "red blue green fox owl cat rests runs".split(),
              "validation": ["gold", "elk"], "test": ["waits", "jumps"]}
    return [{"id": f"echo-{word}", "split": split, "word": word,
             "messages": [{"role": "user", "content": "echo: " + word},
                          {"role": "assistant", "content": word}]}
            for split, words in groups.items() for word in words]


def chat_batch(rows, context):
    serialized = [serialize(row["messages"]) for row in rows]
    width = max(len(labels) for _, labels in serialized)
    if width > context:
        raise ValueError("Conversation exceeds configured context; truncation is not implicit")
    x = torch.full((len(rows), width), PAD, dtype=torch.long)
    y = torch.full_like(x, IGNORE)
    for index, (ids, labels) in enumerate(serialized):
        x[index, :len(labels)] = torch.tensor(ids[:-1])
        y[index, :len(labels)] = torch.tensor(labels)
    if not (y != IGNORE).any():
        raise ValueError("No assistant targets")
    return x, y


@torch.inference_mode()
def score_task(model, rows):
    model.eval()
    x, y = chat_batch(rows, model.config.context)
    count = int((y != IGNORE).sum())
    nll = F.cross_entropy(model(x).reshape(-1, model.config.vocab_size),
                          y.reshape(-1), ignore_index=IGNORE, reduction="sum").item() / count
    outputs = []
    for row in rows:
        prefix, _ = serialize(row["messages"][:1], generation=True)
        generated = generate_ids(model, prefix, max_new_tokens=12)
        complete = generated["finish_reason"] == "stop" and generated["text"].endswith("\n")
        answer = generated["text"][:-1] if complete else None
        outputs.append({"id": row["id"], "expected": row["word"], "generated": generated,
                        "canonical_answer": answer, "correct": complete and answer == row["word"]})
    return {"assistant_nll": nll, "supervised_targets": count,
            "correct": sum(item["correct"] for item in outputs), "total": len(outputs), "outputs": outputs}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--evaluate", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.evaluate:
        model = load_artifact(args.evaluate)
        rows = json.loads((args.evaluate / "sft-data.json").read_text())
        report = json.loads((args.evaluate / "sft-report.json").read_text())
        if sha256(args.evaluate / "sft-data.json") != report["data_sha256"]:
            raise ValueError("SFT dataset changed")
        test = score_task(model, [row for row in rows if row["split"] == "test"])
        recorded = report["final"]["test"]
        if abs(test["assistant_nll"] - recorded["assistant_nll"]) > 1e-7:
            raise ValueError("Reloaded SFT score differs")
        assert [row["generated"]["token_ids"] for row in test["outputs"]] == [
            row["generated"]["token_ids"] for row in recorded["outputs"]]
        print(json.dumps({"test": test, "lineage": report["lineage"], "reload": "passed"}, indent=2))
        return
    if args.base is None or args.output is None or args.steps < 1:
        parser.error("training requires --base, --output and positive --steps")
    model = load_artifact(args.base)
    if (args.base / "chat_template.json").exists():
        raise ValueError("This SFT experiment starts from the base continuation artifact")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    rows = task_data()
    splits = {name: [r for r in rows if r["split"] == name] for name in ("train", "validation", "test")}
    initial = {name: score_task(model, data) for name, data in splits.items()}
    optimizer = optimizer_for(model, 0.002)
    x, y = chat_batch(splits["train"], model.config.context)
    count = int((y != IGNORE).sum())
    history = []
    for step in range(args.steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x).reshape(-1, model.config.vocab_size), y.reshape(-1),
                               ignore_index=IGNORE, reduction="sum") / count
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite SFT loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        history.append({"step": step + 1, "assistant_nll": loss.item(), "targets": count,
                        "gradient_norm": norm.item()})
    final = {name: score_task(model, data) for name, data in splits.items()}
    base_rows = json.loads((args.base / "corpus.json").read_text())
    base_language = measure(model, [row["text"] for row in base_rows if row["split"] == "validation"])
    torch.save(model.state_dict(), args.output / "model.pt")
    write_json(args.output / "config.json", asdict(model.config))
    write_json(args.output / "tokenizer.json", TOKENIZER)
    write_json(args.output / "chat_template.json", CHAT_TEMPLATE)
    write_json(args.output / "sft-data.json", rows)
    write_json(args.output / "sft-history.json", history)
    report = {"model_name": "PicoLLM", "purpose": "tiny echo-task adaptation; not general instruction evaluation",
              "lineage": {"base_model_sha256": sha256(args.base / "model.pt"),
                          "base_manifest_sha256": sha256(args.base / "manifest.json"),
                          "sft_model_sha256": sha256(args.output / "model.pt")},
              "settings": {"steps": args.steps, "seed": args.seed, "lr": 0.002,
                           "selection": "fixed final step; test results never select a checkpoint"},
              "data_sha256": sha256(args.output / "sft-data.json"),
              "initial": initial, "final": final, "base_language_after_sft": base_language,
              "environment": {"torch": str(torch.__version__), "device": "cpu", "dtype": "float32"}}
    write_json(args.output / "sft-report.json", report)
    names = ("config.json", "tokenizer.json", "model.pt", "chat_template.json")
    write_json(args.output / "manifest.json", {"format_version": 1,
               "files": {name: sha256(args.output / name) for name in names}})
    restored = load_artifact(args.output)
    with torch.inference_mode():
        torch.testing.assert_close(model.eval()(x), restored(x), rtol=0, atol=0)
    print(json.dumps({"output": str(args.output), "lineage": report["lineage"],
                      "train_exact_match": [final["train"]["correct"], final["train"]["total"]],
                      "test_exact_match": [final["test"]["correct"], final["test"]["total"]]}, indent=2))


if __name__ == "__main__":
    main()
