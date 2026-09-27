"""Verify an original lab artifact, reload on CPU, and rescore its holdout."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from course_model import PicoLLM, ModelConfig
from chat import CHAT_TEMPLATE
from train import TOKENIZER, measure
from tokenizer import EOS, encode


def unigram_baseline(train_texts, heldout_texts):
    """Add-one byte/EOS model fitted only on training documents."""
    counts = {token: 1 for token in (*range(256), EOS)}
    for text in train_texts:
        for token in encode(text)[1:]:
            counts[token] += 1
    denominator = sum(counts.values())
    nll_sum = byte_nll_sum = 0.0
    targets = byte_count = 0
    for text in heldout_texts:
        for token in encode(text)[1:]:
            loss = -math.log(counts[token] / denominator)
            nll_sum += loss
            targets += 1
            if token != EOS:
                byte_nll_sum += loss
                byte_count += 1
    if not targets or not byte_count:
        raise ValueError("Baseline requires nonempty held-out text")
    return {"nll": nll_sum / targets, "targets": targets,
            "bits_per_byte": byte_nll_sum / (byte_count * math.log(2)), "bytes": byte_count}


def load_artifact(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    expected = {"config.json", "tokenizer.json", "model.pt"}
    if manifest.get("format_version") != 1 or not expected <= set(manifest.get("files", {})) or set(manifest.get("files", {})) - expected - {"chat_template.json"}:
        raise ValueError("Unsupported artifact manifest")
    for name in sorted(manifest["files"]):
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != manifest["files"][name]:
            raise ValueError(f"Checksum mismatch: {name}")
    if json.loads((directory / "tokenizer.json").read_text()) != TOKENIZER:
        raise ValueError("This evaluator requires the baseline byte tokenizer")
    if "chat_template.json" in manifest["files"]:
        if json.loads((directory / "chat_template.json").read_text()) != CHAT_TEMPLATE:
            raise ValueError("Unsupported chat template")
    config = ModelConfig(**json.loads((directory / "config.json").read_text()))
    if config.vocab_size != TOKENIZER["vocab_size"]:
        raise ValueError("Model and tokenizer vocabulary sizes differ")
    model = PicoLLM(config).float().cpu()
    model.load_state_dict(torch.load(directory / "model.pt", weights_only=True,
                                     map_location="cpu"), strict=True)
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = load_artifact(args.artifact)
    rows = json.loads((args.artifact / "corpus.json").read_text())
    for row in rows:
        if hashlib.sha256(row["text"].encode()).hexdigest() != row["sha256"]:
            raise ValueError("Corpus record checksum mismatch")
    validation = [r["text"] for r in rows if r["split"] == "validation"]
    report = json.loads((args.artifact / "run-report.json").read_text())
    signature = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    if signature != report["settings"]["corpus_sha256"]:
        raise ValueError("Corpus manifest differs from run report")
    validation_score = measure(model, validation)
    difference = abs(validation_score["nll"] - report["final"]["validation"]["nll"])
    if difference > 1e-7:
        raise ValueError("Reloaded validation loss differs from recorded evaluation")
    texts = [r["text"] for r in rows if r["split"] == args.split]
    if not texts:
        raise ValueError(f"Artifact has no {args.split} documents")
    measured = validation_score if args.split == "validation" else measure(model, texts)
    baseline = unigram_baseline([r["text"] for r in rows if r["split"] == "train"], texts)
    print(json.dumps({"split": args.split, "validation_nll": validation_score["nll"],
                      "nll": measured["nll"], "bits_per_byte": measured["bits_per_byte"],
                      "targets": measured["targets"], "unigram_baseline": baseline,
                      "recorded_nll_difference": difference}, indent=2))


if __name__ == "__main__":
    main()
