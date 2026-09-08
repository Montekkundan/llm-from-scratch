"""Verify an original lab artifact, reload on CPU, and rescore its holdout."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from course_model import PicoLLM, ModelConfig
from chat import CHAT_TEMPLATE
from train import TOKENIZER, measure


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
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = load_artifact(args.artifact)
    rows = json.loads((args.artifact / "corpus.json").read_text())
    for row in rows:
        if hashlib.sha256(row["text"].encode()).hexdigest() != row["sha256"]:
            raise ValueError("Corpus record checksum mismatch")
    texts = [r["text"] for r in rows if r["split"] == "validation"]
    report = json.loads((args.artifact / "run-report.json").read_text())
    signature = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    if signature != report["settings"]["corpus_sha256"]:
        raise ValueError("Corpus manifest differs from run report")
    measured = measure(model, texts)
    difference = abs(measured["nll"] - report["final"]["validation"]["nll"])
    if difference > 1e-7:
        raise ValueError("Reloaded validation loss differs from recorded evaluation")
    print(json.dumps({"validation_nll": measured["nll"],
                      "bits_per_byte": measured["bits_per_byte"],
                      "targets": measured["targets"],
                      "recorded_nll_difference": difference}, indent=2))


if __name__ == "__main__":
    main()
