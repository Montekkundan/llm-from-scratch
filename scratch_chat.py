"""Greedy native ChatML inference for complete scratch PicoLLM checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from chat_data import NEUTRAL_SYSTEM, SYSTEM, conversation_hash
from chat_eval import effective_messages, generation_cases
from course_model import ModelConfig
from gpu_checkpoint import atomic_json, load_checkpoint, resolve_checkpoint, sha256
from gpu_model import GPUPicoLLM
from gpu_train import autocast


def checkpoint_directory(path):
    path = Path(path)
    if path.is_file() and path.name == "checkpoint.pt":
        return path.parent
    if (path / "latest.json").exists() and not (path / "complete.json").exists():
        return resolve_checkpoint(path, "latest")
    return path


def resolve_device(name, dtype="float32"):
    device = torch.device(name)
    if device.type not in ("cpu", "cuda", "mps"):
        raise ValueError("Device must be cpu, cuda or mps")
    if dtype not in ("float32", "bfloat16"):
        raise ValueError("dtype must be float32 or bfloat16")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device unavailable")
        device = torch.device("cuda", 0 if device.index is None else device.index)
        torch.cuda.set_device(device)
        if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA bfloat16 unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS device unavailable")
    if device.type == "cpu":
        torch.set_num_threads(1)
    return device


def load_weights(path, device="cpu", dtype="float32"):
    directory = checkpoint_directory(path)
    state = load_checkpoint(directory)
    config = ModelConfig(**state["resolved_config"]["model"])
    model = GPUPicoLLM(config).to(resolve_device(device, dtype))
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model, state, directory


def load_tokenizer(directory, metadata, vocab_size, template_sha=None):
    from transformers import AutoTokenizer
    directory = Path(directory)
    files = sorted(path for path in directory.iterdir()
                   if path.is_file() and not path.name.startswith("._"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode() + b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    if not files or metadata.get("config_sha256") != digest.hexdigest():
        raise ValueError("Saved tokenizer files differ from the pretraining tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True,
                                               trust_remote_code=False)
    if len(tokenizer) != vocab_size or tokenizer.eos_token_id != 2:
        raise ValueError("Require the matching ChatML tokenizer with EOS ID 2")
    if tokenizer.convert_tokens_to_ids("<|endoftext|>") != 0:
        raise ValueError("Require document separator ID 0")
    if template_sha and hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != template_sha:
        raise ValueError("Chat template differs from the SFT preparation")
    return tokenizer


@torch.inference_mode()
def generate_ids(model, prompt_ids, max_new_tokens=128, dtype="float32"):
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max-new-tokens must be a positive integer")
    if not isinstance(prompt_ids, list) or not prompt_ids:
        raise ValueError("Require a nonempty list of prompt token IDs")
    if any(type(value) is not int or not 0 <= value < model.config.vocab_size for value in prompt_ids):
        raise ValueError("Prompt IDs must fit the model vocabulary")
    if len(prompt_ids) >= model.config.context:
        raise ValueError("Prompt leaves no room in the model context")
    device = next(model.parameters()).device
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output = []
    model.eval()
    budget = min(max_new_tokens, model.config.context - len(prompt_ids))
    for _ in range(budget):
        with autocast(device, dtype):
            logits = model.lm_head(model.hidden_states(ids)[:, -1]).float()
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("Nonfinite generation logits")
        token = int(logits.argmax(-1).item())
        output.append(token)
        if token in (0, 2):
            return output, "eos_document" if token == 0 else "eos_chatml"
        ids = torch.cat((ids, ids.new_tensor([[token]])), dim=1)
    return output, "context" if budget < max_new_tokens else "length"


def generate_response(model, tokenizer, messages, max_new_tokens=128, dtype="float32"):
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    ids, reason = generate_ids(model, list(prompt), max_new_tokens, dtype)
    return {"output_token_ids": ids, "response": tokenizer.decode(ids, skip_special_tokens=True),
            "finish_reason": reason, "prompt_tokens": len(prompt)}


def evaluate_cases(model, tokenizer, cases, max_new_tokens=128, dtype="float32"):
    results = []
    for case in cases:
        systems = [("course_identity", SYSTEM)]
        if "identity" in case["category"].lower():
            systems.append(("neutral_identity", NEUTRAL_SYSTEM))
        for condition, system in systems:
            messages = effective_messages(case["messages"], system)
            result = {"id": case["id"], "category": case["category"], "condition": condition,
                      "messages": messages, "conversation_sha256": conversation_hash(messages),
                      "effective_prompt_sha256": hashlib.sha256(json.dumps(messages, ensure_ascii=False,
                          sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                      "expectations": case.get("expectations"), "human_review": None}
            try:
                result.update(generate_response(model, tokenizer, messages, max_new_tokens, dtype))
            except (ValueError, RuntimeError, FloatingPointError) as error:
                result.update({"runtime_error": str(error), "response": "", "output_token_ids": [],
                               "finish_reason": "error"})
            result["identity_name_mentioned"] = "picollm" in result["response"].lower()
            result["identity_creator_mentioned"] = "montek singh kundan" in result["response"].lower()
            results.append(result)
    return results


def interactive(model, tokenizer, max_new_tokens, dtype, system):
    messages = [{"role": "system", "content": system}]
    print("Scratch PicoLLM checkpoint. Type /quit to leave or /clear to reset.")
    while True:
        try:
            text = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if text.strip() == "/quit":
            return
        if text.strip() == "/clear":
            messages = [{"role": "system", "content": system}]
            continue
        if not text.strip():
            continue
        messages.append({"role": "user", "content": text})
        try:
            result = generate_response(model, tokenizer, messages, max_new_tokens, dtype)
        except ValueError as error:
            messages.pop()
            print(str(error) + ". Use /clear to reset.")
            continue
        print("PicoLLM:", result["response"])
        messages.append({"role": "assistant", "content": result["response"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True, help="Saved pretraining tokenizer directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--suite")
    parser.add_argument("--prompt")
    parser.add_argument("--output")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--system", choices=("course", "neutral"), default="course")
    args = parser.parse_args()
    if args.interactive and (args.suite or args.prompt):
        parser.error("--interactive cannot be combined with --suite or --prompt")
    if not args.interactive and not args.prompt and not args.output:
        parser.error("--output is required for suite evaluation")
    model, state, directory = load_weights(args.checkpoint, args.device, args.dtype)
    tokenizer = load_tokenizer(args.tokenizer, state["data_identity"]["tokenizer"],
                               model.config.vocab_size, state.get("chat_template_sha256"))
    system = SYSTEM if args.system == "course" else NEUTRAL_SYSTEM
    if args.interactive:
        interactive(model, tokenizer, args.max_new_tokens, args.dtype, system)
        return
    if args.prompt:
        cases = [{"id": "terminal-prompt", "category": "conversation",
                  "messages": [{"role": "user", "content": args.prompt}]}]
        messages = effective_messages(cases[0]["messages"], system)
        results = [{"id": "terminal-prompt", "condition": args.system,
                    "messages": messages, **generate_response(model, tokenizer, messages,
                                                               args.max_new_tokens, args.dtype)}]
    else:
        cases = json.loads(Path(args.suite).read_text())["cases"] if args.suite else generation_cases()
        results = evaluate_cases(model, tokenizer, cases, args.max_new_tokens, args.dtype)
    report = {"checkpoint_sha256": sha256(directory / "checkpoint.pt"),
              "base_checkpoint_sha256": state.get("base_checkpoint_sha256"),
              "stage": state.get("stage", "scratch_pretraining"), "training_status": state["status"],
              "tokenizer": state["data_identity"]["tokenizer"], "suite_sha256": sha256(args.suite) if args.suite else None,
              "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens,
                             "stop_token_ids": [2, 0], "dtype": args.dtype},
              "limitations": "Unedited model generations. Identity substring flags are weak proxies; human review must use this scratch checkpoint's actual provenance.",
              "results": results}
    if args.output:
        atomic_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
