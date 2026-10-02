"""Offline, unedited development samples from immutable complete checkpoints."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import time

from chat_data import MODEL_ID, MODEL_REVISION, user_hash
from chat_eval import evaluate_generations
from gpu_checkpoint import atomic_json, fingerprint, sha256
from scratch_chat import generate_ids, evaluate_cases, load_tokenizer, load_weights, resolve_device

FROZEN_SUITE = Path(__file__).resolve().parent / "evaluation" / "chat-evaluation.json"
PRETRAIN_PROMPTS = [
    "A small language model learns patterns in text by",
    "After the teacher opened the notebook, the students",
    "For a repeatable experiment, first record the",
]
CHAT_CASES = [
    {"id": "monitor-greeting", "category": "monitor_greeting",
     "messages": [{"role": "user", "content": "Hello there. What can you help me practice today?"}]},
    {"id": "monitor-explanation", "category": "monitor_course",
     "messages": [{"role": "user", "content": "In two short sentences, connect tokenization with next-token prediction."}]},
    {"id": "monitor-identity", "category": "monitor_identity",
     "messages": [{"role": "user", "content": "What is your course-assistant name, and who is credited for this course version?"}]},
]


def check_prompt_isolation(suite_path):
    suite = json.loads(Path(suite_path).read_text())
    excluded = {user_hash(message["content"]) for case in suite["cases"]
                for message in case["messages"] if message["role"] == "user"}
    prompts = PRETRAIN_PROMPTS + [message["content"] for case in CHAT_CASES
                                 for message in case["messages"] if message["role"] == "user"]
    if any(user_hash(prompt) in excluded for prompt in prompts):
        raise ValueError("Development monitor prompt overlaps the frozen final suite")
    return sha256(suite_path)


def sample_signals(ids, response, reason):
    body = [token for token in ids if token not in (0, 2)]
    longest, run, previous = 0, 0, None
    for token in body:
        run = run + 1 if token == previous else 1
        longest, previous = max(longest, run), token
    grams = Counter(tuple(body[index:index + 4]) for index in range(max(0, len(body) - 3)))
    windows = sum(grams.values())
    repeated = sum(count - 1 for count in grams.values()) / windows if windows else 0.0
    return {"generated_tokens": len(ids), "content_tokens": len(body),
            "unique_token_fraction": len(set(body)) / len(body) if body else None,
            "longest_identical_token_run": longest, "repeated_4gram_fraction": repeated,
            "repetition_flag": longest >= 8 or len(body) >= 16 and repeated >= 0.5,
            "length_limited": reason in ("length", "context"),
            "empty_response": not response.strip()}


def recent_losses(records, checkpoint_step):
    result = {}
    for row in records:
        step = row.get("global_step", row.get("step"))
        if type(step) is not int or step > checkpoint_step:
            continue
        for name in ("loss", "assistant_token_loss", "nll", "eval_loss"):
            value = row.get(name)
            if isinstance(value, (int, float)) and math.isfinite(value):
                if name not in result or step >= result[name]["step"]:
                    result[name] = {"step": step, "value": value}
    return result


def read_loss_log(path, checkpoint_step):
    rows = []
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if not line.endswith("\n"):
                    break
                raise
    return recent_losses(rows, checkpoint_step)


def probe_scratch(checkpoint, tokenizer_directory, kind="pretrain", device="cpu",
                  dtype="float32", max_new_tokens=32, tokenizer=None):
    checkpoint = Path(checkpoint)
    if not (checkpoint / "complete.json").is_file():
        raise ValueError("Pass an immutable completed checkpoint directory, not a latest pointer")
    model, state, directory = load_weights(checkpoint, device, dtype)
    if kind not in ("pretrain", "scratch_sft") or (state.get("stage") == "scratch_sft") != (kind == "scratch_sft"):
        raise ValueError("Probe kind differs from the scratch checkpoint stage")
    current_sources = {name: sha256(Path(__file__).with_name(name))
                       for name in ("gpu_model.py", "course_model.py", "scratch_chat.py")}
    for name in ("gpu_model.py", "course_model.py"):
        if name in state.get("source_sha256", {}) and state["source_sha256"][name] != current_sources[name]:
            raise ValueError("Inference model source differs from the checkpoint: " + name)
    if tokenizer is None:
        tokenizer = load_tokenizer(tokenizer_directory, state["data_identity"]["tokenizer"],
                                   model.config.vocab_size, state.get("chat_template_sha256"))
    results = []
    if kind == "pretrain":
        for index, prompt in enumerate(PRETRAIN_PROMPTS):
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            ids, reason = generate_ids(model, list(prompt_ids), max_new_tokens, dtype)
            text = tokenizer.decode(ids, skip_special_tokens=True)
            results.append({"id": f"monitor-continuation-{index + 1}", "prompt": prompt,
                            "prompt_token_ids": list(prompt_ids), "output_token_ids": ids,
                            "response": text, "finish_reason": reason})
    else:
        results = evaluate_cases(model, tokenizer, CHAT_CASES, max_new_tokens, dtype)
    for row in results:
        row["signals"] = sample_signals(row["output_token_ids"], row["response"], row["finish_reason"])
    return {"kind": kind, "checkpoint": str(directory), "checkpoint_sha256": sha256(directory / "checkpoint.pt"),
            "checkpoint_hash_scope": "checkpoint.pt", "global_step": state["global_step"],
            "consumed_tokens": state["tokens"], "supervised_tokens": state.get("supervised_tokens"),
            "training_status": state["status"], "checkpoint_validation": state.get("validation"),
            "checkpoint_validation_note": "The stored validation may precede this checkpoint. Use logged loss steps when available.",
            "tokenizer": state["data_identity"]["tokenizer"], "model_sources_sha256": current_sources,
            "device": device, "dtype": dtype, "results": results}


def probe_lora(checkpoint, tokenizer_directory, config_path, device="mps", max_new_tokens=32):
    from chat_finetune import validate_checkpoint
    checkpoint = Path(checkpoint)
    marker = validate_checkpoint(checkpoint)
    config = json.loads(Path(config_path).read_text())
    if (config.get("model_id"), config.get("model_revision")) != (MODEL_ID, MODEL_REVISION):
        raise ValueError("LoRA monitor requires the pinned foundation and revision")
    tokenizer_directory = Path(tokenizer_directory)
    tokenizer_files = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                       "added_tokens.json", "chat_template.jinja", "vocab.json", "merges.txt")
    for name in tokenizer_files:
        if name in marker["files"] and sha256(tokenizer_directory / name) != marker["files"][name]:
            raise ValueError("LoRA tokenizer differs from the completed checkpoint: " + name)
    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    dtype = "bfloat16" if str(device).startswith("cuda") else "float32"
    target = resolve_device(device, dtype)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_directory), local_files_only=True,
                                               trust_remote_code=False)
    foundation = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
        local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float32,
        attn_implementation="sdpa").to(target)
    model = PeftModel.from_pretrained(foundation, str(checkpoint), is_trainable=False,
                                     local_files_only=True)
    results = evaluate_generations(model, tokenizer, CHAT_CASES, max_new_tokens)
    for row in results:
        ids = row["output_token_ids"]
        row["finish_reason"] = "eos_chatml" if ids and ids[-1] == tokenizer.eos_token_id else \
                               "length" if len(ids) == max_new_tokens else "other_stop"
        row["signals"] = sample_signals(ids, row["response"], row["finish_reason"])
    trainer_state = json.loads((checkpoint / "trainer_state.json").read_text())
    step = marker["global_step"]
    return {"kind": "lora", "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint / "COMPLETE.json"),
            "checkpoint_hash_scope": "COMPLETE.json containing verified full-state file checksums",
            "global_step": step, "consumed_tokens": marker["input_tokens_seen"],
            "supervised_tokens": marker["supervised_tokens_seen"], "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION, "device": str(target), "dtype": dtype,
            "logged_losses": recent_losses(trainer_state.get("log_history", []), step), "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--kind", choices=("pretrain", "scratch_sft", "lora"), required=True)
    parser.add_argument("--config", help="Pinned foundation config for LoRA")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32",
                        help="Scratch precision; LoRA uses bf16 on CUDA and float32 elsewhere")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--metrics", help="Optional immutable or live JSONL loss log; only steps at/before checkpoint are read")
    parser.add_argument("--frozen-suite", default=str(FROZEN_SUITE))
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.kind == "lora" and not args.config:
        parser.error("Require positive max-new-tokens and --config for LoRA")
    suite_sha = check_prompt_isolation(args.frozen_suite)
    began = time.perf_counter()
    if args.kind == "lora":
        report = probe_lora(args.checkpoint, args.tokenizer, args.config, args.device, args.max_new_tokens)
    else:
        report = probe_scratch(args.checkpoint, args.tokenizer, args.kind, args.device,
                               args.dtype, args.max_new_tokens)
    if args.metrics:
        report["logged_losses"] = read_loss_log(args.metrics, report["global_step"])
    report.update({"monitor_prompt_set_sha256": fingerprint(PRETRAIN_PROMPTS if args.kind == "pretrain" else CHAT_CASES),
                   "final_suite_sha256_for_exclusion_check": suite_sha, "final_suite_used_for_generation": False,
                   "probe_source_sha256": sha256(__file__), "run_seconds": time.perf_counter() - began,
                   "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens,
                                  "stop_token_ids": [2] if args.kind == "lora" else [2, 0]},
                   "scope": "Development monitoring, not final-suite acceptance. Samples are unedited; repetition and length signals are diagnostic heuristics."})
    atomic_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
