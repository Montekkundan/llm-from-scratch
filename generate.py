"""Continue text or a role-lines-v1 message with the exported PicoLLM weights."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from tokenizer import BOS, EOS, PAD, encode
from chat import CHAT_TEMPLATE, serialize
from evaluate import load_artifact


@torch.inference_mode()
def generate_ids(model, prefix, max_new_tokens=24, temperature=0.0, seed=7, cached=False):
    if temperature < 0 or max_new_tokens < 1 or not prefix:
        raise ValueError("Require a nonempty prefix, positive generation length and nonnegative temperature")
    if len(prefix) + max_new_tokens > model.config.context:
        raise ValueError("Prompt plus requested generation exceeds PicoLLM context")
    model.eval()
    ids = torch.tensor([prefix], dtype=torch.long)
    rng = torch.Generator().manual_seed(seed)
    generated, finish, cache = [], "length", None
    if cached:
        logits, cache = model.forward_cached(ids)
    for _ in range(max_new_tokens):
        if not cached:
            logits = model(ids)
        scores = logits[0, -1].clone()
        scores[BOS] = scores[PAD] = -torch.inf
        token = int(scores.argmax()) if temperature == 0 else int(
            torch.multinomial((scores / temperature).softmax(-1), 1, generator=rng))
        generated.append(token)
        if token == EOS:
            finish = "stop"
            break
        ids = torch.cat((ids, torch.tensor([[token]], dtype=torch.long)), 1)
        if cached and len(generated) < max_new_tokens:
            logits, cache = model.forward_cached(ids[:, -1:], cache)
    text = bytes(token for token in generated if 0 <= token < 256).decode("utf-8", errors="replace")
    return {"text": text, "token_ids": generated, "finish_reason": finish,
            "usage": {"prompt_tokens": len(prefix), "completion_tokens": len(generated)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt")
    group.add_argument("--message")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cached", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = load_artifact(args.artifact)
    if args.message is not None:
        if json.loads((args.artifact / "chat_template.json").read_text()) != CHAT_TEMPLATE:
            raise ValueError("Message mode requires the exported role-lines-v1 template")
        prefix, _ = serialize([{"role": "user", "content": args.message}], generation=True)
    else:
        prefix = encode(args.prompt, eos=False)
    result = generate_ids(model, prefix, args.max_new_tokens, args.temperature, args.seed, args.cached)
    result["model_sha256"] = hashlib.sha256((args.artifact / "model.pt").read_bytes()).hexdigest()
    result["manifest_sha256"] = hashlib.sha256((args.artifact / "manifest.json").read_bytes()).hexdigest()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as file:
            json.dump(result, file, indent=2)
            file.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
