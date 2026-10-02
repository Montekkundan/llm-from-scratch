"""Continue text or a role-lines-v1 message with the exported PicoLLM weights."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from tokenizer import BOS, EOS, PAD, encode
from chat import CHAT_TEMPLATE, serialize
from evaluate import load_artifact
from train import require_new_path


def check_filter(top_k, top_p):
    if type(top_k) is not int or top_k < 0 or not 0 < top_p <= 1:
        raise ValueError("Require integer top_k >= 0 (0 is off) and 0 < top_p <= 1 (1 is off)")


def filter_scores(scores, top_k=0, top_p=1.0):
    """Mask logits outside the sampling support; the defaults leave them untouched.

    top_k keeps the k highest logits (0 keeps all). top_p keeps the smallest set of
    highest-probability tokens whose softmax mass is at least p: a token survives
    when the mass strictly before it in descending order is below p, so the best
    token always survives. Top-k is applied first and top-p renormalises within it.
    """
    check_filter(top_k, top_p)
    if top_k and top_k < scores.shape[-1]:
        keep = torch.zeros_like(scores, dtype=torch.bool).scatter(-1, scores.topk(top_k).indices, True)
        scores = scores.masked_fill(~keep, -torch.inf)
    if top_p < 1:
        probabilities, order = scores.softmax(-1).sort(descending=True)
        total = probabilities.cumsum(-1)
        before = torch.cat((torch.zeros_like(total[..., :1]), total[..., :-1]), -1)
        beyond = before >= top_p
        scores = scores.masked_fill(torch.zeros_like(beyond).scatter(-1, order, beyond), -torch.inf)
    return scores


@torch.inference_mode()
def generate_ids(model, prefix, max_new_tokens=24, temperature=0.0, seed=7, cached=False,
                 top_k=0, top_p=1.0):
    if temperature < 0 or max_new_tokens < 1 or not prefix:
        raise ValueError("Require a nonempty prefix, positive generation length and nonnegative temperature")
    check_filter(top_k, top_p)  # greedy decoding (temperature 0) ignores both
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
            torch.multinomial(filter_scores(scores / temperature, top_k, top_p).softmax(-1), 1, generator=rng))
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
    parser.add_argument("--top-k", type=int, default=0, help="sample only among the k highest logits; 0 is off")
    parser.add_argument("--top-p", type=float, default=1.0, help="nucleus mass in (0, 1]; 1 is off")
    parser.add_argument("--cached", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output:
        require_new_path(args.output)
    torch.set_num_threads(1)
    model = load_artifact(args.artifact)
    if args.message is not None:
        if json.loads((args.artifact / "chat_template.json").read_text()) != CHAT_TEMPLATE:
            raise ValueError("Message mode requires the exported role-lines-v1 template")
        prefix, _ = serialize([{"role": "user", "content": args.message}], generation=True)
    else:
        prefix = encode(args.prompt, eos=False)
    result = generate_ids(model, prefix, args.max_new_tokens, args.temperature, args.seed, args.cached,
                          args.top_k, args.top_p)
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
