"""Bounded experiments on the same PicoLLM artifact; no capability benchmarks."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from course_model import PicoLLM, ModelConfig
from evaluate import load_artifact
from generate import generate_ids
from tokenizer import BOS, IGNORE, encode
from train import batch, measure, require_new_path


def cache_check(model):
    torch.manual_seed(19)
    ids = torch.randint(0, 256, (1, model.config.context))
    ids[0, 0] = BOS
    errors = []
    for cut in sorted({1, min(7, model.config.context - 1), model.config.context - 1}):
        if cut < 1:
            continue
        _, cache = model.forward_cached(ids[:, :cut])
        cache.assert_prefix(ids[:, :cut])
        logits, complete = model.forward_cached(ids[:, cut:], cache)
        error = (logits - model(ids)[:, cut:]).abs().max().item()
        assert torch.allclose(logits, model(ids)[:, cut:], atol=2e-5, rtol=2e-5)
        errors.append({'prefix': cut, 'chunk': ids.shape[1] - cut, 'max_abs_logit_error': error})
        assert complete.tokens.shape == ids.shape
    _, cache = model.forward_cached(ids[:, :1])
    try:
        cache.assert_prefix(ids[:, :1] + 1)
    except ValueError:
        pass
    else:
        raise AssertionError('edited prefix was accepted')
    altered = copy.deepcopy(model)
    try:
        altered.forward_cached(ids[:, 1:2], cache)
    except ValueError:
        pass
    else:
        raise AssertionError('cache from another model was accepted')
    before = model.token_embedding.weight.clone()
    with torch.no_grad():
        model.token_embedding.weight.add_(0.01)
    try:
        model.forward_cached(ids[:, 1:2], cache)
    except ValueError:
        pass
    else:
        raise AssertionError('stale parameter cache was accepted')
    with torch.no_grad():
        model.token_embedding.weight.copy_(before)
    return {'prefix_checks': errors, 'edited_prefix_rejected': True,
            'other_model_rejected': True, 'changed_weights_rejected': True}


def profile(model):
    prefix = encode('red fox ', eos=False)
    budget = min(12, model.config.context - len(prefix))
    if budget < 1:
        raise ValueError('profile prompt does not fit this model context')
    result = {}
    for cached in (False, True):
        generate_ids(model, prefix, budget, cached=cached)
        times = []
        for _ in range(5):
            start = time.perf_counter()
            output = generate_ids(model, prefix, budget, cached=cached)
            times.append(time.perf_counter() - start)
        result['cached' if cached else 'full_prefix'] = {
            'median_seconds': statistics.median(times), 'samples_seconds': times,
            'token_ids': output['token_ids'], 'completion_tokens': output['usage']['completion_tokens']}
    assert result['cached']['token_ids'] == result['full_prefix']['token_ids']
    return {**result, 'scope': 'five CPU wall-clock samples on one prompt; no GPU throughput claim'}


def quantize_probe(model, texts):
    reference = measure(model, texts)
    changed = copy.deepcopy(model)
    maximum, matrices = 0.0, 0
    with torch.no_grad():
        for parameter in changed.parameters():
            if parameter.ndim < 2:
                continue
            scale = parameter.abs().max().clamp_min(1e-12) / 127
            quantized = (parameter / scale).round().clamp(-127, 127).to(torch.int8)
            reconstructed = quantized.float() * scale
            maximum = max(maximum, (parameter - reconstructed).abs().max().item())
            parameter.copy_(reconstructed)
            matrices += 1
    assert changed.lm_head.weight is changed.token_embedding.weight
    after = measure(changed, texts)
    return {'matrix_tensors': matrices, 'max_abs_weight_error': maximum,
            'reference_nll': reference['nll'], 'reconstructed_nll': after['nll'],
            'nll_delta': after['nll'] - reference['nll'],
            'scope': 'symmetric per-tensor int8 round/dequantize probe; storage and kernels remain FP32'}


def rank_gradient_check(model, texts):
    model = model.double().train()
    x, y = batch(texts[:2], model.config.context, [0, 0])
    parameters = list(model.parameters())
    global_count = int((y != IGNORE).sum())
    full = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), ignore_index=IGNORE, reduction='sum') / global_count
    expected = torch.autograd.grad(full, parameters)
    replica_gradients = []
    world_size = 2
    for rank in range(world_size):
        local = F.cross_entropy(model(x[rank:rank+1]).flatten(0, 1), y[rank:rank+1].flatten(), ignore_index=IGNORE, reduction='sum')
        replica_gradients.append(torch.autograd.grad(world_size * local / global_count, parameters))
    errors = [((a + b) / world_size - ref).abs().max().item()
              for a, b, ref in zip(*replica_gradients, expected)]
    assert max(errors) < 1e-10
    return {'simulated_ranks': 2, 'global_targets': global_count,
            'max_abs_gradient_error': max(errors), 'scope': 'DDP mean-gradient algebra on CPU; no process group launched'}


def recomputation_check(model, texts):
    direct = model.double().train()
    recompute = copy.deepcopy(direct)
    x, y = batch(texts[:2], model.config.context, [0, 0])
    logits = direct(x)
    hidden = recompute.token_embedding(x)
    for block in recompute.blocks:
        hidden = checkpoint(block, hidden, use_reentrant=False)
    checkpoint_logits = recompute.lm_head(recompute.final_norm(hidden))
    for owner, values in ((direct, logits), (recompute, checkpoint_logits)):
        F.cross_entropy(values.flatten(0, 1), y.flatten(), ignore_index=IGNORE).backward()
    error = max((a.grad - b.grad).abs().max().item() for a, b in zip(direct.parameters(), recompute.parameters()))
    assert error < 1e-10
    return {'max_abs_gradient_error': error, 'scope': 'actual block recomputation gradients; no GPU memory or FSDP claim'}


def scaling_budget(model):
    cfg = model.config
    count = sum(p.numel() for p in model.parameters())
    variants = []
    for layers in (cfg.layers, cfg.layers * 2, cfg.layers * 4):
        variant = ModelConfig(**{**asdict(cfg), 'layers': layers})
        n = sum(p.numel() for p in PicoLLM(variant).parameters())
        variants.append({'layers': layers, 'parameters': n, 'approx_flops_per_1m_tokens': 6 * n * 1_000_000})
    return {'base_parameters': count, 'candidate_budgets': variants,
            'scope': '6ND planning approximation only; no fitted scaling law or trained larger models'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=['cache', 'profile', 'quantize', 'scaling', 'distributed', 'recompute', 'ablation-plan'])
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require_new_path(args.output)
    torch.set_num_threads(1)
    model = load_artifact(args.artifact)
    rows = json.loads((args.artifact / 'corpus.json').read_text())
    texts = [row['text'] for row in rows if row['split'] == 'validation']
    if args.experiment == 'cache': result = cache_check(model)
    elif args.experiment == 'profile': result = profile(model)
    elif args.experiment == 'quantize': result = quantize_probe(model, texts)
    elif args.experiment == 'scaling': result = scaling_budget(model)
    elif args.experiment == 'distributed': result = rank_gradient_check(model, texts)
    elif args.experiment == 'recompute': result = recomputation_check(model, texts)
    else:
        result = {'claim': 'Parameter-matched GELU versus baseline SwiGLU changes held-out NLL at a fixed token budget',
                  'baseline_config': asdict(model.config), 'candidate_gelu_width': 3 * model.config.ff_width // 2,
                  'primary_metric': 'paired held-out per-target NLL difference', 'seeds': [7, 11, 19],
                  'keep_fixed': ['corpus hashes and split', 'tokenizer', 'token budget', 'evaluation denominator'],
                  'status': 'preregistered plan only; candidate architecture and six training runs are not executed'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'model': 'PicoLLM', 'experiment': args.experiment,
              'model_sha256': hashlib.sha256((args.artifact / 'model.pt').read_bytes()).hexdigest(),
              'config': asdict(model.config), 'torch': str(torch.__version__), 'result': result}
    with args.output.open('x') as handle: json.dump(report, handle, indent=2); handle.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
