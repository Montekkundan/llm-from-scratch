"""Executable milestones against PicoLLM's actual source, not a second toy model."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from course_model import PicoLLM, ModelConfig, apply_rope, RMSNorm
from tokenizer import encode, decode, train_bpe, encode_bpe, decode_bpe, BOS, EOS, PAD, IGNORE
from train import corpus, batch, measure, optimizer_for, learning_rate
from chat import serialize


def run(name):
    torch.manual_seed(7)
    torch.set_num_threads(1)
    model = PicoLLM().eval()
    ids = torch.tensor([[BOS, 104, 105, EOS]], dtype=torch.long)
    if name == "objective":
        logits = model(ids)[:, :-1]
        labels = ids[:, 1:]
        nll = -logits.log_softmax(-1).gather(-1, labels[..., None]).mean()
        torch.testing.assert_close(nll, F.cross_entropy(logits.reshape(-1, 259), labels.reshape(-1)))
        return {"prediction_pairs": list(zip(ids[0, :-1].tolist(), labels[0].tolist())), "initial_nll": nll.item()}
    if name == "tokenizer":
        texts = ["", "café", "e\u0301", "🙂", "你好", "line\n"]
        assert all(decode(encode(s)) == s for s in texts)
        assert encode("café") == [256, 99, 97, 102, 195, 169, 257]
        return {"roundtrips": len(texts), "vocab_size": 259, "cafe_ids": encode("café")}
    if name == "bpe":
        merges = train_bpe(["abab", "abab"], 2)
        assert merges == [(97, 98, 259), (259, 259, 260)]
        assert encode_bpe("abab", merges) == [260]
        assert decode_bpe(encode_bpe("café abab", merges), merges) == "café abab"
        return {"format": "teaching-bpe-v1", "merges": merges, "vocab_size": 261,
                "deployment_tokenizer_changed": False}
    if name == "embedding":
        e = model.token_embedding
        selected = e(torch.tensor([2, 0, 2]))
        torch.testing.assert_close(selected, F.one_hot(torch.tensor([2, 0, 2]), 259).float() @ e.weight)
        selected.sum().backward()
        assert e.weight.grad[2].eq(2).all() and e.weight.grad[0].eq(1).all()
        assert model.lm_head.weight is e.weight
        return {"embedding_parameters": e.weight.numel(), "tied": True}
    if name == "rope":
        x = torch.randn(1, 4, 5, 16, dtype=torch.float64)
        y = torch.randn_like(x)
        positions = torch.arange(5)
        rx, ry = apply_rope(x, positions), apply_rope(y, positions)
        torch.testing.assert_close(rx.square().sum(-1), x.square().sum(-1))
        torch.testing.assert_close(rx @ ry.transpose(-2, -1),
                                   apply_rope(x, positions+9) @ apply_rope(y, positions+9).transpose(-2, -1))
        return {"head_width": 16, "pairing": "adjacent", "base": 10000, "common_shift": 9}
    if name in ("attention", "heads"):
        a = model.blocks[0].attention
        x = torch.randn(2, 5, 64, requires_grad=True)
        y = a(x)
        assert y.shape == (2, 5, 64)
        value = a.qkv(x[:, :1]).chunk(3, -1)[2]
        torch.testing.assert_close(y[:, :1], a.output(value))
        y[:, 2].sum().backward()
        assert torch.count_nonzero(x.grad[:, 3:]) == 0
        return {"output_shape": list(y.shape), "heads": a.heads,
                "parameters": sum(p.numel() for p in a.parameters()), "future_input_gradient": 0}
    if name == "swiglu":
        ffn = model.blocks[0].ffn
        x = torch.randn(1, 4, 64)
        expected = ffn.output(F.silu(ffn.gate(x)) * ffn.value(x))
        torch.testing.assert_close(ffn(x), expected)
        changed = x.clone(); changed[:, -1] += 3
        torch.testing.assert_close(ffn(x)[:, :-1], ffn(changed)[:, :-1])
        return {"parameters": sum(p.numel() for p in ffn.parameters()), "hidden_width": 176}
    if name == "norm":
        x = torch.randn(2, 4, 64)
        norm = RMSNorm(64, 1e-5)
        torch.testing.assert_close(norm(x), nn.RMSNorm(64, eps=1e-5)(x))
        block = model.blocks[0]
        with torch.no_grad():
            block.attention.output.weight.zero_(); block.ffn.output.weight.zero_()
        torch.testing.assert_close(block(x), x, rtol=0, atol=0)
        return {"eps": 1e-5, "normalized_axis": "features", "zero_branch_identity": True}
    if name in ("decoder", "debug"):
        logits = model(ids)
        for cut in range(1, ids.shape[1]):
            changed = ids.clone(); changed[:, cut:] = 33
            torch.testing.assert_close(logits[:, :cut], model(changed)[:, :cut], rtol=0, atol=0)
        F.cross_entropy(logits[:, :-1].reshape(-1,259),ids[:,1:].reshape(-1)).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        return {"shape": list(logits.shape), "parameters": sum(p.numel() for p in model.parameters()),
                "prefix_cuts": 3, "all_parameter_gradients_finite": True}
    if name == "corpus":
        rows = corpus();train = {r['sha256'] for r in rows if r['split']=='train'}
        valid = {r['sha256'] for r in rows if r['split']=='validation'}
        assert not train & valid and len(train | valid) == 64
        return {"train_documents": len(train), "validation_documents": len(valid), "rows": rows}
    if name == "batches":
        x,y = batch(["hi", "café"])
        assert int((y != IGNORE).sum()) == 9
        assert y[0,:3].tolist() == [104,105,EOS]
        assert x[0,3:].eq(PAD).all() and y[0,3:].eq(IGNORE).all()
        return {"inputs": x.tolist(), "targets": y.tolist(), "supervised_targets": 9}
    if name == "optimizer":
        opt = optimizer_for(model, 0.003)
        parameters = [p for group in opt.param_groups for p in group['params']]
        assert len(parameters) == len({id(p) for p in parameters})
        before = model.token_embedding.weight.detach().clone()
        F.cross_entropy(model(ids)[:,:-1].reshape(-1,259), ids[:,1:].reshape(-1)).backward();opt.step()
        assert not torch.equal(before, model.token_embedding.weight)
        return {"parameter_groups": len(opt.param_groups), "unique_parameter_objects": len(parameters), "update_changed_weights": True}
    if name == "loop":
        model.double(); other=copy.deepcopy(model)
        x,y=batch(["hi", "café"]); count=(y!=IGNORE).sum()
        F.cross_entropy(model(x).reshape(-1,259),y.reshape(-1),ignore_index=IGNORE,reduction='sum').div(count).backward()
        for i in range(2):
            F.cross_entropy(other(x[i:i+1]).reshape(-1,259),y[i:i+1].reshape(-1),ignore_index=IGNORE,reduction='sum').div(count).backward()
        error=max((p.grad-q.grad).abs().max().item() for p,q in zip(model.parameters(),other.parameters()))
        assert error<1e-10
        return {"microbatch_gradient_max_abs_error": error, "total_targets": int(count)}
    if name == "budget":
        count=sum(p.numel() for p in model.parameters());assert count==117248
        return {"parameters": count, "fp32_weight_bytes": count*4, "dense_cache_bytes_at_128": 2*2*128*64*4,
                "first_lr": learning_rate(0,160), "last_lr": learning_rate(159,160)}
    if name == "evaluation":
        texts=[r['text'] for r in corpus() if r['split']=='validation']
        a,b=measure(model,texts,batch_size=1),measure(model,texts,batch_size=8)
        assert abs(a['nll']-b['nll'])<1e-6
        return {"nll": a['nll'], "targets": a['targets'], "batching_difference": abs(a['nll']-b['nll'])}
    if name == "identity":
        payload=json.dumps({'config':asdict(model.config),'corpus':corpus()},sort_keys=True).encode()
        return {"config_and_corpus_sha256": hashlib.sha256(payload).hexdigest(),
                "source_sha256": {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}}
    if name == "chat":
        messages=[{'role':'user','content':'echo: red'},{'role':'assistant','content':'red'}]
        tokens,labels=serialize(messages);prefix,_=serialize(messages[:1],generation=True)
        assert tokens[:len(prefix)]==prefix
        assert [n for n in labels if n!=IGNORE]==[*b'red\n',EOS]
        return {"training_ids": tokens, "generation_prefix_ids": prefix, "shifted_labels": labels,
                "supervised_targets": sum(n!=IGNORE for n in labels)}
    raise ValueError('Unknown checkpoint: '+name)


NAMES = ['objective','tokenizer','bpe','embedding','rope','attention','heads','swiglu','norm',
         'decoder','corpus','batches','optimizer','loop','budget','evaluation','debug','identity','chat']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', choices=NAMES+['all'])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    names=NAMES if args.checkpoint=='all' else [args.checkpoint]
    results={name:run(name) for name in names}
    report={'model':'PicoLLM','torch':str(torch.__version__),'checks':results}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as file:json.dump(report,file,indent=2);file.write('\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
