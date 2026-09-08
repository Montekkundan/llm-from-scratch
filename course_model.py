"""PicoLLM: the original CPU-scale decoder built throughout this course.

PyTorch 2.9+ is the only external dependency. This module intentionally uses
explicit attention to keep the mathematical operations visible. It is not an
optimized inference runtime. The optional cache is an explicit teaching implementation.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 259
    width: int = 64
    heads: int = 4
    layers: int = 2
    context: int = 128
    ff_width: int = 176
    rope_base: float = 10000.0
    eps: float = 1e-5

    def __post_init__(self):
        for name in ("vocab_size", "width", "heads", "layers", "context", "ff_width"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("width/heads must be an even integer for adjacent-pair RoPE")
        if not math.isfinite(self.rope_base) or self.rope_base <= 1:
            raise ValueError("rope_base must be finite and greater than one")
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be finite and positive")


class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        # The course baseline is float32; reductions remain explicit.
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps) * self.weight


def apply_rope(x, positions, base=10000.0):
    """x[B,H,T,d_h], positions[T]; rotate adjacent coordinate pairs."""
    head_width = x.shape[-1]
    inv_freq = base ** (-torch.arange(0, head_width, 2, device=x.device, dtype=x.dtype) / head_width)
    angles = positions.to(device=x.device, dtype=x.dtype)[:, None] * inv_freq[None, :]
    cos, sin = angles.cos()[None, None, :, :], angles.sin()[None, None, :, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class CausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.head_width = config.width // config.heads
        self.rope_base = config.rope_base
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.output = nn.Linear(config.width, config.width, bias=False)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def split(tensor):
            return tensor.reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
        q, k, v = map(split, (q, k, v))
        positions = torch.arange(length, device=x.device)
        q = apply_rope(q, positions, self.rope_base)
        k = apply_rope(k, positions, self.rope_base)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_width)
        forbidden = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        probabilities = scores.masked_fill(forbidden, float("-inf")).softmax(dim=-1)
        combined = (probabilities @ v).transpose(1, 2).contiguous().reshape(batch, length, width)
        return self.output(combined)


class SwiGLU(nn.Module):
    def __init__(self, width, ff_width):
        super().__init__()
        self.gate = nn.Linear(width, ff_width, bias=False)
        self.value = nn.Linear(width, ff_width, bias=False)
        self.output = nn.Linear(ff_width, width, bias=False)

    def forward(self, x):
        return self.output(F.silu(self.gate(x)) * self.value(x))


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config.width, config.eps)
        self.attention = CausalAttention(config)
        self.ffn_norm = RMSNorm(config.width, config.eps)
        self.ffn = SwiGLU(config.width, config.ff_width)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class PicoLLM(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config if config is not None else ModelConfig()
        cfg = self.config
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.width)
        self.blocks = nn.ModuleList(DecoderBlock(cfg) for _ in range(cfg.layers))
        self.final_norm = RMSNorm(cfg.width, cfg.eps)
        self.lm_head = nn.Linear(cfg.width, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        # Initialize each Parameter once, including the shared embedding/head.
        for parameter in self.parameters():
            if parameter.ndim >= 2:
                nn.init.normal_(parameter, mean=0.0, std=0.02)

    def forward(self, input_ids):
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a rank-two torch.long tensor [B,T]")
        if input_ids.shape[0] == 0 or not 1 <= input_ids.shape[1] <= self.config.context:
            raise ValueError("Require B>=1 and 1<=T<=context")
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))

    @torch.inference_mode()
    def forward_cached(self, input_ids, cache=None):
        """Append a nonempty chunk, returning logits and an immutable cache wrapper.

        Cache entries contain already-rotated keys and unrotated values. Training
        and another model instance invalidate a cache; editing a prefix requires
        starting again with cache=None. Call cache.assert_prefix before resuming
        an externally supplied conversation. This path is inference-only.
        """
        if self.training:
            raise ValueError("Cached inference requires model.eval()")
        if input_ids.ndim != 2 or input_ids.dtype != torch.long or not input_ids.numel():
            raise ValueError("input_ids must be nonempty torch.long [B,T]")
        versions = tuple(p._version for p in self.parameters())
        if cache is not None:
            if cache.owner != id(self) or cache.versions != versions:
                raise ValueError("Cache belongs to another model or changed parameters")
            if cache.tokens.shape[0] != input_ids.shape[0] or cache.tokens.device != input_ids.device:
                raise ValueError("Cache batch or device differs")
        offset = 0 if cache is None else cache.tokens.shape[1]
        length = input_ids.shape[1]
        if offset + length > self.config.context:
            raise ValueError("Cached sequence exceeds context")
        x = self.token_embedding(input_ids)
        positions = torch.arange(offset, offset + length, device=x.device)
        updated = []
        for index, block in enumerate(self.blocks):
            attention = block.attention
            normalized = block.attention_norm(x)
            q, k, v = attention.qkv(normalized).chunk(3, -1)
            def split(tensor):
                return tensor.reshape(x.shape[0], length, attention.heads,
                                      attention.head_width).transpose(1, 2)
            q, k, v = map(split, (q, k, v))
            q = apply_rope(q, positions, attention.rope_base)
            k = apply_rope(k, positions, attention.rope_base)
            if cache is not None:
                previous_k, previous_v = cache.keys_values[index]
                k = torch.cat((previous_k, k), -2)
                v = torch.cat((previous_v, v), -2)
            allowed = torch.arange(offset + length, device=x.device)[None, :] <= positions[:, None]
            scores = q @ k.transpose(-2, -1) / math.sqrt(attention.head_width)
            probabilities = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
            joined = (probabilities @ v).transpose(1, 2).contiguous().reshape_as(x)
            x = x + attention.output(joined)
            x = x + block.ffn(block.ffn_norm(x))
            updated.append((k, v))
        tokens = input_ids.clone() if cache is None else torch.cat((cache.tokens, input_ids), 1)
        return self.lm_head(self.final_norm(x)), KVCache(tuple(updated), tokens, id(self), versions)


@dataclass(frozen=True)
class KVCache:
    keys_values: tuple
    tokens: torch.Tensor
    owner: int
    versions: tuple

    def assert_prefix(self, tokens):
        if not torch.equal(self.tokens, tokens):
            raise ValueError("Edited prefix: discard the old cache")


# Existing course integrations keep their import and identical state-dict keys.
DecoderLM = PicoLLM
