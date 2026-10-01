"""SDPA execution for PicoLLM with the original parameter names."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from course_model import ModelConfig, PicoLLM, CausalAttention, RMSNorm, SwiGLU


class FP32RMSNorm(RMSNorm):
    def forward(self, x):
        values = x.float()
        normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(x.dtype)


def fp32_rope(x, positions, base):
    width = x.shape[-1]
    frequency = base ** (-torch.arange(0, width, 2, device=x.device, dtype=torch.float32) / width)
    angles = positions.to(device=x.device, dtype=torch.float32)[:, None] * frequency[None, :]
    cosine, sine = angles.cos()[None, None], angles.sin()[None, None]
    values = x.float()
    even, odd = values[..., 0::2], values[..., 1::2]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), -1).flatten(-2).to(x.dtype)


class SDPAAttention(CausalAttention):
    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q, k, v = [value.reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
                   for value in (q, k, v)]
        positions = torch.arange(length, device=x.device)
        q = fp32_rope(q, positions, self.rope_base)
        k = fp32_rope(k, positions, self.rope_base)
        joined = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        return self.output(joined.transpose(1, 2).contiguous().reshape(batch, length, width))


class GPUDecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = FP32RMSNorm(config.width, config.eps)
        self.attention = SDPAAttention(config)
        self.ffn_norm = FP32RMSNorm(config.width, config.eps)
        self.ffn = SwiGLU(config.width, config.ff_width)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class GPUPicoLLM(PicoLLM):
    def __init__(self, config=None):
        nn.Module.__init__(self)
        self.config = config or ModelConfig()
        cfg = self.config
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.width)
        self.blocks = nn.ModuleList(GPUDecoderBlock(cfg) for _ in range(cfg.layers))
        self.final_norm = FP32RMSNorm(cfg.width, cfg.eps)
        self.lm_head = nn.Linear(cfg.width, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        for parameter in self.parameters():
            if parameter.ndim >= 2:
                nn.init.normal_(parameter, mean=0.0, std=0.02)

    def hidden_states(self, input_ids):
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be torch.long [B,T]")
        if input_ids.shape[0] < 1 or not 1 <= input_ids.shape[1] <= self.config.context:
            raise ValueError("Require B>=1 and 1<=T<=context")
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)

    def forward(self, input_ids):
        return self.lm_head(self.hidden_states(input_ids))

    def loss(self, input_ids, targets, chunk_size=128):
        if targets.shape != input_ids.shape or targets.dtype != torch.long:
            raise ValueError("Next-token targets must be torch.long with the input shape")
        if chunk_size < 1:
            raise ValueError("Loss chunk size must be positive")
        hidden = self.hidden_states(input_ids)

        def chunk_loss(values, labels):
            logits = self.lm_head(values).float()
            return F.cross_entropy(logits.reshape(-1, self.config.vocab_size),
                                   labels.reshape(-1), reduction="sum")

        total = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, input_ids.shape[1], chunk_size):
            values, labels = hidden[:, start:start + chunk_size], targets[:, start:start + chunk_size]
            if torch.is_grad_enabled():
                value = checkpoint(chunk_loss, values, labels, use_reentrant=False,
                                   preserve_rng_state=False)
            else:
                value = chunk_loss(values, labels)
            total = total + value
        return total / targets.numel()
