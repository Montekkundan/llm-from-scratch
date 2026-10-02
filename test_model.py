"""Oracle tests for PicoLLM's operators: attention scale and mask, RoPE and cache ownership.

The model's own weights are tiny (std 0.02), which makes attention scores almost zero and
hides a wrong scale. These tests use float64 and weights large enough that scores are
O(1), so removing or changing 1/sqrt(head_width) changes the output by far more than
the tolerance.
"""
import copy
import gc
import math
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

import course_model
from course_model import CausalAttention, ModelConfig, PicoLLM, apply_rope
from gpu_model import GPUPicoLLM

# (width, heads): head widths 2, 4, 8, 16, 64, 16, 16, 32 and 12.
HEAD_LAYOUTS = [(4, 2), (8, 2), (16, 2), (32, 2), (64, 1), (64, 4), (48, 3), (96, 3), (24, 2)]


def loud(module, width, seed=0):
    """Float64 matrices with std sqrt(2/width): correctly scaled scores have std about 2."""
    generator = torch.Generator().manual_seed(seed)
    module = module.double()
    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.ndim >= 2:
                parameter.copy_(torch.randn(parameter.shape, generator=generator,
                                            dtype=torch.float64) * math.sqrt(2.0 / width))
    return module


def rotate_reference(x, positions, base=10000.0):
    """Complex RoPE: the pair (x[2j], x[2j+1]) is multiplied by exp(i * position * base^(-2j/d))."""
    d = x.shape[-1]
    pairs = torch.view_as_complex(x.reshape(*x.shape[:-1], d // 2, 2).contiguous())
    theta = base ** (-2.0 * torch.arange(d // 2, dtype=x.dtype) / d)
    angle = positions.to(x.dtype)[:, None] * theta[None, :]
    return torch.view_as_real(pairs * torch.polar(torch.ones_like(angle), angle)).reshape(x.shape)


def reference_attention(attention, x, scale, explicit):
    """softmax(Q K^T * scale + mask) V from the module's own projections, RoPE by complex rotation."""
    batch, length, width = x.shape
    q, k, v = (t.reshape(batch, length, attention.heads, attention.head_width).transpose(1, 2)
               for t in attention.qkv(x).chunk(3, -1))
    positions = torch.arange(length)
    q, k = (rotate_reference(t, positions, attention.rope_base) for t in (q, k))
    if explicit:
        mask = torch.full((length, length), -torch.inf, dtype=x.dtype).triu(1)
        joined = (q @ k.transpose(-2, -1) * scale + mask).softmax(-1) @ v
    else:
        joined = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
    return attention.output(joined.transpose(1, 2).reshape(batch, length, width))


class AttentionOracleTests(unittest.TestCase):
    def build(self, width, heads):
        torch.manual_seed(width * 10 + heads)
        attention = loud(CausalAttention(ModelConfig(width=width, heads=heads)), width, seed=width + heads)
        return attention, torch.randn(2, 7, width, dtype=torch.float64)

    def test_matches_sdpa_and_explicit_softmax_for_every_head_width(self):
        for width, heads in HEAD_LAYOUTS:
            with self.subTest(width=width, heads=heads):
                attention, x = self.build(width, heads)
                scale = 1 / math.sqrt(width // heads)
                sdpa = reference_attention(attention, x, None, explicit=False)
                explicit = reference_attention(attention, x, scale, explicit=True)
                torch.testing.assert_close(sdpa, explicit, rtol=1e-10, atol=1e-10)
                torch.testing.assert_close(attention(x), sdpa, rtol=1e-10, atol=1e-10)
                torch.testing.assert_close(attention(x), explicit, rtol=1e-10, atol=1e-10)

    def test_the_oracle_is_sensitive_to_a_wrong_scale(self):
        # Guards the fixture: a missing factor or the model width instead of the head
        # width must move the output by far more than the comparison tolerance.
        for width, heads in HEAD_LAYOUTS:
            with self.subTest(width=width, heads=heads):
                attention, x = self.build(width, heads)
                actual = attention(x)
                wrong = [1.0, 1 / math.sqrt(width // heads) ** 2]
                if heads > 1:
                    wrong.append(1 / math.sqrt(width))
                for scale in wrong:
                    other = reference_attention(attention, x, scale, explicit=True)
                    self.assertGreater((actual - other).abs().max().item(), 1e-3, scale)

    def test_future_tokens_cannot_change_earlier_outputs(self):
        attention, x = self.build(64, 4)
        for cut in range(1, x.shape[1]):
            changed = x.clone()
            changed[:, cut:] = torch.randn_like(changed[:, cut:])
            torch.testing.assert_close(attention(x)[:, :cut], attention(changed)[:, :cut],
                                       rtol=0, atol=1e-12)


class RotaryTests(unittest.TestCase):
    def test_matches_complex_rotation_for_adjacent_pairs(self):
        positions = torch.tensor([0, 1, 2, 5, 6, 7, 30, 31, 200])
        for head_width in (2, 4, 16, 64):
            for base in (10000.0, 500.0):
                with self.subTest(head_width=head_width, base=base):
                    x = torch.randn(2, 3, len(positions), head_width, dtype=torch.float64)
                    torch.testing.assert_close(apply_rope(x, positions, base),
                                               rotate_reference(x, positions, base),
                                               rtol=1e-12, atol=1e-12)

    def test_rotation_preserves_every_vector_norm(self):
        x = torch.randn(2, 3, 9, 16, dtype=torch.float64)
        rotated = apply_rope(x, torch.arange(9) * 37)
        torch.testing.assert_close(rotated.norm(dim=-1), x.norm(dim=-1), rtol=1e-12, atol=1e-12)

    def test_score_depends_only_on_the_position_difference(self):
        # The same query and key content at every position: S[i, j] must be constant
        # along each diagonal j - i = const, and different diagonals must differ.
        torch.manual_seed(5)
        length, head_width = 12, 16
        q = torch.randn(1, 1, 1, head_width, dtype=torch.float64).expand(1, 1, length, head_width)
        k = torch.randn(1, 1, 1, head_width, dtype=torch.float64).expand(1, 1, length, head_width)
        positions = torch.arange(length)
        scores = (apply_rope(q, positions) @ apply_rope(k, positions).transpose(-2, -1))[0, 0]
        for offset in range(-length + 1, length):
            diagonal = torch.diagonal(scores, offset=offset)
            torch.testing.assert_close(diagonal, diagonal[:1].expand_as(diagonal), rtol=1e-12, atol=1e-12)
        self.assertGreater((scores[0, 1:] - scores[0, 1]).abs().max().item(), 1e-3)

    def test_common_shift_leaves_distinct_vector_scores_unchanged(self):
        torch.manual_seed(6)
        q = torch.randn(1, 2, 8, 16, dtype=torch.float64)
        k = torch.randn(1, 2, 8, 16, dtype=torch.float64)
        positions = torch.arange(8)
        base = apply_rope(q, positions) @ apply_rope(k, positions).transpose(-2, -1)
        for shift in (1, 17, 1000):
            shifted = apply_rope(q, positions + shift) @ apply_rope(k, positions + shift).transpose(-2, -1)
            torch.testing.assert_close(shifted, base, rtol=1e-9, atol=1e-9)


CACHE_CONFIG = ModelConfig(vocab_size=32, width=16, heads=2, layers=2, context=24, ff_width=32)


def small_model(seed=1, cls=PicoLLM):
    torch.manual_seed(seed)
    return cls(CACHE_CONFIG).eval()


class CachedForwardTests(unittest.TestCase):
    def test_cached_logits_match_full_prefix_for_sharp_attention(self):
        # forward and forward_cached each carry their own 1/sqrt(head_width) and RoPE
        # offset; with non-negligible scores a change to either one alone separates them.
        model = loud(small_model(), CACHE_CONFIG.width)
        ids = torch.randint(0, CACHE_CONFIG.vocab_size, (2, CACHE_CONFIG.context))
        full = model(ids)
        for cut in (1, 5, CACHE_CONFIG.context - 1):
            with self.subTest(cut=cut):
                logits, cache = model.forward_cached(ids[:, :cut])
                torch.testing.assert_close(logits, full[:, :cut], rtol=1e-10, atol=1e-10)
                logits, cache = model.forward_cached(ids[:, cut:], cache)
                torch.testing.assert_close(logits, full[:, cut:], rtol=1e-10, atol=1e-10)
                cache.assert_prefix(ids)
        _, cache = model.forward_cached(ids[:, :1])
        for index in range(1, ids.shape[1]):
            logits, cache = model.forward_cached(ids[:, index:index + 1], cache)
            torch.testing.assert_close(logits[:, 0], full[:, index], rtol=1e-10, atol=1e-10)


class CacheOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.ids = torch.tensor([[1, 2, 3]])
        self.next = torch.tensor([[4]])

    def test_the_owning_model_accepts_its_cache(self):
        model = small_model()
        _, cache = model.forward_cached(self.ids)
        logits, _ = model.forward_cached(self.next, cache)
        self.assertEqual(logits.shape, (1, 1, CACHE_CONFIG.vocab_size))

    def test_another_instance_with_identical_weights_rejects_the_cache(self):
        owner = small_model()
        _, cache = owner.forward_cached(self.ids)
        for other in (small_model(), copy.deepcopy(owner)):
            with self.assertRaisesRegex(ValueError, "Cache belongs to another model"):
                other.forward_cached(self.next, cache)

    def test_a_forced_id_collision_cannot_make_another_model_accept_the_cache(self):
        # Two live models never share an id; force it to prove the guard is not id based.
        first, second = small_model(), small_model()
        with mock.patch.object(course_model, "id", lambda value: 4242, create=True):
            _, cache = first.forward_cached(self.ids)
            with self.assertRaisesRegex(ValueError, "Cache belongs to another model"):
                second.forward_cached(self.next, cache)

    def test_a_new_model_that_reuses_a_freed_models_id_rejects_its_cache(self):
        # CPython usually hands a freed object's address to a later object of the same
        # size, so building many models after the owner is freed reproduces the id reuse.
        accepted = []
        for _ in range(5):
            model = small_model()
            _, cache = model.forward_cached(self.ids)
            del model
            gc.collect()
            candidates = [small_model() for _ in range(40)]
            for other in candidates:
                try:
                    other.forward_cached(self.next, cache)
                except ValueError:
                    pass
                else:
                    accepted.append(id(other))
        self.assertEqual(accepted, [], "a stale cache was accepted by a different model")

    def test_the_gpu_variant_has_per_instance_ownership_too(self):
        model, other = small_model(cls=GPUPicoLLM), small_model(cls=GPUPicoLLM)
        _, cache = model.forward_cached(self.ids)
        model.forward_cached(self.next, cache)
        with self.assertRaisesRegex(ValueError, "Cache belongs to another model"):
            other.forward_cached(self.next, cache)

    def test_changed_weights_and_edited_prefixes_are_still_rejected(self):
        model = small_model()
        _, cache = model.forward_cached(self.ids)
        with self.assertRaisesRegex(ValueError, "Edited prefix"):
            cache.assert_prefix(self.ids + 1)
        with torch.no_grad():
            model.token_embedding.weight.add_(0.01)
        with self.assertRaisesRegex(ValueError, "changed parameters"):
            model.forward_cached(self.next, cache)


if __name__ == "__main__":
    unittest.main()
