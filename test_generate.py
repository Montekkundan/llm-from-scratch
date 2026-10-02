"""Sampling contracts: the default path is unchanged and top-k / top-p filter correctly."""
import math
import unittest

import torch

from course_model import ModelConfig, PicoLLM
from generate import filter_scores, generate_ids
from tokenizer import BOS, EOS, PAD, encode


class FixedLogits(torch.nn.Module):
    """The same next-token logits at every position, so sampling is the only moving part."""

    def __init__(self, logits):
        super().__init__()
        self.config = ModelConfig()
        self.logits = logits

    def forward(self, ids):
        return self.logits.expand(ids.shape[0], ids.shape[1], -1).clone()


def fixed_model(seed=3):
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(259, generator=generator) * 2
    logits[EOS] = -30.0  # keep generation running for the full budget
    return FixedLogits(logits), logits


def original_generate(model, prefix, max_new_tokens, temperature, seed):
    """The pre-top-k loop, verbatim: argmax, or multinomial over softmax(logits / T)."""
    ids = torch.tensor([prefix], dtype=torch.long)
    rng = torch.Generator().manual_seed(seed)
    generated = []
    for _ in range(max_new_tokens):
        scores = model(ids)[0, -1].clone()
        scores[BOS] = scores[PAD] = -torch.inf
        token = int(scores.argmax()) if temperature == 0 else int(
            torch.multinomial((scores / temperature).softmax(-1), 1, generator=rng))
        generated.append(token)
        if token == EOS:
            break
        ids = torch.cat((ids, torch.tensor([[token]], dtype=torch.long)), 1)
    return generated


class DefaultPathTests(unittest.TestCase):
    def test_top_k_zero_and_top_p_one_leave_scores_untouched(self):
        scores = torch.randn(259)
        self.assertIs(filter_scores(scores), scores)
        self.assertIs(filter_scores(scores, 0, 1.0), scores)
        self.assertIs(filter_scores(scores, 259, 1.0), scores)

    def test_defaults_reproduce_the_original_sampling_loop(self):
        torch.manual_seed(2)
        prefix = encode("red fox ", eos=False)
        for model in (fixed_model()[0], PicoLLM().eval()):
            for temperature in (0.0, 0.7, 1.0, 1.5):
                for seed in (1, 7, 19):
                    with self.subTest(model=type(model).__name__, temperature=temperature, seed=seed):
                        expected = original_generate(model, prefix, 16, temperature, seed)
                        self.assertEqual(generate_ids(model, prefix, 16, temperature, seed)["token_ids"], expected)
                        explicit = generate_ids(model, prefix, 16, temperature, seed, top_k=0, top_p=1.0)
                        self.assertEqual(explicit["token_ids"], expected)

    def test_cached_and_uncached_filtered_sampling_agree(self):
        torch.manual_seed(2)
        model, prefix = PicoLLM().eval(), encode("red fox ", eos=False)
        with torch.no_grad():
            model.token_embedding.weight.mul_(40)  # well-separated logits, robust to rounding
        kwargs = dict(max_new_tokens=16, temperature=0.9, seed=5, top_k=20, top_p=0.9)
        self.assertEqual(generate_ids(model, prefix, cached=True, **kwargs)["token_ids"],
                         generate_ids(model, prefix, cached=False, **kwargs)["token_ids"])


class TopKTests(unittest.TestCase):
    def test_top_k_one_is_greedy_at_any_temperature(self):
        torch.manual_seed(2)
        prefix = encode("red fox ", eos=False)
        for model in (fixed_model()[0], PicoLLM().eval()):
            greedy = generate_ids(model, prefix, 16)["token_ids"]
            for temperature in (0.3, 1.0, 2.0):
                for seed in (1, 2, 3):
                    sampled = generate_ids(model, prefix, 16, temperature, seed, top_k=1)["token_ids"]
                    self.assertEqual(sampled, greedy)

    def test_only_the_k_highest_logits_are_ever_sampled(self):
        model, logits = fixed_model()
        prefix = encode("hi", eos=False)
        for k in (2, 5, 40):
            allowed = set(logits.topk(k).indices.tolist()) - {BOS, PAD}
            seen = {generate_ids(model, prefix, 1, 1.5, seed, top_k=k)["token_ids"][0] for seed in range(300)}
            self.assertTrue(seen <= allowed, (k, seen - allowed))
            self.assertGreater(len(seen), 1)  # still sampling, not collapsed to greedy

    def test_filter_keeps_exactly_k_logits(self):
        scores = torch.randn(259)
        for k in (1, 2, 17, 258):
            kept = torch.isfinite(filter_scores(scores, k))
            self.assertEqual(int(kept.sum()), k)
            self.assertEqual(set(kept.nonzero().flatten().tolist()), set(scores.topk(k).indices.tolist()))
        torch.testing.assert_close(filter_scores(scores, 3)[scores.topk(3).indices], scores[scores.topk(3).indices])


class TopPTests(unittest.TestCase):
    def reference_nucleus(self, scores, p):
        """Smallest set of highest-probability tokens whose cumulative mass reaches p."""
        probabilities, order = scores.softmax(-1).sort(descending=True)
        total, kept = 0.0, []
        for probability, index in zip(probabilities.tolist(), order.tolist()):
            kept.append(index)
            total += probability
            if total >= p:
                break
        return set(kept)

    def test_nucleus_is_the_smallest_set_with_mass_at_least_p(self):
        generator = torch.Generator().manual_seed(11)
        for trial in range(20):
            scores = torch.randn(259, generator=generator, dtype=torch.float64) * (1 + trial % 4)
            for p in (0.05, 0.3, 0.5, 0.9, 0.99):
                with self.subTest(trial=trial, p=p):
                    filtered = filter_scores(scores, 0, p)
                    kept = set(torch.isfinite(filtered).nonzero().flatten().tolist())
                    self.assertEqual(kept, self.reference_nucleus(scores, p))
                    probabilities = scores.softmax(-1)
                    mass = probabilities[sorted(kept)].sum().item()
                    self.assertGreaterEqual(mass, p - 1e-12)
                    self.assertLess(mass - probabilities[sorted(kept)].min().item(), p)  # minimal
                    self.assertIn(int(scores.argmax()), kept)

    def test_masked_logits_are_never_sampled(self):
        model, logits = fixed_model()
        prefix = encode("hi", eos=False)
        for p in (0.2, 0.6, 0.9):
            scores = logits.clone()
            scores[BOS] = scores[PAD] = -torch.inf
            allowed = set(torch.isfinite(filter_scores(scores, 0, p)).nonzero().flatten().tolist())
            seen = {generate_ids(model, prefix, 1, 1.0, seed, top_p=p)["token_ids"][0] for seed in range(300)}
            self.assertTrue(seen <= allowed, (p, seen - allowed))
            self.assertNotIn(BOS, seen)
            self.assertNotIn(PAD, seen)

    def test_the_nucleus_is_taken_after_temperature_scaling(self):
        model, logits = fixed_model()
        scores = logits.clone()
        scores[BOS] = scores[PAD] = -torch.inf
        prefix = encode("hi", eos=False)
        for temperature in (0.5, 2.0):
            allowed = set(torch.isfinite(filter_scores(scores / temperature, 0, 0.7)).nonzero().flatten().tolist())
            unscaled = set(torch.isfinite(filter_scores(scores, 0, 0.7)).nonzero().flatten().tolist())
            self.assertNotEqual(allowed, unscaled)
            seen = {generate_ids(model, prefix, 1, temperature, seed, top_p=0.7)["token_ids"][0] for seed in range(300)}
            self.assertTrue(seen <= allowed, (temperature, seen - allowed))

    def test_top_k_is_applied_before_top_p(self):
        scores = torch.randn(259, dtype=torch.float64) * 2
        filtered = filter_scores(scores, 10, 0.8)
        top10 = scores.topk(10).indices
        kept = torch.isfinite(filtered).nonzero().flatten().tolist()
        self.assertTrue(set(kept) <= set(top10.tolist()))
        within = scores[top10].softmax(-1)
        self.assertGreaterEqual(within[torch.isfinite(filtered[top10])].sum().item(), 0.8 - 1e-12)

    def test_a_tiny_p_keeps_only_the_best_token(self):
        scores = torch.randn(259)
        kept = torch.isfinite(filter_scores(scores, 0, 1e-9)).nonzero().flatten().tolist()
        self.assertEqual(kept, [int(scores.argmax())])


class ArgumentTests(unittest.TestCase):
    def test_invalid_filters_are_rejected(self):
        model, _ = fixed_model()
        prefix = encode("hi", eos=False)
        for kwargs in ({"top_k": -1}, {"top_k": 1.5}, {"top_k": True}, {"top_p": 0.0},
                       {"top_p": 1.5}, {"top_p": -0.1}, {"top_p": math.nan}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    generate_ids(model, prefix, 2, 1.0, **kwargs)
                with self.assertRaises(ValueError):
                    filter_scores(torch.randn(259), **kwargs)

    def test_greedy_decoding_ignores_valid_filters(self):
        model, _ = fixed_model()
        prefix = encode("hi", eos=False)
        self.assertEqual(generate_ids(model, prefix, 6, 0.0, top_k=3, top_p=0.1)["token_ids"],
                         generate_ids(model, prefix, 6, 0.0)["token_ids"])


if __name__ == "__main__":
    unittest.main()
