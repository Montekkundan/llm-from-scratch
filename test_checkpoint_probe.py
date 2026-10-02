"""Real tiny-model sampling and immutable-checkpoint integrity proofs."""
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import torch

from checkpoint_probe import (CHAT_CASES, FROZEN_SUITE, PRETRAIN_PROMPTS, check_prompt_isolation,
                              probe_scratch, read_loss_log, recent_losses, sample_signals)
from course_model import ModelConfig
from gpu_checkpoint import remove_tree, save_checkpoint, sha256
from gpu_model import GPUPicoLLM
from scratch_chat import generate_ids, load_weights


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) % 29 + 3 for character in text]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(token) for token in ids if token not in (0, 2))

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return [1, 3, 4]


class ProbeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(71)
        self.root = Path(tempfile.mkdtemp(prefix="probe-"))
        self.config = ModelConfig(vocab_size=64, width=16, heads=2, layers=1, context=128, ff_width=32)
        self.model = GPUPicoLLM(self.config)
        self.tokenizer = TinyTokenizer()
        self.state = {"global_step": 9, "tokens": 1152, "status": "running",
                      "config_sha256": "fixture-config", "data_sha256": "fixture-data",
                      "model": self.model.state_dict(), "resolved_config": {"model": asdict(self.config)},
                      "data_identity": {"tokenizer": {"model": "fixture", "revision": "fixed"}},
                      "validation": {"nll": 4.2}}
        self.checkpoint = save_checkpoint(self.root, self.state, keep=1)

    def tearDown(self):
        remove_tree(self.root)

    def test_actual_cpu_samples_are_deterministic_and_leave_checkpoint_unchanged(self):
        digest = sha256(self.checkpoint / "checkpoint.pt")
        first = probe_scratch(self.checkpoint, None, "pretrain", "cpu", "float32", 8, self.tokenizer)
        second = probe_scratch(self.checkpoint, None, "pretrain", "cpu", "float32", 8, self.tokenizer)
        self.assertEqual(first, second)
        self.assertEqual(first["global_step"], 9)
        self.assertEqual(first["consumed_tokens"], 1152)
        self.assertEqual(first["checkpoint_sha256"], digest)
        self.assertEqual(len(first["results"]), 3)
        loaded, _, _ = load_weights(self.checkpoint, "cpu")
        expected, reason = generate_ids(loaded, self.tokenizer.encode(PRETRAIN_PROMPTS[0]), 8)
        self.assertEqual(first["results"][0]["output_token_ids"], expected)
        self.assertEqual(first["results"][0]["finish_reason"], reason)
        self.assertGreaterEqual(len(expected), 1)
        self.assertEqual(sha256(self.checkpoint / "checkpoint.pt"), digest)

    def test_changed_checkpoint_refuses_sampling(self):
        with (self.checkpoint / "checkpoint.pt").open("ab") as stream:
            stream.write(b"corrupted")
        with self.assertRaisesRegex(ValueError, "checksum"):
            probe_scratch(self.checkpoint, None, tokenizer=self.tokenizer)

    def test_sft_chat_uses_real_generator_and_both_identity_conditions(self):
        self.state["stage"] = "scratch_sft"
        checkpoint = save_checkpoint(self.root, self.state, keep=1)
        report = probe_scratch(checkpoint, None, "scratch_sft", tokenizer=self.tokenizer, max_new_tokens=4)
        self.assertEqual(len(report["results"]), len(CHAT_CASES) + 1)
        identity = [row for row in report["results"] if row["id"] == "monitor-identity"]
        self.assertEqual([row["condition"] for row in identity], ["course_identity", "neutral_identity"])
        self.assertTrue(all("signals" in row for row in report["results"]))
        with self.assertRaisesRegex(ValueError, "stage"):
            probe_scratch(checkpoint, None, "pretrain", tokenizer=self.tokenizer)

    def test_latest_pointer_is_not_an_immutable_probe_target(self):
        with self.assertRaisesRegex(ValueError, "immutable"):
            probe_scratch(self.root, None, tokenizer=self.tokenizer)

    def test_signals_flag_loops_without_editing_generated_tokens(self):
        ids = [3, 4] * 20 + [2]
        original = ids.copy()
        signals = sample_signals(ids, "a repeated continuation", "eos_chatml")
        self.assertEqual(ids, original)
        self.assertTrue(signals["repetition_flag"])
        self.assertFalse(signals["length_limited"])
        self.assertEqual(signals["longest_identical_token_run"], 1)
        self.assertTrue(sample_signals([3] * 8, "sample", "length")["length_limited"])
        self.assertTrue(sample_signals([0], "", "eos_document")["empty_response"])

    def test_losses_never_include_steps_after_the_checkpoint(self):
        values = recent_losses([{"step": 2, "loss": 5.0}, {"step": 6, "eval_loss": 4.0},
                                {"global_step": 9, "loss": 3.0}, {"global_step": 10, "loss": 2.0}], 9)
        self.assertEqual(values, {"loss": {"step": 9, "value": 3.0},
                                  "eval_loss": {"step": 6, "value": 4.0}})

    def test_live_log_partial_last_line_does_not_hide_complete_losses(self):
        path = self.root / "metrics.jsonl"
        path.write_text('{"global_step": 9, "loss": 3.5}\n{"global_step": 10, "loss":')
        self.assertEqual(read_loss_log(path, 9), {"loss": {"step": 9, "value": 3.5}})

    def test_monitor_phrases_are_disjoint_from_the_frozen_final_suite(self):
        digest = check_prompt_isolation(FROZEN_SUITE)
        self.assertEqual(len(digest), 64)


if __name__ == "__main__":
    unittest.main()
