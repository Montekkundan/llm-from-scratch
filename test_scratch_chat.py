"""Offline proofs for scratch ChatML, assistant masking and exact SFT resume."""
from dataclasses import asdict
import copy
import json
from pathlib import Path
import unittest
import uuid

import torch
from torch.nn import functional as F

from chat_data import NEUTRAL_SYSTEM, SYSTEM, prepare
from course_model import ModelConfig
from test_chat_data import NativeFixtureTokenizer, pair
from gpu_checkpoint import (load_checkpoint, remove_tree, resolve_checkpoint,
                            save_checkpoint, sha256)
from gpu_model import GPUPicoLLM
from gpu_train import TrainingConfig
from scratch_chat import evaluate_cases, generate_ids, load_weights
from scratch_sft import (assistant_loss_sum, collate, load_prepared,
                         train_sft, validate_assistant)

EXPERIMENTS = Path(__file__).resolve().parents[2]


class TinyTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return [1, 3]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(value) for value in ids if not skip_special_tokens or value not in (0, 2))


class MaskAndGenerationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.config = ModelConfig(vocab_size=32, width=16, heads=2, layers=1, context=16, ff_width=32)
        self.model = GPUPicoLLM(self.config)

    def test_causal_masked_chunk_loss_matches_real_logits_and_gradients(self):
        ids = torch.tensor([[1, 3, 4, 5, 2], [1, 6, 7, 2, 0]])
        labels = torch.tensor([[-100, -100, 4, 5, 2], [-100, -100, -100, 2, -100]])
        reference = copy.deepcopy(self.model)
        manual = F.cross_entropy(reference(ids[:, :-1]).reshape(-1, 32),
                                 labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
        actual, targets = assistant_loss_sum(self.model, ids, labels, chunk_size=2)
        self.assertEqual(targets, 4)
        torch.testing.assert_close(actual, manual, rtol=1e-6, atol=1e-6)
        (actual / targets).backward()
        (manual / targets).backward()
        for parameter, other in zip(self.model.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter.grad, other.grad, rtol=1e-5, atol=1e-6)

    def test_accumulation_uses_total_supervised_tokens_not_microbatch_means(self):
        rows = [{"input_ids": [1, 3, 4, 5, 2], "labels": [-100, -100, 4, 5, 2]},
                {"input_ids": [1, 6, 2], "labels": [-100, -100, 2]}]
        reference = copy.deepcopy(self.model)
        ids, labels = collate(rows, torch.device("cpu"))
        full, count = assistant_loss_sum(reference, ids, labels, 2)
        (full / count).backward()
        micro_counts = []
        for row in rows:
            inputs, targets = collate([row], torch.device("cpu"))
            numerator, supervised = assistant_loss_sum(self.model, inputs, targets, 2)
            micro_counts.append(supervised)
            (numerator / count).backward()
        self.assertEqual(micro_counts, [3, 1])
        for parameter, other in zip(self.model.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter.grad, other.grad, rtol=1e-5, atol=1e-6)

    def test_generation_uses_actual_weights_and_both_eos_ids(self):
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.zero_()
        ids, reason = generate_ids(self.model, [1], 4)
        self.assertEqual((ids, reason), ([0], "eos_document"))
        with torch.no_grad():
            self.model.final_norm.weight.fill_(1)
            self.model.token_embedding.weight[1].fill_(0.1)
            self.model.token_embedding.weight[2].fill_(1)
        ids, reason = generate_ids(self.model, [1], 4)
        self.assertEqual((ids, reason), ([2], "eos_chatml"))

    def test_generation_respects_context_and_reports_every_identity_condition(self):
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.zero_()
            self.model.final_norm.weight.fill_(1)
            self.model.token_embedding.weight[1].fill_(0.1)
            self.model.token_embedding.weight[3].fill_(1)
        ids, reason = generate_ids(self.model, [1] * 14, 4)
        self.assertEqual(ids, [3, 3])
        self.assertEqual(reason, "context")
        with self.assertRaisesRegex(ValueError, "no room"):
            generate_ids(self.model, [1] * 16, 4)
        with self.assertRaises(ValueError):
            generate_ids(self.model, [1], 0)
        results = evaluate_cases(self.model, TinyTokenizer(),
            [{"id": "identity", "category": "identity", "messages": [{"role": "user", "content": "Your name?"}]}],
            max_new_tokens=2)
        self.assertEqual([row["condition"] for row in results], ["course_identity", "neutral_identity"])
        self.assertTrue(all(row["output_token_ids"] == [3, 3] for row in results))
        self.assertNotEqual(results[0]["effective_prompt_sha256"], results[1]["effective_prompt_sha256"])


class ScratchSFTTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.root = EXPERIMENTS / "test-artifacts" / ("scratch-" + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.config = ModelConfig(vocab_size=32, width=16, heads=2, layers=1, context=16, ff_width=32)
        self.metadata = {"model": "fixture", "revision": "fixed", "vocab_size": 32}
        self.rows = []
        for index in range(8):
            length = index % 3 + 1
            ids = [1, index + 3, 9] + [10] * length + [2]
            self.rows.append({"input_ids": ids, "attention_mask": [1] * len(ids),
                             "labels": [-100] * 3 + ids[3:]})
        self.heldout = [{"input_ids": [1, 25, 6, 2], "attention_mask": [1] * 4,
                         "labels": [-100, -100, 6, 2]}]
        self.manifest = {"schema": 1, "experiment_type": "scratch", "model_kind": "scratch",
                         "pretrained_model_weights": False, "system": SYSTEM, "identity_training_system": NEUTRAL_SYSTEM,
                         "model_id": "fixture",
                         "model_revision": "fixed", "tokenizer_revision": "fixed",
                         "context": 16, "template_sha256": "a" * 64}
        self.publish()
        self.document = {"epochs": 1, "training": {"steps": 4, "batch_size": 1, "grad_accum": 2,
            "lr": 0.005, "warmup_steps": 1, "dtype": "float32", "validation_every": 2,
            "validation_batches": 2, "loss_chunk_size": 2, "checkpoint_keep": 1, "log_every": 100}}
        self.config_path = self.root / "sft.json"
        self.config_path.write_text(json.dumps(self.document))
        torch.manual_seed(11)
        model = GPUPicoLLM(self.config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        model.loss(torch.tensor([[1, 3]]), torch.tensor([[3, 4]])).backward()
        optimizer.step()
        state = {"global_step": 1, "tokens": 2, "status": "complete", "config_sha256": "base-config",
                 "data_sha256": "base-data", "resolved_config": {"model": asdict(self.config)},
                 "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "data_identity": {"tokenizer": self.metadata}}
        self.base = save_checkpoint(self.root / "base", state, keep=1)

    def tearDown(self):
        remove_tree(self.root)

    def publish(self):
        splits = {}
        for split, rows in (("train", self.rows), ("heldout", self.heldout)):
            path = self.root / (split + ".jsonl")
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            splits[split] = {"sha256": sha256(path), "examples": len(rows),
                "input_tokens": sum(len(row["input_ids"]) for row in rows),
                "supervised_tokens": sum(sum(label != -100 for label in row["labels"][1:]) for row in rows)}
        self.manifest["splits"] = splits
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))

    def saved(self, name):
        return load_checkpoint(resolve_checkpoint(self.root / name, "latest"))

    def test_exact_resume_with_fresh_sft_optimizer_and_loaded_chat_artifact(self):
        full = train_sft(self.config_path, self.root, self.root / "full", self.base, "cpu", save_every="1steps")
        partial = train_sft(self.config_path, self.root, self.root / "resumed", self.base, "cpu", max_steps=2)
        self.assertEqual(partial["status"], "paused_limit")
        resumed = train_sft(self.config_path, self.root, self.root / "resumed", device_name="cpu",
                            resume="latest", save_every="1steps")
        self.assertEqual(full["status"], "complete")
        self.assertEqual(resumed["status"], "complete")
        first, second = self.saved("full"), self.saved("resumed")
        for name in first["model"]:
            self.assertTrue(torch.equal(first["model"][name], second["model"][name]), name)
        for parameter, values in first["optimizer"]["state"].items():
            self.assertEqual(int(values["step"]), 4)
            for name, value in values.items():
                other = second["optimizer"]["state"][parameter][name]
                self.assertTrue(torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other)
        for key in ("scheduler", "cursor", "tokens", "supervised_tokens", "best_validation"):
            self.assertEqual(first[key], second[key])
        self.assertTrue(torch.equal(first["rng"]["cpu"], second["rng"]["cpu"]))
        self.assertEqual(first["base_checkpoint_sha256"], sha256(self.base / "checkpoint.pt"))
        self.assertEqual(first["supervised_tokens"], self.manifest["splits"]["train"]["supervised_tokens"])
        self.assertEqual(first["tokens"], self.manifest["splits"]["train"]["input_tokens"])
        model, state, _ = load_weights(self.root / "resumed", "cpu")
        self.assertEqual(state["stage"], "scratch_sft")
        ids, reason = generate_ids(model, [1, 3], 3)
        self.assertGreaterEqual(len(ids), 1)
        self.assertIn(reason, ("length", "eos_chatml", "eos_document"))

    def test_heldout_loss_is_weighted_only_over_causally_shifted_assistant_targets(self):
        model, _, _ = load_weights(self.base, "cpu")
        config = TrainingConfig(steps=4, warmup_steps=1, dtype="float32", batch_size=1)
        result = validate_assistant(model, self.heldout, config, torch.device("cpu"))
        ids, labels = collate(self.heldout, torch.device("cpu"))
        with torch.inference_mode():
            expected = F.cross_entropy(model(ids[:, :-1]).reshape(-1, 32), labels[:, 1:].reshape(-1),
                                       ignore_index=-100, reduction="sum") / 2
        self.assertEqual(result["supervised_tokens"], 2)
        self.assertAlmostEqual(result["assistant_token_loss"], float(expected), places=5)

    def test_data_integrity_refuses_malformed_or_wrong_identity_training(self):
        self.manifest["experiment_type"] = "pretrained"
        self.publish()
        with self.assertRaisesRegex(ValueError, "scratch-specific"):
            load_prepared(self.root, self.metadata, 16, 32)
        self.manifest["experiment_type"] = "scratch"
        self.rows[0]["labels"][3] = 11
        self.publish()
        with self.assertRaisesRegex(ValueError, "original token"):
            load_prepared(self.root, self.metadata, 16, 32)
        self.rows[0]["labels"][3] = self.rows[0]["input_ids"][3]
        self.heldout = [copy.deepcopy(self.rows[0])]
        self.publish()
        with self.assertRaisesRegex(ValueError, "overlap"):
            load_prepared(self.root, self.metadata, 16, 32)

    def test_changed_base_checkpoint_is_refused_before_model_use(self):
        file = self.base / "checkpoint.pt"
        with file.open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_weights(self.base, "cpu")
        with self.assertRaisesRegex(ValueError, "checksum"):
            train_sft(self.config_path, self.root, self.root / "bad-base", self.base, "cpu")

    def test_real_scratch_chat_data_preparation_loads_without_rewriting_rows(self):
        config = {"experiment_type": "scratch", "context": 1024, "heldout_examples": 1,
                  "identity_repeats": 1, "target_input_tokens": 100, "seed": 7,
                  "model_id": "fixture", "model_revision": "fixed", "dataset_id": "fixture",
                  "dataset_revision": "fixed", "identity": "scratch fixture"}
        output = self.root / "prepared"
        manifest = prepare(config, output, NativeFixtureTokenizer(), [],
                           [{"messages": pair("Independent held-out source question.", "One held-out answer.")}])
        train, heldout, loaded, identity = load_prepared(output, self.metadata, 1024, 32768)
        self.assertEqual(loaded, manifest)
        self.assertEqual(len(train), manifest["splits"]["train"]["examples"])
        self.assertEqual(len(heldout), 1)
        self.assertEqual(identity["experiment_type"], "scratch")
        self.assertEqual(identity["tokenizer"], self.metadata)

    def test_resume_rejects_changed_config_and_prepared_file(self):
        train_sft(self.config_path, self.root, self.root / "strict", self.base, "cpu", max_steps=1)
        self.document["training"]["lr"] = 0.01
        self.config_path.write_text(json.dumps(self.document))
        with self.assertRaisesRegex(ValueError, "differ"):
            train_sft(self.config_path, self.root, self.root / "strict", device_name="cpu", resume="latest")
        self.document["training"]["lr"] = 0.005
        self.config_path.write_text(json.dumps(self.document))
        (self.root / "train.jsonl").write_text("changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            train_sft(self.config_path, self.root, self.root / "strict", device_name="cpu", resume="latest")

    def test_stop_is_an_optimizer_boundary_and_short_final_update_is_retained(self):
        calls = [0]

        def stop():
            calls[0] += 1
            return calls[0] > 1

        result = train_sft(self.config_path, self.root, self.root / "stopped", self.base, "cpu", stop_requested=stop)
        self.assertEqual(result["status"], "paused_signal")
        self.assertEqual(result["examples_seen"], 2)
        self.rows = self.rows[:3]
        self.publish()
        self.document["training"].pop("steps")
        self.config_path.write_text(json.dumps(self.document))
        result = train_sft(self.config_path, self.root, self.root / "short", self.base, "cpu")
        self.assertEqual(result["global_step"], 2)
        self.assertEqual(result["examples_seen"], 3)
        self.assertEqual(result["supervised_tokens"], self.manifest["splits"]["train"]["supervised_tokens"])


if __name__ == "__main__":
    unittest.main()
