"""Tiny CPU verification for SDPA, packed streams and checkpoint continuation."""
from array import array
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F

from course_model import ModelConfig, PicoLLM
from gpu_checkpoint import compatible_data, load_checkpoint, resolve_checkpoint, save_checkpoint, sha256
from gpu_model import GPUPicoLLM
from gpu_train import TokenStream, TrainingConfig, learning_rate, read_data, save_interval, train


EXPERIMENTS = Path(__file__).resolve().parents[2]


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = ModelConfig(vocab_size=32, width=16, heads=2, layers=2, context=8, ff_width=32)

    def test_original_keys_logits_and_gradients_match_sdpa(self):
        torch.manual_seed(7)
        original = PicoLLM(self.config)
        torch.manual_seed(7)
        optimized = GPUPicoLLM(self.config)
        self.assertEqual(list(original.state_dict()), list(optimized.state_dict()))
        for name, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, optimized.state_dict()[name]))
        inputs = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
        labels = torch.tensor([[2, 3, 4, 5], [3, 2, 1, 0]])
        torch.testing.assert_close(original(inputs), optimized(inputs), rtol=1e-5, atol=1e-6)
        first = F.cross_entropy(original(inputs).reshape(-1, 32), labels.reshape(-1))
        second = optimized.loss(inputs, labels, chunk_size=2)
        first.backward(); second.backward()
        torch.testing.assert_close(first, second, rtol=1e-6, atol=1e-6)
        for (name, parameter), (other_name, other) in zip(original.named_parameters(), optimized.named_parameters()):
            self.assertEqual(name, other_name)
            torch.testing.assert_close(parameter.grad, other.grad, rtol=2e-4, atol=2e-6)
        self.assertIs(optimized.lm_head.weight, optimized.token_embedding.weight)

    def test_manual_attention_and_double_precision_gradcheck(self):
        torch.manual_seed(3)
        values = tuple(torch.randn(1, 2, 3, 4, dtype=torch.float64, requires_grad=True) for _ in range(3))
        q, k, v = values
        forbidden = torch.ones(3, 3, dtype=torch.bool).triu(1)
        manual = ((q @ k.transpose(-2, -1) / math.sqrt(4)).masked_fill(forbidden, -torch.inf)
                  .softmax(-1) @ v)
        actual = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        torch.testing.assert_close(manual, actual, rtol=1e-12, atol=1e-12)
        self.assertTrue(torch.autograd.gradcheck(
            lambda a, b, c: F.scaled_dot_product_attention(a, b, c, is_causal=True, dropout_p=0.0),
            values, eps=1e-6, atol=1e-5, rtol=1e-4))

    def test_bfloat16_autocast_loss_and_gradient_are_finite(self):
        model = GPUPicoLLM(self.config)
        inputs = torch.tensor([[1, 2, 3, 4]])
        targets = torch.tensor([[2, 3, 4, 5]])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = model.loss(inputs, targets, chunk_size=2)
        loss.backward()
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                            for parameter in model.parameters()))


class TrainerTests(unittest.TestCase):
    def setUp(self):
        root = EXPERIMENTS / "test-artifacts"
        root.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=root)
        self.directory = Path(self.tmp.name)
        self.config = self.directory / "config.json"
        self.document = {"model": {"vocab_size": 32, "width": 16, "heads": 2, "layers": 1,
                                   "context": 8, "ff_width": 32},
                         "training": {"steps": 4, "batch_size": 2, "grad_accum": 2, "lr": 0.01,
                                      "warmup_steps": 1, "dtype": "float32", "validation_every": 2,
                                      "validation_batches": 2, "loss_chunk_size": 3, "log_every": 100}}
        self.config.write_text(json.dumps(self.document))
        self.manifest = self.directory / "manifest.json"
        self.data = {"format_version": 1, "dtype": "<u4",
                     "tokenizer": {"model": "fixture", "revision": "fixed", "vocab_size": 32},
                     "train": [self.shard("train-a.bin", range(65))],
                     "val": [self.shard("val-a.bin", range(65))]}
        self.publish()

    def tearDown(self):
        self.tmp.cleanup()

    def shard(self, name, values):
        values = array("I", (value % 32 for value in values))
        path = self.directory / name
        path.write_bytes(values.tobytes())
        return {"path": name, "tokens": len(values), "sha256": sha256(path)}

    def publish(self):
        self.manifest.write_text(json.dumps(self.data))

    def saved(self, name):
        output = self.directory / name
        return load_checkpoint(resolve_checkpoint(output, "latest"))

    def assert_state_equal(self, first, second):
        for name in first["model"]:
            self.assertTrue(torch.equal(first["model"][name], second["model"][name]), name)
        for parameter_id, values in first["optimizer"]["state"].items():
            for name, value in values.items():
                other = second["optimizer"]["state"][parameter_id][name]
                self.assertTrue(torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other)
        for key in ("global_step", "tokens", "cursor", "scheduler", "best_validation"):
            self.assertEqual(first[key], second[key])
        self.assertTrue(torch.equal(first["rng"]["cpu"], second["rng"]["cpu"]))

    def test_continuous_matches_interrupted_resume(self):
        self.data["train"].append(self.shard("train-b.bin", range(65, 129))); self.publish()
        full = train(self.config, self.manifest, self.directory / "full", "cpu", save_every="1steps")
        partial = train(self.config, self.manifest, self.directory / "resumed", "cpu",
                        max_steps=2, save_every="1steps")
        self.assertEqual(partial["status"], "paused_limit")
        resumed = train(self.config, self.manifest, self.directory / "resumed", "cpu",
                        resume="latest", save_every="1steps")
        self.assertEqual(full["status"], "complete")
        self.assertEqual(resumed["status"], "complete")
        self.assert_state_equal(self.saved("full"), self.saved("resumed"))
        output = self.directory / "resumed"
        self.assertLessEqual(len(list((output / "checkpoints").glob("step-*"))), 4)
        self.assertTrue((output / "best.json").exists())
        state = self.saved("resumed")
        self.assertEqual(state["model"]["lm_head.weight"].untyped_storage().data_ptr(),
                         state["model"]["token_embedding.weight"].untyped_storage().data_ptr())
        self.assertTrue(all(value.device.type == "cpu" for value in state["model"].values()))
        for name, digest in state["source_sha256"].items():
            self.assertEqual(digest, sha256(Path(__file__).with_name(name)), name)

    def test_shard_exhaustion_pauses_and_append_resumes_exactly(self):
        paused = train(self.config, self.manifest, self.directory / "growing", "cpu", save_every="1steps")
        self.assertEqual(paused["status"], "paused_data")
        self.assertEqual(paused["global_step"], 2)
        previous = self.saved("growing")
        self.data["train"].append(self.shard("train-b.bin", range(65, 129))); self.publish()
        resumed = train(self.config, self.manifest, self.directory / "growing", "cpu", resume="latest")
        self.assertEqual(resumed["status"], "complete")
        reference = train(self.config, self.manifest, self.directory / "reference", "cpu")
        self.assertEqual(reference["status"], "complete")
        self.assert_state_equal(self.saved("growing"), self.saved("reference"))
        self.assertTrue(compatible_data(previous["data_identity"], self.saved("growing")["data_identity"]))

    def test_stop_request_checkpoints_at_optimizer_boundary(self):
        calls = [0]

        def stop():
            calls[0] += 1
            return calls[0] > 1

        result = train(self.config, self.manifest, self.directory / "stopped", "cpu", stop_requested=stop)
        self.assertEqual(result["status"], "paused_signal")
        self.assertEqual(result["global_step"], 1)
        state = self.saved("stopped")
        self.assertEqual(state["tokens"], 32)
        self.assertEqual(state["cursor"]["train"], 32)
        self.assertEqual(state["scheduler"]["completed_steps"], 1)

    def test_token_checks_and_no_wrap(self):
        stream = TokenStream(self.data["train"], self.directory, "<u4", 32)
        try:
            self.assertTrue(torch.equal(stream.read(60, 5), torch.tensor([28, 29, 30, 31, 0])))
            with self.assertRaisesRegex(ValueError, "wrapping"):
                stream.read(64, 2)
        finally:
            stream.close()
        bad = copy.deepcopy(self.data["train"])
        bad[0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "checksum"):
            TokenStream(bad, self.directory, "<u4", 32)
        path = self.directory / "invalid.bin"
        path.write_bytes(array("I", [32]).tobytes())
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            TokenStream([{"path": path.name, "tokens": 1, "sha256": sha256(path)}], self.directory, "<u4", 32)

    def test_uint16_files_use_the_same_packed_contract(self):
        path = self.directory / "small.u16"
        path.write_bytes(array("H", [1, 2, 3, 4, 5]).tobytes())
        record = {"path": path.name, "tokens": 5, "sha256": sha256(path)}
        stream = TokenStream([record], self.directory, "<u2", 32)
        try:
            inputs, targets, cursor = stream.batch(0, 1, 4, torch.device("cpu"))
            self.assertTrue(torch.equal(inputs, torch.tensor([[1, 2, 3, 4]])))
            self.assertTrue(torch.equal(targets, torch.tensor([[2, 3, 4, 5]])))
            self.assertEqual(cursor, 4)
        finally:
            stream.close()

    def test_nonfinite_gradient_refuses_the_optimizer_update(self):
        self.document["training"]["lr"] = 1e20
        self.document["training"]["weight_decay"] = 0.0
        self.config.write_text(json.dumps(self.document))
        output = self.directory / "nonfinite"
        with self.assertRaises((FloatingPointError, RuntimeError)):
            train(self.config, self.manifest, output, "cpu", save_every="1steps")
        records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual(records[-1]["event"], "failure")
        self.assertIn("non-finite", records[-1]["error"])
        self.assertEqual(records[-1]["global_step"], 1)
        self.assertEqual(load_checkpoint(resolve_checkpoint(output, "latest"))["global_step"], 1)

    def test_modified_prefix_and_config_refused_on_resume(self):
        train(self.config, self.manifest, self.directory / "strict", "cpu", max_steps=1)
        changed = copy.deepcopy(self.data)
        changed["tokenizer"]["revision"] = "changed"
        self.manifest.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "differs"):
            train(self.config, self.manifest, self.directory / "strict", "cpu", resume="latest")
        self.publish()
        self.document["training"]["lr"] = 0.02
        self.config.write_text(json.dumps(self.document))
        with self.assertRaisesRegex(ValueError, "differs"):
            train(self.config, self.manifest, self.directory / "strict", "cpu", resume="latest")

    def test_checkpoint_keep_one_preserves_latest_and_best(self):
        self.assertEqual(TrainingConfig(steps=4, warmup_steps=1).checkpoint_keep, 3)
        with self.assertRaisesRegex(ValueError, "checkpoint_keep"):
            TrainingConfig(steps=4, warmup_steps=1, checkpoint_keep=0)
        output = self.directory / "retention"
        state = {"global_step": 1, "tokens": 32, "status": "running",
                 "config_sha256": "c", "data_sha256": "d", "model": {"weight": torch.ones(2)}}
        best = save_checkpoint(output, state, best=True, keep=1)
        state["global_step"] = 2
        middle = save_checkpoint(output, state, keep=1)
        state["global_step"] = 3
        latest = save_checkpoint(output, state, keep=1)
        self.assertTrue(best.exists())
        self.assertTrue(latest.exists())
        self.assertFalse(middle.exists())
        self.assertEqual(set((output / "checkpoints").glob("step-*")), {best, latest})
        self.assertEqual(resolve_checkpoint(output, "latest"), latest)
        self.assertEqual(load_checkpoint(best)["global_step"], 1)
        self.document["training"]["checkpoint_keep"] = 1
        self.config.write_text(json.dumps(self.document))
        result = train(self.config, self.manifest, self.directory / "small-retention", "cpu", save_every="1steps")
        self.assertEqual(result["status"], "paused_data")
        self.assertLessEqual(len(list((self.directory / "small-retention" / "checkpoints").glob("step-*"))), 2)
        self.assertEqual(self.saved("small-retention")["resolved_config"]["training"]["checkpoint_keep"], 1)

    def test_schedule_and_interval_are_explicit(self):
        config = TrainingConfig(steps=4, warmup_steps=1, lr=0.01, dtype="float32")
        self.assertEqual(learning_rate(0, config), 0.01)
        self.assertAlmostEqual(learning_rate(3, config), 0.001)
        self.assertEqual(save_interval("30s"), (30.0, "seconds"))
        self.assertEqual(save_interval("2steps"), (2.0, "steps"))
        with self.assertRaises(ValueError):
            save_interval("1.5steps")


if __name__ == "__main__":
    unittest.main()
