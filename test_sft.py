"""Conversation validation and real CPU training/export checks."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from torch.nn import functional as F

from chat import serialize
from course_model import ModelConfig, PicoLLM
from evaluate import load_artifact
from generate import generate_ids
from sft import chat_batch, read_conversations, score_task, sha256, task_data, validate_generation_budget
from tokenizer import IGNORE


ROOT = Path(__file__).parent


def fixture():
    return [{"id": split, "split": split,
             "messages": [{"role": "user", "content": prompt},
                          {"role": "assistant", "content": "A token is a unit of text."}]}
            for split, prompt in (("train", "hello"), ("validation", "hello!"), ("test", "hey"))]


class ConversationTests(unittest.TestCase):
    def read(self, rows):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.json"
            path.write_text(json.dumps(rows))
            return read_conversations(path)

    def test_rejects_malformed_data(self):
        invalid = [[], {}, [None], fixture()[:2]]
        changes = [("id", ""), ("split", "dev"), ("messages", [])]
        for key, value in changes:
            rows = fixture(); rows[0][key] = value; invalid.append(rows)
        for message in ({"role": "user", "content": 1},
                        {"role": "user", "content": " "},
                        {"role": "tool", "content": "hello"}):
            rows = fixture(); rows[0]["messages"][0] = message; invalid.append(rows)
        rows = fixture(); rows[0]["messages"][-1]["role"] = "user"; invalid.append(rows)
        rows = fixture(); rows[1]["id"] = rows[0]["id"]; invalid.append(rows)
        for index, rows in enumerate(invalid):
            with self.subTest(case=index), self.assertRaises(ValueError):
                self.read(rows)

    def test_rejects_duplicates_and_split_leakage(self):
        duplicate = fixture()
        duplicate[1]["messages"] = copy.deepcopy(duplicate[0]["messages"])
        leaked_prompt = copy.deepcopy(duplicate)
        leaked_prompt[1]["messages"][-1]["content"] = "Another answer"
        leaked_group = fixture()
        leaked_group[0]["group"] = leaked_group[1]["group"] = "shared-source"
        for rows, reason in ((duplicate, "Duplicate"), (leaked_prompt, "prompt"), (leaked_group, "group")):
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                self.read(rows)

    def test_token_limit_checks_full_prompt_context(self):
        rows = self.read(fixture())
        validate_generation_budget(rows, 128, 64)
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_generation_budget(rows, 128, 0)
        with self.assertRaisesRegex(ValueError, "context"):
            validate_generation_budget(rows, 64, 64)

    def test_real_loss_and_generation_use_full_history(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        model = PicoLLM(ModelConfig())
        row = {"id": "history", "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "token?"},
            {"role": "assistant", "content": "A token is a unit of text."}]}
        x, y = chat_batch([row], 128)
        loss = F.cross_entropy(model(x).reshape(-1, 259), y.reshape(-1), ignore_index=IGNORE)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                            for parameter in model.parameters()))
        report = score_task(model, [row], max_new_tokens=24)
        prefix, _ = serialize(row["messages"][:-1], generation=True)
        self.assertEqual(report["outputs"][0]["expected"], row["messages"][-1]["content"])
        self.assertEqual(report["outputs"][0]["generated"]["usage"]["prompt_tokens"], len(prefix))
        self.assertEqual(report["supervised_targets"], len(b"Hello!\n") + len(b"A token is a unit of text.\n") + 2)


class SFTArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.tmp.name)
        cls.base = cls.directory / "base"
        cls.run_script("train.py", "--output", cls.base, "--steps", "160")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @staticmethod
    def run_script(script, *args):
        result = subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                                cwd=ROOT, check=True, capture_output=True, text=True)
        return json.loads(result.stdout)

    def test_default_echo_behavior_is_preserved(self):
        output = self.directory / "echo"
        summary = self.run_script("sft.py", "--base", self.base, "--output", output, "--steps", "160")
        self.assertEqual(summary["train_exact_match"], [8, 8])
        self.assertEqual(summary["test_exact_match"], [0, 2])
        report = json.loads((output / "sft-report.json").read_text())
        self.assertEqual(report["settings"]["max_new_tokens"], 12)
        self.assertEqual(json.loads((output / "sft-data.json").read_text()), task_data())
        self.assertEqual(self.run_script("sft.py", "--evaluate", output)["reload"], "passed")

    def test_custom_long_answer_artifact_reloads_and_generates(self):
        data = self.directory / "conversations.json"
        data.write_text(json.dumps(fixture()))
        output = self.directory / "custom"
        summary = self.run_script("sft.py", "--base", self.base, "--data", data,
                           "--output", output, "--steps", "240", "--max-new-tokens", "48")
        self.assertEqual(summary["train_exact_match"], [1, 1])
        report = json.loads((output / "sft-report.json").read_text())
        self.assertLess(report["final"]["train"]["assistant_nll"], report["initial"]["train"]["assistant_nll"])
        self.assertEqual(report["lineage"]["base_model_sha256"], sha256(self.base / "model.pt"))
        self.assertEqual(report["data_sha256"], sha256(output / "sft-data.json"))
        self.assertEqual(report["settings"]["max_new_tokens"], 48)
        generated = self.run_script("generate.py", output, "--message", "hello", "--max-new-tokens", "48", "--cached")
        self.assertEqual(generated["text"], "A token is a unit of text.\n")
        self.assertEqual(generated["finish_reason"], "stop")
        self.assertGreater(len(generated["token_ids"]), 12)
        restored = load_artifact(output)
        prefix, _ = serialize(fixture()[0]["messages"][:-1], generation=True)
        self.assertEqual(generate_ids(restored, prefix, 48)["token_ids"], generated["token_ids"])
        self.assertEqual(self.run_script("sft.py", "--evaluate", output)["reload"], "passed")


if __name__ == "__main__":
    unittest.main()
