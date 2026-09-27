"""Small integration checks for the real-text holdout protocol."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from evaluate import unigram_baseline
from train import read_documents


ROOT = Path(__file__).parent


def write_split(path, split, text, group):
    path.write_text(json.dumps({"id": split, "group": group, "text": text}) + "\n")


class HeldoutTests(unittest.TestCase):
    def test_converter_writes_declared_test_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for split, text in (("train", "alpha"), ("validation", "beta"), ("test", "gamma")):
                directory = tmp / split / split
                directory.mkdir(parents=True)
                (directory / "sample.txt").write_text(text)
            output = tmp / "data"
            subprocess.run([sys.executable, str(ROOT / "prepare_data.py"),
                            "--train-dir", str(tmp / "train"),
                            "--validation-dir", str(tmp / "validation"),
                            "--test-dir", str(tmp / "test"), "--output", str(output),
                            "--source", "owned fixture", "--license", "original"],
                           check=True, cwd=ROOT, capture_output=True, text=True)
            rows = read_documents(*(output / f"{split}.jsonl"
                                    for split in ("train", "validation", "test")))
            self.assertEqual([row["group"] for row in rows], ["train", "validation", "test"])

    def test_three_way_groups_and_train_only_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = [Path(tmp) / f"{split}.jsonl" for split in ("train", "validation", "test")]
            for path, split, text in zip(files, ("train", "validation", "test"), ("aaaa", "bbbb", "bbbc")):
                write_split(path, split, text, split)
            rows = read_documents(*files)
            self.assertEqual([row["split"] for row in rows], ["train", "validation", "test"])
            train_only = unigram_baseline(["aaaa"], ["bbbb"])
            self.assertEqual(train_only, unigram_baseline(["aaaa"], ["bbbb"]))
            self.assertGreater(train_only["bits_per_byte"], unigram_baseline(["bbbb"], ["bbbb"])["bits_per_byte"])
            write_split(files[2], "test", "bbbc", "validation")
            with self.assertRaisesRegex(ValueError, "group"):
                read_documents(*files)

    def test_training_never_scores_test_and_evaluator_does(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            files = [tmp / f"{split}.jsonl" for split in ("train", "validation", "test")]
            for path, split, text in zip(files, ("train", "validation", "test"),
                                         ("the red fox\n", "the blue owl\n", "a green cat\n")):
                write_split(path, split, text, split)
            output = tmp / "run"
            subprocess.run([sys.executable, str(ROOT / "train.py"), "--output", str(output),
                            "--train-file", str(files[0]), "--validation-file", str(files[1]),
                            "--test-file", str(files[2]), "--steps", "1", "--width", "16",
                            "--heads", "2", "--layers", "1", "--context", "8", "--ff-width", "32"],
                           check=True, cwd=ROOT, capture_output=True, text=True)
            report = json.loads((output / "run-report.json").read_text())
            self.assertEqual(report["test_documents"], 1)
            self.assertNotIn("test", report["initial"])
            self.assertNotIn("test", report["final"])
            result = subprocess.run([sys.executable, str(ROOT / "evaluate.py"), str(output),
                                     "--split", "test"], check=True, cwd=ROOT,
                                    capture_output=True, text=True)
            scored = json.loads(result.stdout)
            self.assertEqual(scored["split"], "test")
            self.assertGreater(scored["targets"], 0)
            self.assertGreater(scored["unigram_baseline"]["bits_per_byte"], 0)


if __name__ == "__main__":
    unittest.main()
