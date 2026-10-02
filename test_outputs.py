"""Output safety and portability: nothing is overwritten, and nothing assumes this machine."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import checks
import experiments
import generate
import prepare_data
import sft
import train
from train import require_new_path


class ExistingOutputTests(unittest.TestCase):
    def run_main(self, module, *argv):
        with mock.patch.object(sys, "argv", [module.__name__ + ".py", *argv]):
            module.main()

    def test_every_lesson_command_names_the_way_forward_before_doing_any_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            report, run = Path(temporary) / "report.json", Path(temporary) / "run"
            report.write_text("keep me")
            run.mkdir()
            (run / "model.pt").write_text("keep me")
            missing = str(Path(temporary) / "no-such-artifact")
            cases = [
                (checks, ("objective", "--output", str(report)), report),
                (train, ("--output", str(run)), run),
                (generate, (missing, "--prompt", "x", "--output", str(report)), report),
                (experiments, ("cache", "--artifact", missing, "--output", str(report)), report),
                (sft, ("--base", missing, "--output", str(run)), run),
                (prepare_data, ("--train-dir", missing, "--validation-dir", missing, "--output", str(run),
                                "--source", "s", "--license", "l"), run),
            ]
            for module, argv, existing in cases:
                with self.subTest(command=module.__name__):
                    with self.assertRaises(FileExistsError) as caught:
                        self.run_main(module, *argv)
                    message = str(caught.exception)
                    self.assertIn(str(existing), message)
                    self.assertIn("never overwritten", message)
                    self.assertIn("Remove it", message)
                    self.assertIn("--output", message)
            self.assertEqual(report.read_text(), "keep me")
            self.assertEqual((run / "model.pt").read_text(), "keep me")

    def test_a_path_that_does_not_exist_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            require_new_path(Path(temporary) / "fresh.json")
            require_new_path(Path(temporary) / "fresh-dir", option="--out")

    def test_the_message_names_the_removal_command_and_the_option(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, file = Path(temporary) / "d", Path(temporary) / "f.json"
            directory.mkdir()
            file.write_text("{}")
            with self.assertRaisesRegex(FileExistsError, r"rm -r .*/d\) or pass --out with"):
                require_new_path(directory, option="--out")
            with self.assertRaisesRegex(FileExistsError, r"\(rm .*/f\.json\) or pass --output with"):
                require_new_path(file)


class PortabilityTests(unittest.TestCase):
    def test_no_source_assumes_a_machine_path_or_a_checkout_depth(self):
        # A clone anywhere, run from any directory, must behave the same.
        root = Path(__file__).resolve().parent
        for path in sorted(root.glob("*.py")):
            if path.name == Path(__file__).name:
                continue
            text = path.read_text(encoding="utf-8")
            for needle in ("/Volumes/", "/Users/", "/home/", ".parents["):
                self.assertFalse(needle in text, f"{path.name} depends on {needle!r}")


if __name__ == "__main__":
    unittest.main()
