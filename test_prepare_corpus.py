"""Offline checks for incremental corpus preparation."""

import json
import os
from pathlib import Path
import struct
import tempfile
import unittest

from prepare_corpus import (atomic_json, commit_shard, new_manifest,
                            prepare_text_shard, split_for_text, verify_manifest,
                            _source_files)


class FakeTokenizer:
    eos_token_id = 2

    def convert_tokens_to_ids(self, token):
        assert token == "<|endoftext|>"
        return 0

    def __call__(self, texts, **options):
        assert options["add_special_tokens"] is False
        return {"input_ids": [[ord(char) for char in text] for text in texts]}


def document_for(split, *, exclude=()):
    for index in range(100):
        text = f"document {index}"
        if text not in exclude and split_for_text(text, 2, 0)[0] == split:
            return text
    raise AssertionError(f"No synthetic {split} document")


def manifest():
    return new_manifest("a" * 40, "b" * 40, "c" * 64, 49152, 0, 1000, 2, 0)


class PrepareCorpusTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("PICO_HF_SMOKE") == "1", "requires Hugging Face API access")
    def test_pinned_hub_tree(self):
        from huggingface_hub import HfApi

        sources = _source_files(HfApi(), "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9")
        self.assertEqual(len(sources), 14)
        self.assertEqual(sources[0]["path"], "sample/10BT/000_00000.parquet")
        self.assertTrue(all(source["size"] > 0 for source in sources))

    def test_split_is_stable_and_pinned(self):
        self.assertEqual(split_for_text("same document", 100, 0),
                         split_for_text("same document", 100, 0))
        with self.assertRaises(ValueError):
            split_for_text("text", 1, 0)
        with self.assertRaisesRegex(ValueError, "commit hashes"):
            new_manifest("main", "b" * 40, "c" * 64, 49152, 0, 1000, 2, 0)
        self.assertEqual(manifest()["dtype"], "<u2")
        with self.assertRaisesRegex(ValueError, "fit the dtype"):
            new_manifest("a" * 40, "b" * 40, "c" * 64, 65537, 0, 1000, 2, 0)

    def test_atomic_publish_resume_prefix_and_validation_deduplication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = manifest()
            atomic_json(root / "manifest.json", initial)
            train_a = document_for("train")
            train_b = document_for("train", exclude=(train_a,))
            val = document_for("val")
            first, hashes = prepare_text_shard(
                [train_a, val, val], FakeTokenizer(), root, 0, set(), 1000,
                batch_size=2, val_modulus=2)
            self.assertEqual(json.loads((root / "manifest.json").read_text()), initial)
            self.assertEqual(first["train"]["tokens"], len(train_a) + 1)
            self.assertEqual(first["val"]["tokens"], len(val) + 1)
            self.assertEqual(first["hashes"]["documents"], 1)
            self.assertTrue(first["train"]["path"].endswith(".u16"))
            self.assertEqual(struct.unpack(f"<{len(train_a) + 1}H",
                                           (root / first["train"]["path"]).read_bytes()),
                             tuple(ord(char) for char in train_a) + (0,))

            source_a = {"path": "sample/10BT/000.parquet", "sha256": "1" * 64}
            committed = commit_shard(root, initial, source_a, first)
            self.assertEqual(verify_manifest(root, committed), hashes)
            prior_train = committed["train"][:]
            prior_val = committed["val"][:]

            second, second_hashes = prepare_text_shard(
                [val, train_b], FakeTokenizer(), root, 1, hashes, 1000,
                batch_size=2, val_modulus=2)
            self.assertIsNone(second["val"])
            self.assertFalse(second_hashes)
            committed = commit_shard(root, committed,
                                     {"path": "sample/10BT/001.parquet", "sha256": "2" * 64},
                                     second)
            self.assertEqual(committed["train"][:1], prior_train)
            self.assertEqual(committed["val"], prior_val)
            self.assertEqual(verify_manifest(root, committed), hashes)
            wrong_dtype = json.loads(json.dumps(committed))
            wrong_dtype["dtype"] = "<u4"
            with self.assertRaisesRegex(ValueError, "Corrupt prepared shard"):
                verify_manifest(root, wrong_dtype)
            with self.assertRaisesRegex(ValueError, "already committed"):
                commit_shard(root, committed, source_a, first)

            (root / first["train"]["path"]).write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "Corrupt prepared shard"):
                verify_manifest(root, committed)

    def test_target_stops_at_document_boundary_without_repeating_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_a = document_for("train")
            train_b = document_for("train", exclude=(train_a,))
            entries, _ = prepare_text_shard(
                [train_a, train_b], FakeTokenizer(), root, 0, set(), len(train_a) + 1,
                val_modulus=2)
            self.assertTrue(entries["truncated"])
            self.assertEqual(entries["rows"], 1)
            self.assertEqual(entries["train"]["tokens"], len(train_a) + 1)

    def test_uint32_is_explicit_and_uint16_rejects_large_ids(self):
        train = document_for("train")

        class LargeIdTokenizer(FakeTokenizer):
            def __call__(self, texts, **options):
                assert options["add_special_tokens"] is False
                return {"input_ids": [[70_000] for _ in texts]}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "outside u16"):
                prepare_text_shard([train], LargeIdTokenizer(), root, 0, set(), 1000,
                                   val_modulus=2)
            self.assertFalse(list((root / "train").glob("*.u16")))
            wide = new_manifest("a" * 40, "b" * 40, "c" * 64, 70001, 0,
                                1000, 2, 0, storage_dtype="u32")
            entries, _ = prepare_text_shard([train], LargeIdTokenizer(), root, 0, set(),
                                            1000, val_modulus=2, storage_dtype="u32")
            self.assertTrue(entries["train"]["path"].endswith(".u32"))
            self.assertEqual(struct.unpack("<2I", (root / entries["train"]["path"]).read_bytes()),
                             (70_000, 0))
            committed = commit_shard(root, wide,
                                     {"path": "sample/10BT/000.parquet", "sha256": "1" * 64},
                                     entries)
            self.assertEqual(committed["dtype"], "<u4")
            self.assertEqual(verify_manifest(root, committed), set())


if __name__ == "__main__":
    unittest.main()
