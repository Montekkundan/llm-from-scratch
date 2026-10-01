"""Prepare pinned FineWeb-Edu Parquet into resumable token shards.

Requires optional runtime packages: huggingface_hub, pyarrow, transformers.
No model weights are downloaded. A manifest entry appears only after its token
files are complete, hashed, and atomically renamed.
"""

from __future__ import annotations

import argparse
from array import array
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Iterable


DATASET = "HuggingFaceFW/fineweb-edu"
CONFIG = "sample-10BT"
TOKENIZER = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
DOCUMENT_SEPARATOR = "<|endoftext|>"
REVISION = re.compile(r"[0-9a-f]{40}\Z")
MAX_UINT32 = (1 << 32) - 1
DTYPES = {"u16": ("<u2", "H", 2, (1 << 16) - 1),
          "u32": ("<u4", "I", 4, MAX_UINT32)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_fingerprint(directory: Path) -> str:
    files = sorted(path for path in directory.iterdir() if path.is_file())
    if not files:
        raise ValueError("No saved tokenizer files")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def document_separator_id(tokenizer) -> int:
    separator = tokenizer.convert_tokens_to_ids(DOCUMENT_SEPARATOR)
    if type(separator) is not int or separator != 0:
        raise ValueError("Pinned tokenizer must map <|endoftext|> to ID 0")
    return separator


def split_for_text(text: str, val_modulus: int, val_bucket: int) -> tuple[str, str]:
    if val_modulus < 2 or not 0 <= val_bucket < val_modulus:
        raise ValueError("Require val_modulus >= 2 and 0 <= val_bucket < val_modulus")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    bucket = int(digest[:16], 16) % val_modulus
    return ("val" if bucket == val_bucket else "train"), digest


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def new_manifest(dataset_revision: str, tokenizer_revision: str,
                 tokenizer_sha: str, vocab_size: int, eos_id: int,
                 target_train_tokens: int, val_modulus: int, val_bucket: int,
                 *, storage_dtype: str = "u16") -> dict:
    if not REVISION.fullmatch(dataset_revision) or not REVISION.fullmatch(tokenizer_revision):
        raise ValueError("Dataset and tokenizer revisions must be 40-character commit hashes")
    if storage_dtype not in DTYPES:
        raise ValueError("Token dtype must be u16 or u32")
    dtype, _, _, maximum = DTYPES[storage_dtype]
    if target_train_tokens <= 0 or not 0 <= eos_id <= maximum or not 0 < vocab_size <= maximum + 1:
        raise ValueError("Target tokens, tokenizer vocabulary, and separator ID must fit the dtype")
    split_for_text("", val_modulus, val_bucket)
    return {
        "format_version": 1,
        "dtype": dtype,
        "dataset": {"repo": DATASET, "revision": dataset_revision,
                    "config": CONFIG, "shards": []},
        "tokenizer": {"model": TOKENIZER, "revision": tokenizer_revision,
                      "config_sha256": tokenizer_sha, "vocab_size": vocab_size,
                      "eos_id": eos_id, "document_separator": DOCUMENT_SEPARATOR,
                      "add_bos": False},
        "split": {"method": "sha256-text-utf8", "val_modulus": val_modulus,
                  "val_bucket": val_bucket, "unique_validation": True},
        "target_train_tokens": target_train_tokens,
        "prepared_tokens": {"train": 0, "val": 0},
        "train": [], "val": [], "val_document_hashes": [],
    }


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _checked_path(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid manifest path: {name}")
    return root / relative


def verify_manifest(root: Path, manifest: dict) -> set[str]:
    """Verify committed files and rebuild the unique validation-hash set."""
    widths = {spec[0]: spec[2] for spec in DTYPES.values()}
    if manifest.get("format_version") != 1 or manifest.get("dtype") not in widths:
        raise ValueError("Unsupported token manifest")
    width = widths[manifest["dtype"]]
    for split in ("train", "val"):
        total = 0
        for entry in manifest[split]:
            path = _checked_path(root, entry["path"])
            if path.stat().st_size != entry["tokens"] * width or sha256_file(path) != entry["sha256"]:
                raise ValueError(f"Corrupt prepared shard: {path}")
            total += entry["tokens"]
        if total != manifest["prepared_tokens"][split]:
            raise ValueError(f"Manifest {split} token count differs from files")
    seen: set[str] = set()
    for entry in manifest["val_document_hashes"]:
        path = _checked_path(root, entry["path"])
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Corrupt validation hash shard: {path}")
        values = path.read_text(encoding="ascii").splitlines()
        if len(values) != entry["documents"] or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in values):
            raise ValueError(f"Invalid validation hash shard: {path}")
        if len(seen.union(values)) != len(seen) + len(values):
            raise ValueError(f"Duplicate validation document hash: {path}")
        seen.update(values)
    return seen


def _token_bytes(ids: list[int], eos_id: int, storage_dtype: str) -> bytes:
    _, typecode, width, maximum = DTYPES[storage_dtype]
    if any(type(token) is not int or not 0 <= token <= maximum for token in (*ids, eos_id)):
        raise ValueError(f"Tokenizer returned an ID outside {storage_dtype}")
    values = array(typecode, (*ids, eos_id))
    if values.itemsize != width:
        raise RuntimeError(f"This platform does not use {width * 8}-bit unsigned integers")
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


def prepare_text_shard(records: Iterable[str], tokenizer, root: Path, index: int,
                       seen_val_hashes: set[str], remaining_train_tokens: int,
                       *, batch_size: int = 64, val_modulus: int = 100,
                       val_bucket: int = 0, storage_dtype: str = "u16") -> tuple[dict, set[str]]:
    """Tokenize one source shard; return ready entries, but do not edit manifest."""
    if batch_size <= 0 or remaining_train_tokens <= 0 or index < 0:
        raise ValueError("Invalid batch size, source index, or remaining token target")
    if storage_dtype not in DTYPES:
        raise ValueError("Token dtype must be u16 or u32")
    eos_id = document_separator_id(tokenizer)
    names = {"train": root / "train" / f"{index:05d}.{storage_dtype}",
             "val": root / "val" / f"{index:05d}.{storage_dtype}",
             "hashes": root / "val-hashes" / f"{index:05d}.txt"}
    for path in names.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary = {key: path.with_name(f".{path.name}.{os.getpid()}.tmp")
                 for key, path in names.items()}
    counts = {"train": 0, "val": 0, "val_documents": 0, "rows": 0}
    digests = {key: hashlib.sha256() for key in names}
    new_val_hashes: set[str] = set()
    truncated = False
    try:
        with temporary["train"].open("wb") as train, temporary["val"].open("wb") as val, \
                temporary["hashes"].open("wb") as hashes:
            handles = {"train": train, "val": val, "hashes": hashes}
            source = iter(records)
            while True:
                batch = list(itertools.islice(source, batch_size))
                if not batch:
                    break
                valid = [text for text in batch if isinstance(text, str) and text.strip()]
                encoded = tokenizer(valid, add_special_tokens=False,
                                    return_attention_mask=False,
                                    return_token_type_ids=False)["input_ids"] if valid else []
                if len(encoded) != len(valid):
                    raise ValueError("Tokenizer returned the wrong number of documents")
                encoded_iter = iter(encoded)
                for text in batch:
                    counts["rows"] += 1
                    if text is None or text == "":
                        continue
                    if not isinstance(text, str):
                        raise ValueError("Parquet text column must contain strings or nulls")
                    if not text.strip():
                        continue
                    ids = next(encoded_iter)
                    split, digest = split_for_text(text, val_modulus, val_bucket)
                    if split == "val" and (digest in seen_val_hashes or digest in new_val_hashes):
                        continue
                    payload = _token_bytes(ids, eos_id, storage_dtype)
                    handles[split].write(payload)
                    digests[split].update(payload)
                    counts[split] += len(ids) + 1
                    if split == "val":
                        new_val_hashes.add(digest)
                        line = (digest + "\n").encode("ascii")
                        hashes.write(line)
                        digests["hashes"].update(line)
                        counts["val_documents"] += 1
                    if counts["train"] >= remaining_train_tokens:
                        truncated = True
                        break
                if truncated:
                    break
            for handle in handles.values():
                handle.flush()
                os.fsync(handle.fileno())
        if counts["train"] == 0:
            raise ValueError("Source shard produced no training tokens")
        for key in ("train", "val", "hashes"):
            if (key == "val" and counts["val"] == 0) or (key == "hashes" and counts["val_documents"] == 0):
                temporary[key].unlink()
            else:
                os.replace(temporary[key], names[key])
        entries = {
            "train": {"path": _relative(root, names["train"]), "tokens": counts["train"],
                      "sha256": digests["train"].hexdigest()},
            "val": ({"path": _relative(root, names["val"]), "tokens": counts["val"],
                     "sha256": digests["val"].hexdigest()} if counts["val"] else None),
            "hashes": ({"path": _relative(root, names["hashes"]),
                        "documents": counts["val_documents"],
                        "sha256": digests["hashes"].hexdigest()} if counts["val_documents"] else None),
            "rows": counts["rows"], "truncated": truncated,
        }
        return entries, new_val_hashes
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


def commit_shard(root: Path, manifest: dict, source: dict, entries: dict) -> dict:
    """Publish one completed source shard, leaving prior entries unchanged."""
    if source["path"] in {item["path"] for item in manifest["dataset"]["shards"]}:
        raise ValueError("Source shard was already committed")
    updated = json.loads(json.dumps(manifest))
    updated["train"].append(entries["train"])
    updated["prepared_tokens"]["train"] += entries["train"]["tokens"]
    if entries["val"] is not None:
        updated["val"].append(entries["val"])
        updated["prepared_tokens"]["val"] += entries["val"]["tokens"]
        updated["val_document_hashes"].append(entries["hashes"])
    updated["dataset"]["shards"].append({**source, "rows": entries["rows"],
                                           "truncated": entries["truncated"]})
    atomic_json(root / "manifest.json", updated)
    return updated


def _progress(root: Path, event: str, **details) -> None:
    with (root / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": event, **details}, sort_keys=True) + "\n")
        handle.flush()


def _source_files(api, revision: str) -> list[dict]:
    from huggingface_hub.hf_api import RepoFile

    files = []
    for item in api.list_repo_tree(DATASET, path_in_repo="sample/10BT", recursive=False,
                                   expand=True, revision=revision, repo_type="dataset"):
        if not isinstance(item, RepoFile) or not item.path.endswith(".parquet"):
            continue
        lfs = item.lfs
        expected = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        files.append({"path": item.path, "size": item.size, "expected_sha256": expected})
    files.sort(key=lambda item: item["path"])
    if not files:
        raise ValueError("No Parquet shards found at pinned dataset revision")
    return files


def _texts_from_parquet(path: Path) -> Iterable[str]:
    import pyarrow.parquet as parquet

    source = parquet.ParquetFile(path)
    if "text" not in source.schema.names:
        raise ValueError(f"No text column in {path}")
    for batch in source.iter_batches(batch_size=512, columns=["text"]):
        yield from batch.column(0).to_pylist()


def run(args: argparse.Namespace) -> dict:
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer

    if not REVISION.fullmatch(args.dataset_revision) or not REVISION.fullmatch(args.tokenizer_revision):
        raise ValueError("Pass exact 40-character dataset and tokenizer commit revisions")
    if args.max_shards is not None and args.max_shards < 1:
        raise ValueError("--max-shards must be positive")
    root = args.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    saved = root / "tokenizer"
    if (root / "manifest.json").exists():
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        tokenizer = AutoTokenizer.from_pretrained(str(saved), use_fast=True,
                                                   local_files_only=True, trust_remote_code=False)
        separator = document_separator_id(tokenizer)
        expected = new_manifest(args.dataset_revision, args.tokenizer_revision,
                                tokenizer_fingerprint(saved), len(tokenizer),
                                separator, args.max_train_tokens,
                                args.val_modulus, args.val_bucket,
                                storage_dtype=args.dtype)
        for key in ("format_version", "dtype", "tokenizer", "split", "target_train_tokens"):
            if manifest[key] != expected[key]:
                raise ValueError(f"Resume settings differ: {key}")
        for key in ("repo", "revision", "config"):
            if manifest["dataset"][key] != expected["dataset"][key]:
                raise ValueError(f"Resume settings differ: dataset.{key}")
        seen_val_hashes = verify_manifest(root, manifest)
    else:
        if any(any(any((root / split).glob(pattern)) for pattern in ("*.u16", "*.u32"))
               for split in ("train", "val")
               if (root / split).exists()):
            raise ValueError("Output has token files but no manifest")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, revision=args.tokenizer_revision,
                                                   use_fast=True, trust_remote_code=False)
        if not tokenizer.is_fast:
            raise ValueError("Pinned tokenizer must be fast")
        separator = document_separator_id(tokenizer)
        saved.mkdir(exist_ok=False)
        tokenizer.save_pretrained(saved)
        manifest = new_manifest(args.dataset_revision, args.tokenizer_revision,
                                tokenizer_fingerprint(saved), len(tokenizer),
                                separator, args.max_train_tokens,
                                args.val_modulus, args.val_bucket,
                                storage_dtype=args.dtype)
        atomic_json(root / "manifest.json", manifest)
        seen_val_hashes = set()
    if manifest["prepared_tokens"]["train"] >= args.max_train_tokens:
        return manifest
    sources = _source_files(HfApi(), args.dataset_revision)
    processed = manifest["dataset"]["shards"]
    if len(processed) > len(sources) or any(item["path"] != sources[index]["path"]
                                                for index, item in enumerate(processed)):
        raise ValueError("Committed source shards are not a prefix of pinned source order")
    for index, source in enumerate(sources[len(processed):], start=len(processed)):
        if args.max_shards is not None and index >= args.max_shards:
            break
        local = Path(hf_hub_download(DATASET, source["path"], repo_type="dataset",
                                     revision=args.dataset_revision, local_dir=root / "raw"))
        actual_sha = sha256_file(local)
        if source["expected_sha256"] is not None and actual_sha != source["expected_sha256"]:
            raise ValueError(f"Downloaded source hash mismatch: {source['path']}")
        if local.stat().st_size != source["size"]:
            raise ValueError(f"Downloaded source size mismatch: {source['path']}")
        _progress(root, "downloaded", source=source["path"], sha256=actual_sha)
        remaining = args.max_train_tokens - manifest["prepared_tokens"]["train"]
        entries, new_hashes = prepare_text_shard(
            _texts_from_parquet(local), tokenizer, root, index, seen_val_hashes, remaining,
            batch_size=args.batch_size, val_modulus=args.val_modulus,
            val_bucket=args.val_bucket, storage_dtype=args.dtype)
        source_record = {"path": source["path"],
                         "url": f"https://huggingface.co/datasets/{DATASET}/resolve/{args.dataset_revision}/{source['path']}",
                         "sha256": actual_sha, "size": source["size"]}
        manifest = commit_shard(root, manifest, source_record, entries)
        seen_val_hashes.update(new_hashes)
        _progress(root, "ready", source=source["path"],
                  train_tokens=manifest["prepared_tokens"]["train"],
                  val_tokens=manifest["prepared_tokens"]["val"])
        if not args.keep_raw:
            local.unlink()
        if manifest["prepared_tokens"]["train"] >= args.max_train_tokens:
            break
    if manifest["prepared_tokens"]["train"] < args.max_train_tokens:
        _progress(root, "incomplete", train_tokens=manifest["prepared_tokens"]["train"],
                  target_train_tokens=args.max_train_tokens)
        if args.max_shards is None or len(manifest["dataset"]["shards"]) == len(sources):
            raise RuntimeError("Pinned source exhausted before the requested training-token target")
    else:
        _progress(root, "target_reached", train_tokens=manifest["prepared_tokens"]["train"])
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset-revision", required=True,
                        help="Pinned FineWeb-Edu dataset commit SHA")
    parser.add_argument("--tokenizer-revision", required=True,
                        help="Pinned SmolLM2 tokenizer commit SHA")
    parser.add_argument("--max-train-tokens", type=int, default=8_000_110_593)
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="u16")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-modulus", type=int, default=100)
    parser.add_argument("--val-bucket", type=int, default=0)
    parser.add_argument("--max-shards", type=int, help="Stop after this many source shards, for a bounded run")
    parser.add_argument("--keep-raw", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"manifest": str(args.output / "manifest.json"),
                      "prepared_tokens": result["prepared_tokens"],
                      "target_train_tokens": result["target_train_tokens"]}, indent=2))


if __name__ == "__main__":
    main()
