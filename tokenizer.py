"""PicoLLM's fixed byte codec and a separate, optional teaching BPE extension."""
from collections import Counter

BOS, EOS, PAD, IGNORE = 256, 257, 258, -100
TOKENIZER = {"type": "utf8-bytes", "vocab_size": 259,
             "bos_id": BOS, "eos_id": EOS, "pad_id": PAD}


def encode(text, *, bos=True, eos=True):
    return ([BOS] if bos else []) + list(text.encode("utf-8", errors="strict")) + ([EOS] if eos else [])


def decode(ids):
    if len(ids) < 2 or ids[0] != BOS or ids[-1] != EOS:
        raise ValueError("Expected BOS, content bytes, EOS")
    if any(type(i) is not int or not 0 <= i < 256 for i in ids[1:-1]):
        raise ValueError("Only byte IDs belong inside content")
    return bytes(ids[1:-1]).decode("utf-8", errors="strict")


def replace_pair(ids, pair, new_id):
    output, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            output.append(new_id)
            i += 2
        else:
            output.append(ids[i])
            i += 1
    return output


def train_bpe(texts, num_merges):
    sequences = [list(text.encode("utf-8")) for text in texts]
    merges = []
    for _ in range(num_merges):
        counts = Counter(pair for ids in sequences for pair in zip(ids, ids[1:]))
        if not counts:
            break
        pair = min(counts, key=lambda p: (-counts[p], p))
        if counts[pair] < 2:
            break
        new_id = 259 + len(merges)
        merges.append((*pair, new_id))
        sequences = [replace_pair(ids, pair, new_id) for ids in sequences]
    return merges


def encode_bpe(text, merges):
    ids = list(text.encode("utf-8"))
    ranked = {(a, b): (rank, new) for rank, (a, b, new) in enumerate(merges)}
    while len(ids) > 1:
        candidates = [p for p in zip(ids, ids[1:]) if p in ranked]
        if not candidates:
            break
        pair = min(candidates, key=lambda p: ranked[p][0])
        ids = replace_pair(ids, pair, ranked[pair][1])
    return ids


def decode_bpe(ids, merges):
    vocabulary = {i: bytes([i]) for i in range(256)}
    for a, b, new in merges:
        vocabulary[new] = vocabulary[a] + vocabulary[b]
    return b"".join(vocabulary[i] for i in ids).decode("utf-8")
