#!/usr/bin/env python3
"""Build the cookbook's data from the public upstream datasets.

    data/gsm8k/train.jsonl       512 rows   openai/gsm8k            (notebook 03)
    data/gsm8k/test.jsonl        256 rows   the held-out test split
    data/banking77/*.jsonl       1001 / 231 / 231, legacy-datasets/banking77  (notebook 02)

Every repo is pinned by commit, and every output is checked against a recorded md5, so
"the same data" is a verified claim rather than a filename convention. A pin moving or a
transform changing fails loudly here instead of silently changing a score. Files you put in
a data directory yourself are used as they are and never overwritten.

    python prepare_data.py              # notebook 03's GSM8K data
    python prepare_data.py --banking77  # notebook 02's Banking77 splits
    python prepare_data.py --check      # verify only, build nothing
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
from pathlib import Path


BANKING = Path(__file__).parent / "data" / "banking77"
BANKING_REPO = "legacy-datasets/banking77"
BANKING_PIN = "f54121560de48f2852f90be299010d1d6dc612ec"
BANKING_SEED = 0
# 77 intents times these counts. Equal per intent, so every label appears in every split and
# the sizes stay 1001 / 231 / 231; a plain random draw would leave some intents unrepresented.
BANKING_PER_INTENT = {"train.jsonl": 13, "validation.jsonl": 3, "test.jsonl": 3}
SYSTEM_PROMPT = ("You are a banking customer service intent classifier. Given a customer "
                 "message, classify it into exactly one of these intents:\n\n{intents}"
                 "\n\nRespond with ONLY the intent name, nothing else.")

BANKING_MD5 = {
    "train.jsonl": "c93851b883aeb5e9f912b40da3f04a75",
    "validation.jsonl": "9fef123cf093c22969f0577e56ef6fa2",
    "test.jsonl": "b06bd32c5d70500f02fe71fb15f6047a",
}


GSM8K = Path(__file__).parent / "data" / "gsm8k"
GSM8K_REPO = "openai/gsm8k"
GSM8K_PIN = "740312add88f781978c0658806c59bc2815b9866"
GSM8K_SEED = 0
GSM8K_TRAIN_ROWS = 512     # 10 steps x 16 problems needs 160; the rest is headroom
GSM8K_EVAL_ROWS = 256
#: GSM8K puts the final answer after a `####` marker at the end of its worked solution.
FINAL_ANSWER = re.compile(r"####\s*(.+)")

GSM8K_MD5 = {
    "train.jsonl": "57f073ff7d52e1980bdae6617494eb7d",
    "test.jsonl": "0c4c50f2da06a98a4aa0ee377e8ce2f6",
}


def gsm8k_row(record):
    """One GSM8K record to our row shape, with the bare number as the label."""
    found = FINAL_ANSWER.search(record["answer"])
    if found is None:
        return None
    return {"input": [{"role": "user", "content": record["question"]}],
            "label": found.group(1).strip().replace(",", ""), "source": "gsm8k"}


def build_gsm8k():
    """Notebook 03's splits: a seeded sample of the train and test splits."""
    from datasets import load_dataset

    GSM8K.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(GSM8K_REPO, "main", revision=GSM8K_PIN)
    rng = random.Random(GSM8K_SEED)
    bad = 0
    for name, split, count in (("train.jsonl", "train", GSM8K_TRAIN_ROWS),
                               ("test.jsonl", "test", GSM8K_EVAL_ROWS)):
        rows = [row for row in (gsm8k_row(r) for r in dataset[split]) if row is not None]
        blob = serialise(rng.sample(rows, count))
        (GSM8K / name).write_bytes(blob)
        bad += not report(name, blob, GSM8K_MD5[name])
    return bad


def serialise(rows):
    return "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")


def report(name, blob, want):
    """Print one file's verdict against its fingerprint, and say whether it passed."""
    digest = hashlib.md5(blob).hexdigest()
    ok = digest == want
    rows = sum(1 for line in blob.decode("utf-8").splitlines() if line.strip())
    print(f"  {'ok      ' if ok else 'MISMATCH'} {name:22s} {rows:5d} rows  {digest}"
          + ("" if ok else f"  expected {want}"))
    return ok


def check_on_disk(directory, fingerprints):
    """Compare what is already on disk against the recorded fingerprints."""
    bad = 0
    for name in fingerprints:
        path = directory / name
        if not path.is_file():
            print(f"  MISSING  {name}")
            bad += 1
            continue
        bad += not report(name, path.read_bytes(), fingerprints[name])
    return bad


def chat_row(text, intent, system_prompt):
    return {"messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": text},
                         {"role": "assistant", "content": intent}]}


def take_per_intent(pool, per_intent, rng):
    """Draw an equal number of (intent, text) pairs per intent, consuming them.

    Consuming is what keeps train and validation disjoint when both come from one split.
    """
    picked = []
    for intent in sorted(pool):
        rng.shuffle(pool[intent])
        picked.extend((intent, text) for text in pool[intent][:per_intent])
        pool[intent] = pool[intent][per_intent:]
    rng.shuffle(picked)
    return picked


def build_banking77():
    """Rebuild notebook 02's Banking77 splits, deterministically.

    The retired builder seeded from `hash(name)`, which Python randomises per process, so it
    could not reproduce its own output. This seeds from a literal.
    """
    from datasets import load_dataset

    BANKING.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(BANKING_REPO, revision=BANKING_PIN)
    intents = list(dataset["train"].features["label"].names)
    system_prompt = SYSTEM_PROMPT.format(intents=", ".join(intents))

    def by_intent(split):
        grouped = {}
        for record in dataset[split]:
            grouped.setdefault(intents[record["label"]], []).append(record["text"])
        for texts in grouped.values():
            texts.sort()                    # sort first, so the seed is the only ordering
        return grouped

    rng = random.Random(BANKING_SEED)
    train_pool, test_pool = by_intent("train"), by_intent("test")
    written = {}
    for name in ("train.jsonl", "validation.jsonl"):
        written[name] = take_per_intent(train_pool, BANKING_PER_INTENT[name], rng)
    written["test.jsonl"] = take_per_intent(test_pool, BANKING_PER_INTENT["test.jsonl"], rng)

    bad = 0
    for name, pairs in written.items():
        rows = [chat_row(text, intent, system_prompt) for intent, text in pairs]
        blob = serialise(rows)
        (BANKING / name).write_bytes(blob)
        bad += not report(name, blob, BANKING_MD5[name])
    (BANKING / "intents.txt").write_text("\n".join(sorted(intents)) + "\n")
    return bad


class StaleData(RuntimeError):
    """What is on disk is not what the recorded numbers were measured on."""


def matches(directory, fingerprints):
    """Whether every file on disk is byte-identical to the example data."""
    return all(hashlib.md5((directory / name).read_bytes()).hexdigest() == want
               for name, want in fingerprints.items())


def ensure(directory, fingerprints, builder):
    """Build the example data if it is absent. Files already there are never overwritten."""
    missing = [name for name in fingerprints if not (directory / name).is_file()]
    if len(missing) == len(fingerprints):
        if builder():
            raise StaleData(
                "Some files do not match the recorded fingerprints. An upstream pin or a "
                "transform has changed, and the notebook's recorded numbers were measured on "
                "the expected data, so do not train on this until it is resolved.")
        print(f"\n{directory.name} is ready")
        return
    if missing:
        raise FileNotFoundError(f"{directory} is missing {', '.join(missing)}")
    if matches(directory, fingerprints):
        print(f"{directory.name} already built and verified")
        return
    print(f"{directory} differs from the example data; using it as-is. "
          "The recorded numbers in the notebook will not apply.")


def ensure_banking77():
    """Notebook 02's data. Call from the notebook; no subprocess, no CLI."""
    ensure(BANKING, BANKING_MD5, build_banking77)


def ensure_gsm8k():
    """Notebook 03's data. Call from the notebook; no subprocess, no CLI."""
    ensure(GSM8K, GSM8K_MD5, build_gsm8k)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="verify what is on disk and build nothing")
    parser.add_argument("--banking77", action="store_true",
                        help="build notebook 02's Banking77 splits")
    arguments = parser.parse_args()

    directory, fingerprints, builder = ((BANKING, BANKING_MD5, ensure_banking77)
                                        if arguments.banking77
                                        else (GSM8K, GSM8K_MD5, ensure_gsm8k))
    if arguments.check:
        raise SystemExit(1 if check_on_disk(directory, fingerprints) else 0)
    try:
        builder()
    except StaleData as stale:
        raise SystemExit(f"\n{stale}")


if __name__ == "__main__":
    main()
