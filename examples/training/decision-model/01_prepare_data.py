#!/usr/bin/env python3
"""Build chat-format SFT data from AG News, shaped for a *decision model*.

The one design choice that matters: every class label STARTS WITH A DIFFERENT
token in the base model's tokenizer. After fine-tuning, the first token picks the
label and the rest of it follows with ~100% certainty, so the first-token
logprobs of one `max_tokens=1` call are the class probabilities. (Ours happen to
be single tokens: World / Sports / Business / Technology for Qwen3.5.)

    pip install pyarrow tokenizers
    python 01_prepare_data.py            # writes data/{train,val,test}.jsonl
"""
import json
import random
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

from decision_model import LABELS, SYSTEM  # LABELS follow AG News ids 0..3 (3 = Sci/Tech -> "Technology")

HF = "https://huggingface.co/datasets/fancyzhx/ag_news/resolve/main/data/{}-00000-of-00001.parquet"
TOKENIZER = "https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/tokenizer.json"
N_TRAIN_PER_CLASS, N_VAL_PER_CLASS, N_TEST_PER_CLASS = 2000, 200, 500
SEED = 7
OUT = Path(__file__).parent / "data"


def fetch(url: str, dest: Path, attempts: int = 4) -> Path:
    if dest.exists():
        return dest
    for attempt in range(1, attempts + 1):
        print(f"downloading {url} (attempt {attempt})")
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(dest)  # only a complete download ever lands at `dest`
            return dest
        except OSError as err:
            tmp.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            print(f"  failed ({err}); retrying")


def clean(text: str) -> str:
    return " ".join(text.replace("\\", " ").split())  # AG News has stray backslashes


def rows(path: Path):
    cols = pq.read_table(path).to_pydict()
    return [(clean(t), l) for t, l in zip(cols["text"], cols["label"])]


def balanced(data, per_class, rng):
    out = []
    for label in range(len(LABELS)):
        pool = [r for r in data if r[1] == label]
        out += rng.sample(pool, per_class)
    rng.shuffle(out)
    return out


def write(name, data):
    path = OUT / f"{name}.jsonl"
    with path.open("w") as f:
        for text, label in data:
            f.write(json.dumps({"messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": text},
                {"role": "assistant", "content": LABELS[label]},
            ]}) + "\n")
    print(f"{path}  {len(data)} rows")


def check_first_tokens():
    """Labels may be several tokens long, but no two may start with the same token."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(fetch(TOKENIZER, OUT / "tokenizer.json")))
    first = {l: tok.encode(l, add_special_tokens=False).ids[0] for l in LABELS}
    clashes = [l for l in LABELS if list(first.values()).count(first[l]) > 1]
    assert not clashes, f"{clashes} start with the same token; rename them so each starts differently"
    print("every label starts with a different token:", first)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    check_first_tokens()
    rng = random.Random(SEED)
    train_pool = rows(fetch(HF.format("train"), OUT / "train.parquet"))
    test_pool = rows(fetch(HF.format("test"), OUT / "test.parquet"))
    train = balanced(train_pool, N_TRAIN_PER_CLASS + N_VAL_PER_CLASS, rng)
    by_class = {l: [r for r in train if r[1] == l] for l in range(len(LABELS))}
    val = [r for l in by_class for r in by_class[l][:N_VAL_PER_CLASS]]
    train = [r for l in by_class for r in by_class[l][N_VAL_PER_CLASS:]]
    rng.shuffle(train); rng.shuffle(val)
    write("train", train)
    write("val", val)
    write("test", balanced(test_pool, N_TEST_PER_CLASS, rng))
