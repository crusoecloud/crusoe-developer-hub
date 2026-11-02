"""Dataset loading. Kept deliberately small: a list of dicts is all the lessons need."""

from __future__ import annotations

import json


def load_math500() -> list[dict]:
    """The MATH-500 benchmark: 500 competition problems with final answers.

    Each record has `problem`, `solution`, `answer`, `subject`, `level` (1 to 5),
    and `unique_id`. Downloaded once from the Hugging Face Hub and cached.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("HuggingFaceH4/MATH-500", "test.jsonl", repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_math_train() -> list[dict]:
    r"""The MATH training split: 7,500 competition problems with worked solutions.

    The records carry no final-answer field, so the reference answer is the last
    \boxed{...} of each solution, pulled out with the verifier's own `last_boxed`;
    the few records without a box or without a difficulty level are skipped. MATH-500 is drawn from the test
    split, so nothing here overlaps with the benchmark. Same field names as
    `load_math500`: `problem`, `solution`, `answer`, `subject`, `level`, `unique_id`.
    Downloaded once as one parquet file from a Hub mirror of the dataset and cached.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from .verify import last_boxed

    path = hf_hub_download("DigitalLearningGmbH/MATH-lighteval", "data/train-00000-of-00001.parquet", repo_type="dataset")
    records = []
    for i, row in enumerate(pq.read_table(path).to_pylist()):
        answer = last_boxed(row["solution"])
        level = (row["level"] or "").removeprefix("Level ")  # a missing level is skipped below
        if answer is None or not level.isdigit():
            continue
        records.append(
            {
                "problem": row["problem"],
                "solution": row["solution"],
                "answer": answer.strip(),
                "subject": row["type"],
                "level": int(level),
                "unique_id": f"train/{i}",
            }
        )
    return records
