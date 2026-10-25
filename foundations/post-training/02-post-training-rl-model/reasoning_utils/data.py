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
