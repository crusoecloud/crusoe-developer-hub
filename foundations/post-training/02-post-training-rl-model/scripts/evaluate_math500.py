"""Evaluate Qwen3-0.6B on MATH-500 from the command line.

Examples, from the track directory:

    uv run scripts/evaluate_math500.py --model base --n 50
    uv run scripts/evaluate_math500.py --model reasoning --n 500 --compile

Writes one JSON line per problem to outputs/ plus a small summary JSON next to it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reasoning_utils import MATH_PROMPT, evaluate_math, get_device, load_math500, load_qwen3  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["base", "reasoning"], default="base")
    parser.add_argument("--n", type=int, default=500, help="number of problems, taken from the start")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--compile", action="store_true", help="torch.compile the model (slow first call, then faster)")
    parser.add_argument("--template-file", default=None, help="text file with a prompt template containing {problem}")
    parser.add_argument("--out", default=None, help="JSONL path; default outputs/math500_<model>_<n>.jsonl")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = get_device()
    model, tokenizer = load_qwen3(args.model, device=device, compile=args.compile)
    problems = load_math500()[: args.n]
    template = Path(args.template_file).read_text() if args.template_file else MATH_PROMPT

    out = Path(args.out or f"outputs/math500_{args.model}_{args.n}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    summary = evaluate_math(
        model,
        tokenizer,
        device,
        problems,
        max_new_tokens=args.max_new_tokens,
        template=template,
        log_path=out,
        verbose=args.verbose,
    )
    summary.pop("records")
    summary.update(
        model=args.model,
        compile=args.compile,
        max_new_tokens=args.max_new_tokens,
        device=torch.cuda.get_device_name(0) if device.type == "cuda" else str(device),
        torch=torch.__version__,
    )
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(f"summary written to {out.with_suffix('.summary.json')}")


if __name__ == "__main__":
    main()
