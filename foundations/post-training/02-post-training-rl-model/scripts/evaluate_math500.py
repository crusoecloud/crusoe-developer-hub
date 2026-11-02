"""Evaluate Qwen3-0.6B on MATH-500 from the command line.

Examples, from the track directory:

    uv run scripts/evaluate_math500.py --model base --n 50
    uv run scripts/evaluate_math500.py --model reasoning --n 500 --compile
    uv run scripts/evaluate_math500.py --model base --n 500 --compile --cot --n-samples 5 --temperature 0.8 --top-p 0.9
    uv run scripts/evaluate_math500.py --model base --n 500 --compile --cot --n-samples 5 --select best --score confidence
    uv run scripts/evaluate_math500.py --model base --n 500 --compile --refine 1 --score heuristic --temperature 0.7 --top-p 0.9
    uv run scripts/evaluate_math500.py --model base --n 500 --compile --checkpoint checkpoints/grpo_100x8/final.pt

Writes one JSON line per problem to outputs/ plus a small summary JSON next to it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reasoning_utils import (  # noqa: E402
    COT_PROMPT,
    MATH_PROMPT,
    avg_logprob,
    evaluate_math,
    get_device,
    heuristic_score,
    load_math500,
    load_qwen3,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["base", "reasoning"], default="base")
    parser.add_argument("--n", type=int, default=500, help="number of problems, taken from the start")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--compile", action="store_true", help="torch.compile the model (slow first call, then faster)")
    parser.add_argument("--cot", action="store_true", help="use the step-by-step prompt template")
    parser.add_argument("--template-file", default=None, help="text file with a prompt template containing {problem}")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 is greedy; above 0 samples")
    parser.add_argument("--top-p", type=float, default=None, help="top-p cutoff for sampling, e.g. 0.9")
    parser.add_argument("--n-samples", type=int, default=1, help="chains per problem; above 1 uses a self-consistency vote")
    parser.add_argument("--select", choices=["vote", "best"], default="vote", help="how to pick among sampled chains")
    parser.add_argument("--score", choices=["none", "heuristic", "confidence"], default="none", help="score function")
    parser.add_argument("--refine", type=int, default=0, help="critique-and-rewrite iterations per problem")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default=None, help="a state dict saved by the GRPO notebook or scripts/train_grpo.py")
    parser.add_argument("--out", default=None, help="JSONL path; default outputs/math500_<model>_<n>.jsonl")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = get_device()
    model, tokenizer = load_qwen3(args.model, device=device, compile=args.compile, checkpoint=args.checkpoint)
    problems = load_math500()[: args.n]
    if args.template_file:
        template = Path(args.template_file).read_text()
    else:
        template = COT_PROMPT if args.cot else MATH_PROMPT
    torch.manual_seed(args.seed)

    score_fn = None
    if args.score == "heuristic":
        score_fn = heuristic_score
    elif args.score == "confidence":

        def score_fn(prompt, text):
            return avg_logprob(model, tokenizer, prompt, text, device)

    tag = f"{args.model}_{args.n}" + ("_cot" if args.cot else "")
    if args.n_samples > 1:
        tag += f"_{args.select}{args.n_samples}"
    if args.refine:
        tag += f"_refine{args.refine}"
    if args.score != "none":
        tag += f"_{args.score}"
    if args.checkpoint:
        tag += "_" + Path(args.checkpoint).stem
    out = Path(args.out or f"outputs/math500_{tag}.jsonl")
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
        temperature=args.temperature,
        top_p=args.top_p,
        n_samples=args.n_samples,
        select=args.select,
        score_fn=score_fn,
        refine_iterations=args.refine,
    )
    summary.pop("records")
    summary.update(
        model=args.model,
        compile=args.compile,
        max_new_tokens=args.max_new_tokens,
        cot=args.cot,
        temperature=args.temperature,
        top_p=args.top_p,
        n_samples=args.n_samples,
        select=args.select,
        score=args.score,
        refine=args.refine,
        checkpoint=args.checkpoint,
        device=torch.cuda.get_device_name(0) if device.type == "cuda" else str(device),
        torch=torch.__version__,
    )
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(f"summary written to {out.with_suffix('.summary.json')}")


if __name__ == "__main__":
    main()
