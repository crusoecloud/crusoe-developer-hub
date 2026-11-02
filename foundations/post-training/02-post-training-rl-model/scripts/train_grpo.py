"""Train Qwen3-0.6B-Base on MATH problems with GRPO from the command line.

Examples, from the track directory:

    uv run scripts/train_grpo.py
    uv run scripts/train_grpo.py --steps 200 --n-rollouts 8 --max-new-tokens 512 --checkpoint-every 25

Writes one JSON line per step to outputs/ and checkpoints to checkpoints/. Evaluate a checkpoint with

    uv run scripts/evaluate_math500.py --model base --n 500 --compile --checkpoint checkpoints/<run>/grpo-step0100.pt
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reasoning_utils import get_device, load_math_train, load_qwen3, save_checkpoint, train_grpo  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--n-rollouts", type=int, default=4, help="sampled responses per problem")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-compile", action="store_true", help="skip torch.compile (slower steps, no warm-up)")
    parser.add_argument("--run", default=None, help="run name; default grpo_<steps>x<rollouts>")
    args = parser.parse_args()

    run = args.run or f"grpo_{args.steps}x{args.n_rollouts}"
    device = get_device()
    model, tokenizer = load_qwen3("base", device=device, compile=not args.no_compile)
    model = model.float()  # train with float32 weights; see the GRPO notebook for why
    train_data = load_math_train()
    random.Random(args.seed).shuffle(train_data)
    torch.manual_seed(args.seed)

    train_grpo(
        model,
        tokenizer,
        train_data,
        device,
        steps=args.steps,
        n_rollouts=args.n_rollouts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        lr=args.lr,
        log_path=Path("outputs") / f"{run}.jsonl",
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=Path("checkpoints") / run,
    )
    save_checkpoint(model, Path("checkpoints") / run / "final.pt")


if __name__ == "__main__":
    main()
