"""Run a model over a set of math problems and report accuracy.

Introduced in the math verifier notebook. Every later notebook calls this to
answer the only question that matters: did the change make the model solve
more problems?
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .generate import generate_text
from .verify import extract_final_answer, grade

MATH_PROMPT = (
    "You are a helpful math assistant.\n"
    "Answer the question and write the final result on a new line as:\n"
    "\\boxed{ANSWER}\n\n"
    "Question:\n{problem}\n\nAnswer:"
)


def render_math_prompt(problem: str, template: str = MATH_PROMPT) -> str:
    """Fill `{problem}` into the template. Plain replace, so braces like \\boxed{ANSWER} survive."""
    return template.replace("{problem}", problem)


def evaluate_math(
    model,
    tokenizer,
    device,
    problems: list[dict],
    max_new_tokens: int = 2048,
    template: str = MATH_PROMPT,
    log_path: str | Path | None = None,
    verbose: bool = False,
) -> dict:
    """Generate, extract, grade. Returns a summary dict and optionally writes one JSON line per problem."""
    records = []
    n_correct = 0
    total_tokens = 0
    start = time.time()
    if log_path:
        Path(log_path).write_text("", encoding="utf-8")  # start a fresh log; records are appended below

    for i, row in enumerate(problems, start=1):
        prompt = render_math_prompt(row["problem"], template)
        completion = generate_text(model, tokenizer, prompt, device, max_new_tokens=max_new_tokens)
        predicted = extract_final_answer(completion)
        correct = grade(predicted, row["answer"])

        n_correct += int(correct)
        n_tokens = len(tokenizer.encode(completion, chat_template=False))
        total_tokens += n_tokens
        record = {
            "index": i,
            "problem": row["problem"],
            "answer": row["answer"],
            "subject": row.get("subject"),
            "level": row.get("level"),
            "completion": completion,
            "predicted": predicted,
            "correct": correct,
            "tokens": n_tokens,
        }
        records.append(record)
        if log_path:
            # Open per record so an exception anywhere in the loop never leaves a handle dangling.
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        elapsed = time.time() - start
        eta = elapsed / i * (len(problems) - i)
        status = f"{i}/{len(problems)}  correct so far: {n_correct}  elapsed: {elapsed / 60:.1f} min  eta: {eta / 60:.1f} min"
        if verbose:
            print(
                f"{status}\n  predicted: {predicted!r}  answer: {row['answer']!r}  {'OK' if correct else 'WRONG'}"
            )
        else:
            print(status.ljust(90), end="\r", flush=True)

    seconds = time.time() - start
    n = len(problems)
    summary = {
        "n": n,
        "n_correct": n_correct,
        "accuracy": n_correct / n if n else 0.0,
        "avg_tokens": total_tokens / n if n else 0.0,
        "seconds": seconds,
        "records": records,
    }
    print(
        f"\naccuracy: {summary['accuracy'] * 100:.1f}% ({n_correct}/{n})   "
        f"avg response: {summary['avg_tokens']:.0f} tokens   time: {seconds / 60:.1f} min"
    )
    return summary
