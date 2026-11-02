"""Run a model over a set of math problems and report accuracy.

Introduced in the math verifier notebook. Every later notebook calls this to
answer the only question that matters: did the change make the model solve
more problems? The sampling notebook adds `temperature`, `top_p`, and
`n_samples` (self-consistency voting) without changing the loop itself. The
self-refinement notebook adds `score_fn` with `select="best"` (best-of-n) and
`refine_iterations` (critique and rewrite).
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

# The same template with one sentence appended after "Answer:", asking for the work, not just the result.
COT_PROMPT = MATH_PROMPT + " Show your work step by step."


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
    temperature: float = 0.0,
    top_p: float | None = None,
    n_samples: int = 1,
    select: str = "vote",
    score_fn=None,
    refine_iterations: int = 0,
) -> dict:
    """Generate, extract, grade. Returns a summary dict and optionally writes one JSON line per problem.

    With n_samples > 1 each problem is answered by several sampled completions:
    a self-consistency vote (`select="vote"`) or the one `score_fn` ranks
    highest (`select="best"`). With refine_iterations > 0 the draft is
    critiqued and rewritten that many times, `score_fn` deciding whether each
    rewrite is kept. `tokens` counts every generated text, which is the real
    cost of each method.
    """
    from .refine import self_refine  # imported here because refine.py imports the prompt templates above
    from .scoring import best_of_n
    from .voting import self_consistency

    if select not in ("vote", "best"):
        raise ValueError(f"select must be 'vote' or 'best', got {select!r}")
    if select == "best" and n_samples > 1 and score_fn is None:
        raise ValueError("select='best' needs a score_fn")
    if refine_iterations > 0 and n_samples > 1:
        raise ValueError("refine_iterations and n_samples > 1 cannot be combined; refinement works on one draft")

    records = []
    n_correct = 0
    total_tokens = 0
    start = time.time()
    if log_path:
        Path(log_path).write_text("", encoding="utf-8")  # start a fresh log; records are appended below

    for i, row in enumerate(problems, start=1):
        prompt = render_math_prompt(row["problem"], template)
        if refine_iterations > 0:
            refined = self_refine(
                model,
                tokenizer,
                row["problem"],
                device,
                refine_iterations,
                score_fn,
                template,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            completion, predicted = refined["final"], refined["answer"]
            completions, answers = refined["generated"], [predicted]
        elif n_samples == 1:
            completion = generate_text(
                model, tokenizer, prompt, device, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p
            )
            completions = [completion]
            predicted = extract_final_answer(completion)
            answers = [predicted]
        elif select == "best":
            pick = best_of_n(
                model, tokenizer, prompt, device, score_fn, n_samples, temperature, top_p, max_new_tokens=max_new_tokens
            )
            completions, completion = pick["completions"], pick["final"]
            answers = [extract_final_answer(c) for c in completions]
            predicted = answers[pick["best"]]
        else:
            vote = self_consistency(
                model, tokenizer, prompt, device, n_samples, temperature, top_p, max_new_tokens=max_new_tokens
            )
            completions, answers, predicted = vote["completions"], vote["answers"], vote["final"]
            completion = completions[answers.index(predicted)]  # the winning chain, for the log
        correct = grade(predicted, row["answer"])

        n_correct += int(correct)
        n_tokens = sum(len(tokenizer.encode(c, chat_template=False)) for c in completions)
        total_tokens += n_tokens
        record = {
            "index": i,
            "problem": row["problem"],
            "answer": row["answer"],
            "subject": row.get("subject"),
            "level": row.get("level"),
            "completion": completion,
            "predicted": predicted,
            "answers": answers,
            "correct": correct,
            "tokens": n_tokens,
        }
        records.append(record)
        if log_path:
            # Open per record so an exception anywhere in the loop never leaves a handle dangling.
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        if verbose:
            print(f"{i}/{len(problems)} predicted: {predicted!r} answer: {row['answer']!r} correct: {correct}")
        elif i % 10 == 0 or i == len(problems):
            print(f"{i}/{len(problems)} correct so far: {n_correct}")

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
    print(f"accuracy: {summary['accuracy']:.1%} ({n_correct}/{n})")
    print(f"average response: {summary['avg_tokens']:.0f} tokens")
    print(f"time: {seconds / 60:.1f} min")
    return summary
