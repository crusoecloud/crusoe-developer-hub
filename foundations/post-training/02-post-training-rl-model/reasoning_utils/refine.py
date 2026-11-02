"""Self-refinement: the model critiques its own solution and rewrites it.

Built in the self-refinement notebook on top of `generate_text` (sampling)
and the score functions in `scoring.py`. The evaluation harness calls
`self_refine` when asked for one or more refinement iterations.
"""

from __future__ import annotations

from .evaluate import MATH_PROMPT, render_math_prompt
from .generate import generate_text
from .verify import extract_final_answer

CRITIQUE_PROMPT = (
    "You are checking a solution to a math problem. Point out any mistake in the reasoning or the arithmetic, "
    "or say that the solution is correct. Then give a short plan for fixing it.\n\n"
    "Question:\n{problem}\n\nSolution:\n{draft}\n\nCritique (at most 100 words):"
)

REFINE_PROMPT = (
    "Rewrite the solution using the critique. Show the corrected steps and write the final result "
    "on a new line as:\n\\boxed{ANSWER}\n\n"
    "Question:\n{problem}\n\nSolution:\n{draft}\n\nCritique:\n{critique}\n\nRevised solution:"
)


def render_critique_prompt(problem, draft):
    return CRITIQUE_PROMPT.replace("{problem}", problem).replace("{draft}", draft.strip())


def render_refine_prompt(problem, draft, critique):
    return (
        REFINE_PROMPT.replace("{problem}", problem)
        .replace("{draft}", draft.strip())
        .replace("{critique}", critique.strip())
    )


def self_refine(
    model,
    tokenizer,
    problem,
    device,
    iterations=1,
    score_fn=None,
    template=MATH_PROMPT,
    max_new_tokens=1024,
    max_critique_tokens=256,
    temperature=0.7,
    top_p=0.9,
    verbose=False,
):
    """Draft a solution, then critique and rewrite it `iterations` times.

    With a `score_fn(prompt, text)` a revision is kept only when it scores at
    least as high as the text it replaces; without one every revision is kept.
    An empty revision (the model stopped right away, usually after a critique
    that found nothing to fix) is never kept. Returns the final text, its
    extracted answer, one record per iteration, and every text the model
    generated along the way, which is the real cost.
    """
    prompt = render_math_prompt(problem, template)
    current = generate_text(
        model, tokenizer, prompt, device, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p
    )
    current_score = score_fn(prompt, current) if score_fn else 0.0
    generated = [current]
    steps = []

    for it in range(iterations):
        critique = generate_text(
            model,
            tokenizer,
            render_critique_prompt(problem, current),
            device,
            max_new_tokens=max_critique_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        revised = generate_text(
            model,
            tokenizer,
            render_refine_prompt(problem, current, critique),
            device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        revised_score = score_fn(prompt, revised) if score_fn else 0.0
        generated += [critique, revised]
        steps.append(
            {
                "draft": current,
                "critique": critique,
                "revised": revised,
                "score_before": current_score,
                "score_after": revised_score,
            }
        )
        if verbose:
            print(
                f"iteration {it + 1}: {extract_final_answer(current)!r} -> {extract_final_answer(revised)!r},",
                f"score {current_score:.3f} -> {revised_score:.3f}",
            )
        if revised.strip() and revised_score >= current_score:  # keep it unless empty or scored worse
            current, current_score = revised, revised_score

    return {"final": current, "answer": extract_final_answer(current), "steps": steps, "generated": generated}
