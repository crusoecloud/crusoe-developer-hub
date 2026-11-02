"""Self-consistency: sample several answers to the same question and keep the most common one.

Built in the sampling notebook on top of `generate_text` (sampling) and the
verifier (answer extraction and equivalence). The evaluation harness calls it
when asked for more than one sample per problem.
"""

from __future__ import annotations

from collections import Counter

from .generate import generate_text
from .verify import extract_final_answer, grade


def majority_vote(answers):
    """The most common answer and the full tally.

    Two answers count as the same vote when the verifier says they are
    equivalent, so "7/2", "\\frac{7}{2}" and "3.5" pool their votes. The tally
    is keyed by the first spelling seen in each group, and a tie goes to the
    group that appeared first.
    """
    votes = Counter()
    for answer in answers:
        for seen in votes:
            if answer == seen or grade(answer, seen):
                votes[seen] += 1
                break
        else:
            votes[answer] += 1
    return votes.most_common(1)[0][0], votes


def self_consistency(
    model, tokenizer, prompt, device, n_samples=5, temperature=0.8, top_p=0.9, max_new_tokens=1024, verbose=False
):
    """Sample n_samples completions, extract each final answer, and vote.

    Returns the completions, the extracted answers, the tally, and `final`,
    the answer with the most votes.
    """
    completions, answers = [], []
    for i in range(n_samples):
        text = generate_text(
            model, tokenizer, prompt, device, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p
        )
        completions.append(text)
        answers.append(extract_final_answer(text))
        if verbose:
            print(f"sample {i + 1}/{n_samples}: {answers[-1]!r}")

    final, votes = majority_vote(answers)
    return {"completions": completions, "answers": answers, "votes": votes, "final": final}
