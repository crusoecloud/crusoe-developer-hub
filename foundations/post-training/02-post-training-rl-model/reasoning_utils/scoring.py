"""Scores for a model response that need no reference answer.

Built in the self-refinement notebook. A score function takes `(prompt, text)`
and returns a float, higher is better. Two are provided: a rule-based score
that looks at the format and length of the text, and the model's own
confidence, the average log-probability of the response tokens. Either can
rank sampled chains (best-of-n) or gate a revision in the refinement loop.
"""

from __future__ import annotations

import math

import torch

from .generate import generate_text
from .verify import NUMBER, last_boxed


def heuristic_score(prompt, text, brevity_scale=500.0):
    """Reward a boxed final answer, then any number, and shorter text. The prompt is ignored."""
    score = 0.0
    if last_boxed(text) is not None:
        score += 2.0  # a boxed final answer, the format we asked for
    elif NUMBER.search(text):
        score += 1.0  # no box, but at least a number to extract
    score += 1.5 * math.exp(-len(text) / brevity_scale)  # a brevity bonus that decays with length
    return score


@torch.inference_mode()
def token_logprobs(model, tokenizer, text, device):
    """The log-probability the model assigns to each token of `text` given the tokens before it."""
    token_ids = torch.tensor(tokenizer.encode(text), device=device)
    logits = model(token_ids.unsqueeze(0)).squeeze(0).float()
    logprobs = torch.log_softmax(logits, dim=-1)
    positions = torch.arange(len(token_ids) - 1, device=device)  # position t predicts token t + 1
    next_ids = token_ids[1:]
    return logprobs[positions, next_ids]


@torch.inference_mode()
def avg_logprob(model, tokenizer, prompt, answer, device):
    """Average log-probability of the answer tokens given the prompt: the model's confidence in its own text."""
    prompt_ids = tokenizer.encode(prompt)
    answer_ids = tokenizer.encode(answer)
    if not answer_ids:
        return float("-inf")  # an empty answer is the least confident answer there is
    token_ids = torch.tensor(prompt_ids + answer_ids, device=device)

    logits = model(token_ids.unsqueeze(0)).squeeze(0).float()
    logprobs = torch.log_softmax(logits, dim=-1)

    start = len(prompt_ids) - 1  # the position that predicts the first answer token
    positions = torch.arange(start, len(token_ids) - 1, device=device)
    next_ids = token_ids[start + 1 :]
    return logprobs[positions, next_ids].mean().item()


def best_of_n(
    model, tokenizer, prompt, device, score_fn, n_samples=5, temperature=0.8, top_p=0.9, max_new_tokens=1024, verbose=False
):
    """Sample n_samples completions, score each one, and keep the highest-scoring text."""
    completions, scores = [], []
    for i in range(n_samples):
        text = generate_text(
            model, tokenizer, prompt, device, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p
        )
        completions.append(text)
        scores.append(score_fn(prompt, text))
        if verbose:
            print(f"sample {i + 1}/{n_samples}: score {scores[-1]:.3f}")

    best = max(range(n_samples), key=lambda i: scores[i])
    return {"completions": completions, "scores": scores, "best": best, "final": completions[best]}
