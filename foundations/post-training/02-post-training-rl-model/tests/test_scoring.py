"""Score functions, best-of-n, and the self-refinement loop, offline on a tiny random model or fakes."""

import math

import pytest
import torch

from reasoning_utils.evaluate import MATH_PROMPT, evaluate_math, render_math_prompt
from reasoning_utils.model import Qwen3Config, Qwen3Model
from reasoning_utils.refine import render_critique_prompt, render_refine_prompt, self_refine
from reasoning_utils.scoring import avg_logprob, best_of_n, heuristic_score, token_logprobs

TINY = Qwen3Config(
    vocab_size=50,
    context_length=2048,  # the critique and refine prompts are a few hundred characters, one token each here
    emb_dim=16,
    n_layers=2,
    n_heads=4,
    n_kv_groups=2,
    head_dim=4,
    hidden_dim=24,
    dtype=torch.float32,
)


class FakeTokenizer:
    """Maps each character to a token id, so lengths are predictable."""

    eos_token_id = 0

    def encode(self, text, chat_template=None):
        return [1 + (ord(c) % (TINY.vocab_size - 1)) for c in text]

    def decode(self, ids):
        return "".join(chr(97 + (i % 26)) for i in ids)


def test_heuristic_score_prefers_boxed_then_numbers_then_brevity():
    boxed = heuristic_score("p", r"so \boxed{7}")
    number = heuristic_score("p", "the answer is 7")
    nothing = heuristic_score("p", "no idea")
    assert boxed > number > nothing
    long_boxed = heuristic_score("p", r"\boxed{7}" + " filler" * 400)
    assert boxed > long_boxed > 2.0
    assert math.isclose(nothing, 1.5 * math.exp(-len("no idea") / 500.0))


def test_token_logprobs_shape_and_range():
    model = Qwen3Model(TINY).eval()
    tok = FakeTokenizer()
    lp = token_logprobs(model, tok, "hello", "cpu")
    assert lp.shape == (4,)  # five tokens, four predictions
    assert torch.all(lp <= 0)


def test_avg_logprob_scores_only_the_answer_tokens():
    model = Qwen3Model(TINY).eval()
    tok = FakeTokenizer()
    full = token_logprobs(model, tok, "abc" + "xyz", "cpu")
    score = avg_logprob(model, tok, "abc", "xyz", "cpu")
    assert math.isclose(score, full[-3:].mean().item(), rel_tol=1e-5)


def test_best_of_n_keeps_the_highest_scoring_text():
    model = Qwen3Model(TINY).eval()
    tok = FakeTokenizer()
    torch.manual_seed(0)
    out = best_of_n(model, tok, "ab", "cpu", score_fn=lambda p, t: len(t), n_samples=3, max_new_tokens=5)
    assert len(out["completions"]) == 3
    assert out["final"] == out["completions"][out["best"]]
    assert out["scores"][out["best"]] == max(out["scores"])


def test_render_prompts_keep_braces():
    c = render_critique_prompt("1+1?", "  2  ")
    assert "Question:\n1+1?" in c and "Solution:\n2\n" in c
    r = render_refine_prompt("1+1?", "2", "fine")
    assert r"\boxed{ANSWER}" in r and "Critique:\nfine" in r


def test_self_refine_without_scorer_keeps_every_revision():
    model = Qwen3Model(TINY).eval()
    tok = FakeTokenizer()
    torch.manual_seed(0)
    out = self_refine(model, tok, "1+1?", "cpu", iterations=2, max_new_tokens=4, max_critique_tokens=3)
    assert len(out["steps"]) == 2
    assert len(out["generated"]) == 5  # draft plus two critique and revision pairs
    assert out["final"] == out["steps"][-1]["revised"]


def test_self_refine_with_scorer_rejects_a_worse_revision():
    model = Qwen3Model(TINY).eval()
    tok = FakeTokenizer()
    calls = iter([5.0, 1.0])  # draft scores 5, the revision scores 1
    torch.manual_seed(0)
    out = self_refine(model, tok, "1+1?", "cpu", iterations=1, score_fn=lambda p, t: next(calls), max_new_tokens=4)
    assert out["final"] == out["steps"][0]["draft"]
    assert out["steps"][0]["score_before"] == 5.0 and out["steps"][0]["score_after"] == 1.0


def test_evaluate_math_routes_best_and_refine(capsys):
    model = Qwen3Model(TINY).eval()
    tok = FakeTokenizer()
    problems = [{"problem": "1+1?", "answer": "2"}]
    torch.manual_seed(0)
    best = evaluate_math(
        model, tok, "cpu", problems, max_new_tokens=3, n_samples=2, select="best", score_fn=lambda p, t: 1.0
    )
    assert best["n"] == 1 and len(best["records"][0]["answers"]) == 2
    refined = evaluate_math(model, tok, "cpu", problems, max_new_tokens=3, refine_iterations=1)
    assert refined["records"][0]["tokens"] > 0
    assert render_math_prompt("x", MATH_PROMPT).endswith("Answer:")
    with pytest.raises(ValueError):
        evaluate_math(model, tok, "cpu", problems, max_new_tokens=3, n_samples=2, refine_iterations=1)
