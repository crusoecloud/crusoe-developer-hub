"""Sampling helpers and self-consistency voting, all offline on tiny tensors or a tiny random model."""

import pytest
import torch

from reasoning_utils.generate import generate_tokens, scale_by_temperature, top_p_filter
from reasoning_utils.model import Qwen3Config, Qwen3Model
from reasoning_utils.voting import majority_vote

TINY = Qwen3Config(
    vocab_size=50,
    context_length=64,
    emb_dim=16,
    n_layers=2,
    n_heads=4,
    n_kv_groups=2,
    head_dim=4,
    hidden_dim=24,
    dtype=torch.float32,
)


def test_scale_by_temperature():
    logits = torch.tensor([[2.0, 4.0]])
    assert torch.allclose(scale_by_temperature(logits, 2.0), torch.tensor([[1.0, 2.0]]))
    with pytest.raises(ValueError):
        scale_by_temperature(logits, 0.0)


def test_top_p_filter_keeps_the_crossing_token_and_renormalizes():
    probs = torch.tensor([[0.1, 0.5, 0.4]])
    out = top_p_filter(probs, top_p=0.6)  # 0.5 then 0.4 crosses 0.6, so both stay; 0.1 goes
    assert torch.allclose(out, torch.tensor([[0.0, 0.5 / 0.9, 0.4 / 0.9]]))
    assert torch.allclose(top_p_filter(probs, top_p=1.0), probs)
    assert torch.allclose(top_p_filter(probs, None), probs)


def test_top_p_filter_rows_are_independent():
    probs = torch.tensor([[0.40, 0.30, 0.20, 0.10], [0.05, 0.25, 0.35, 0.35]])
    out = top_p_filter(probs, top_p=0.7)
    assert out.shape == probs.shape
    assert torch.all(out >= 0)
    assert torch.allclose(out.sum(dim=-1), torch.ones(2), atol=1e-6)
    assert out[0, 3] == 0 and out[1, 0] == 0


def test_top_p_filter_draws_only_from_kept_tokens():
    torch.manual_seed(0)
    probs = torch.softmax(torch.tensor([[5.0, 4.9, -20.0, -20.0]]), dim=-1)
    draws = torch.multinomial(top_p_filter(probs, top_p=0.9), num_samples=200, replacement=True)
    assert set(draws.flatten().tolist()) == {0, 1}


def test_generate_tokens_zero_temperature_is_greedy():
    model = Qwen3Model(TINY).eval()
    prompt = torch.randint(0, TINY.vocab_size, (1, 5))
    greedy = [t.item() for t in generate_tokens(model, prompt, 6)]
    torch.manual_seed(0)
    again = [t.item() for t in generate_tokens(model, prompt, 6, temperature=0.0, top_p=0.5)]
    assert greedy == again


def test_generate_tokens_sampling_is_reproducible_and_valid():
    model = Qwen3Model(TINY).eval()
    prompt = torch.randint(0, TINY.vocab_size, (1, 5))

    torch.manual_seed(3)
    a = [t.item() for t in generate_tokens(model, prompt, 8, temperature=1.0, top_p=0.9)]
    torch.manual_seed(3)
    b = [t.item() for t in generate_tokens(model, prompt, 8, temperature=1.0, top_p=0.9)]
    assert a == b
    assert len(a) == 8 and all(0 <= t < TINY.vocab_size for t in a)


def test_majority_vote_pools_equivalent_answers_and_breaks_ties_by_first_seen():
    final, votes = majority_vote(["7/2", "4", r"\frac{7}{2}", "3.5", "4"])
    assert final == "7/2"
    assert votes == {"7/2": 3, "4": 2}
    assert majority_vote(["4", "5", "5", "4"])[0] == "4"
    assert majority_vote(["", "", "x"])[0] == ""
