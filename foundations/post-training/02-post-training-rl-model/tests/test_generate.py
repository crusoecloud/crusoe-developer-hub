"""Generation on a tiny random model: cached greedy decoding must equal the naive loop."""

import torch

from reasoning_utils.generate import generate_tokens
from reasoning_utils.model import Qwen3Config, Qwen3Model

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


def naive_greedy(model, token_ids, max_new_tokens):
    """Recompute the whole sequence every step. Slow, but obviously correct."""
    out = []
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits = model(token_ids)[:, -1]
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
            out.append(nxt)
            token_ids = torch.cat([token_ids, nxt], dim=1)
    return torch.cat(out, dim=1)


def test_cached_matches_naive():
    torch.manual_seed(1)
    model = Qwen3Model(TINY).eval()
    prompt = torch.randint(0, TINY.vocab_size, (1, 5))

    cached = torch.cat(list(generate_tokens(model, prompt, max_new_tokens=12)), dim=1)
    naive = naive_greedy(model, prompt, max_new_tokens=12)
    assert cached.shape == (1, 12)
    assert torch.equal(cached, naive)


def test_stops_at_eos():
    torch.manual_seed(1)
    model = Qwen3Model(TINY).eval()
    prompt = torch.randint(0, TINY.vocab_size, (1, 5))
    first = next(generate_tokens(model, prompt, max_new_tokens=5)).item()

    # Declare the first generated token to be EOS: nothing should be yielded.
    assert list(generate_tokens(model, prompt, max_new_tokens=5, eos_token_id=first)) == []
