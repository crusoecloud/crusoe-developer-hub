"""Greedy, KV-cached text generation.

The text generation notebook builds these two functions step by step. Later
notebooks import them from here so they can focus on what sits on top:
sampling, voting, self-refinement, and reinforcement learning.
"""

from __future__ import annotations

import torch

from .model import KVCache


@torch.inference_mode()
def generate_tokens(model, token_ids: torch.Tensor, max_new_tokens: int, eos_token_id: int | None = None):
    """Yield one new token at a time, shape (batch, 1), using greedy decoding and a KV cache.

    The prompt is processed in one forward pass (prefill). After that, every
    step feeds only the newest token; the cache remembers everything before it.
    """
    model.eval()
    cache = KVCache(n_layers=len(model.layers))
    model.reset_kv_cache()

    logits = model(token_ids, cache=cache)[:, -1]  # prefill: scores for the token after the prompt
    for step in range(max_new_tokens):
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break
        yield next_token
        if step + 1 < max_new_tokens:
            logits = model(next_token, cache=cache)[:, -1]


def generate_text(
    model, tokenizer, prompt: str, device, max_new_tokens: int = 512, stream: bool = False
) -> str:
    """Prompt in, decoded completion out. Set stream=True to print tokens as they arrive."""
    token_ids = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)
    generated = []
    for token in generate_tokens(model, token_ids, max_new_tokens, tokenizer.eos_token_id):
        token_id = token.squeeze(0).item()
        generated.append(token_id)
        if stream:
            print(tokenizer.decode([token_id]), end="", flush=True)
    return tokenizer.decode(generated)
