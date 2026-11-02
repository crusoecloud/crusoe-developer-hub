"""KV-cached text generation: greedy by default, sampling on request.

The text generation notebook builds `generate_tokens` and `generate_text`
step by step with greedy decoding. The sampling notebook adds temperature
scaling and top-p filtering to the same two functions. Later notebooks
import them from here.
"""

from __future__ import annotations

import torch

from .model import KVCache


def scale_by_temperature(logits, temperature):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return logits / temperature


def top_p_filter(probs, top_p):
    if top_p is None or top_p >= 1.0:
        return probs

    # sort by descending probability and take the cumulative sum
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # keep a token while the mass of the tokens ranked above it is below top_p,
    # so the token that crosses the threshold is kept too
    keep = (cumulative - sorted_probs) < top_p

    # zero out the rest and map back to vocabulary order
    kept_sorted = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
    filtered = torch.zeros_like(probs).scatter(-1, sorted_idx, kept_sorted)

    # renormalize so the survivors sum to 1
    return filtered / filtered.sum(dim=-1, keepdim=True)


@torch.inference_mode()
def generate_tokens(model, token_ids, max_new_tokens, eos_token_id=None, temperature=0.0, top_p=None):
    """Yield one new token at a time, shape (batch, 1), using a KV cache.

    The prompt is processed in one forward pass (prefill). After that, every
    step feeds only the newest token; the cache remembers everything before it.
    temperature=0 is greedy decoding; anything above 0 samples.
    """
    model.eval()
    cache = KVCache(n_layers=len(model.layers))
    model.reset_kv_cache()

    logits = model(token_ids, cache=cache)[:, -1]  # prefill: scores for the token after the prompt
    for step in range(max_new_tokens):
        if temperature == 0.0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)  # greedy decoding, as before
        else:
            # NEW: scale the logits, turn them into probabilities, drop the tail, draw a token
            probs = torch.softmax(scale_by_temperature(logits.float(), temperature), dim=-1)
            probs = top_p_filter(probs, top_p)
            next_token = torch.multinomial(probs, num_samples=1)

        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break
        yield next_token
        if step + 1 < max_new_tokens:
            logits = model(next_token, cache=cache)[:, -1]


def generate_text(
    model, tokenizer, prompt, device, max_new_tokens=512, stream=False, temperature=0.0, top_p=None
):
    """Prompt in, decoded completion out. Set stream=True to print tokens as they arrive."""
    token_ids = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)

    generated = []
    for token in generate_tokens(model, token_ids, max_new_tokens, tokenizer.eos_token_id, temperature, top_p):
        token_id = token.squeeze(0).item()
        generated.append(token_id)
        if stream:
            print(tokenizer.decode([token_id]), end="", flush=True)
    return tokenizer.decode(generated)
