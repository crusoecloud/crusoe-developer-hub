"""Offline checks of the transformer: shapes, KV-cache consistency, checkpoint key mapping."""

import torch

from reasoning_utils.model import KVCache, Qwen3Config, Qwen3Model, load_hf_state_dict

TINY = Qwen3Config(
    vocab_size=64,
    context_length=32,
    emb_dim=16,
    n_layers=2,
    n_heads=4,
    n_kv_groups=2,
    head_dim=4,
    hidden_dim=24,
    dtype=torch.float32,
)


def test_forward_shape():
    model = Qwen3Model(TINY).eval()
    ids = torch.randint(0, TINY.vocab_size, (2, 7))
    with torch.inference_mode():
        logits = model(ids)
    assert logits.shape == (2, 7, TINY.vocab_size)


def test_kv_cache_matches_full_forward():
    """Decoding one token at a time through the cache must give the same logits
    as recomputing the whole sequence from scratch."""
    torch.manual_seed(0)
    model = Qwen3Model(TINY).eval()
    ids = torch.randint(0, TINY.vocab_size, (1, 9))

    with torch.inference_mode():
        full = model(ids)  # (1, 9, vocab)

        cache = KVCache(TINY.n_layers)
        model.reset_kv_cache()
        prefix = ids[:, :4]
        cached = [model(prefix, cache=cache)]
        for t in range(4, ids.shape[1]):
            cached.append(model(ids[:, t : t + 1], cache=cache))
        cached = torch.cat(cached, dim=1)

    torch.testing.assert_close(cached, full, atol=1e-5, rtol=1e-5)


def test_hf_key_mapping_round_trip():
    """A checkpoint in Hugging Face naming loads strictly, with and without a tied head."""
    src = Qwen3Model(TINY)
    hf_style = {f"model.{k}": v for k, v in src.state_dict().items() if k != "lm_head.weight"}

    dst = Qwen3Model(TINY)
    load_hf_state_dict(dst, hf_style)
    assert dst.lm_head.weight is dst.embed_tokens.weight  # tied, since lm_head.weight was absent
    torch.testing.assert_close(dst.embed_tokens.weight, src.embed_tokens.weight)

    hf_style["lm_head.weight"] = torch.randn_like(src.lm_head.weight)
    dst2 = Qwen3Model(TINY)
    load_hf_state_dict(dst2, hf_style)
    assert dst2.lm_head.weight is not dst2.embed_tokens.weight
    torch.testing.assert_close(dst2.lm_head.weight, hf_style["lm_head.weight"])
