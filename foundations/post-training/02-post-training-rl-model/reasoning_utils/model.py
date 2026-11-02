"""A decoder-only transformer (the Qwen3 architecture) in plain PyTorch.

This file is plumbing, not the lesson. The notebooks in this track teach
reasoning methods that sit on top of a pretrained language model, so the
model itself is treated as a given: you load it, you call it, you get
logits back. Read this file if you are curious how the forward pass works,
but nothing in the lessons depends on understanding it.

Module and parameter names mirror the checkpoint layout published on the
Hugging Face Hub, so a checkpoint can be loaded with a one-line key rename.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Qwen3Config:
    vocab_size: int = 151_936
    context_length: int = 40_960
    emb_dim: int = 1024
    n_layers: int = 28
    n_heads: int = 16
    n_kv_groups: int = 8
    head_dim: int = 128
    hidden_dim: int = 3072
    rope_base: float = 1_000_000.0
    rms_eps: float = 1e-6
    dtype: torch.dtype = torch.bfloat16


QWEN3_0_6B = Qwen3Config()

HF_REPOS = {
    "base": "Qwen/Qwen3-0.6B-Base",  # pretrained only, no post-training
    "reasoning": "Qwen/Qwen3-0.6B",  # post-trained, emits <think> blocks
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(in_dtype)


def rope_tables(head_dim: int, base: float, context_length: int):
    """Precompute the cos and sin tables for rotary position embeddings."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(context_length).float()
    angles = positions[:, None] * inv_freq[None, :]  # (ctx, head_dim/2)
    angles = torch.cat([angles, angles], dim=-1)  # (ctx, head_dim)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0):
    """Rotate q or k by their absolute position. x: (batch, heads, seq, head_dim)."""
    seq_len, head_dim = x.shape[-2], x.shape[-1]
    half = head_dim // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos[offset : offset + seq_len][None, None]
    sin = sin[offset : offset + seq_len][None, None]
    rotated = torch.cat([-x2, x1], dim=-1)
    return (x * cos + rotated * sin).to(x.dtype)


class Attention(nn.Module):
    """Grouped-query attention with q/k normalization and an optional KV cache."""

    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_groups = cfg.n_kv_groups
        self.group_size = cfg.n_heads // cfg.n_kv_groups
        self.head_dim = cfg.head_dim
        d_q = cfg.n_heads * cfg.head_dim
        d_kv = cfg.n_kv_groups * cfg.head_dim
        self.q_proj = nn.Linear(cfg.emb_dim, d_q, bias=False, dtype=cfg.dtype)
        self.k_proj = nn.Linear(cfg.emb_dim, d_kv, bias=False, dtype=cfg.dtype)
        self.v_proj = nn.Linear(cfg.emb_dim, d_kv, bias=False, dtype=cfg.dtype)
        self.o_proj = nn.Linear(d_q, cfg.emb_dim, bias=False, dtype=cfg.dtype)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_eps)

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_groups, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_groups, self.head_dim).transpose(1, 2)

        q = apply_rope(self.q_norm(q), cos, sin, offset=start_pos)
        k = apply_rope(self.k_norm(k), cos, sin, offset=start_pos)

        if cache is not None:
            prev_k, prev_v = cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)
        new_cache = (k, v)

        # Each KV head serves group_size query heads.
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        scores = (q @ k.transpose(2, 3)) / (self.head_dim**0.5)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).reshape(b, t, -1)
        return self.o_proj(out), new_cache


class MLP(nn.Module):
    """SwiGLU feed-forward block."""

    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.emb_dim, cfg.hidden_dim, bias=False, dtype=cfg.dtype)
        self.up_proj = nn.Linear(cfg.emb_dim, cfg.hidden_dim, bias=False, dtype=cfg.dtype)
        self.down_proj = nn.Linear(cfg.hidden_dim, cfg.emb_dim, bias=False, dtype=cfg.dtype)

    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.emb_dim, cfg.rms_eps)
        self.post_attention_layernorm = RMSNorm(cfg.emb_dim, cfg.rms_eps)

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        h, new_cache = self.self_attn(self.input_layernorm(x), mask, cos, sin, start_pos, cache)
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_cache


class KVCache:
    """Per-layer storage for keys and values already computed during generation."""

    def __init__(self, n_layers: int):
        self.layers = [None] * n_layers

    def get(self, i):
        return self.layers[i]

    def update(self, i, value):
        self.layers[i] = value

    def reset(self):
        self.layers = [None] * len(self.layers)


class Qwen3Model(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.emb_dim, dtype=cfg.dtype)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.emb_dim, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False, dtype=cfg.dtype)

        cos, sin = rope_tables(cfg.head_dim, cfg.rope_base, cfg.context_length)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.current_pos = 0  # how many tokens the KV cache already holds

    def reset_kv_cache(self):
        self.current_pos = 0

    def forward(self, token_ids: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        """token_ids: (batch, seq) -> logits: (batch, seq, vocab)."""
        x = self.embed_tokens(token_ids)
        t = x.shape[1]
        start = self.current_pos if cache is not None else 0
        end = start + t
        if end > self.cfg.context_length:
            raise ValueError(f"sequence of {end} tokens exceeds context length {self.cfg.context_length}")

        # Causal mask: True where attention is NOT allowed.
        if cache is not None:
            self.current_pos = end
            mask = torch.triu(torch.ones(end, end, dtype=torch.bool, device=x.device), diagonal=1)
            mask = mask[start:end, :end]
        else:
            mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
        mask = mask[None, None]  # broadcast over batch and heads

        for i, layer in enumerate(self.layers):
            layer_cache = cache.get(i) if cache is not None else None
            x, new_cache = layer(x, mask, self.cos, self.sin, start_pos=start, cache=layer_cache)
            if cache is not None:
                cache.update(i, new_cache)

        # Match the head's dtype rather than the config's, so a model cast to float32 for training still runs.
        return self.lm_head(self.norm(x).to(self.lm_head.weight.dtype))


def load_hf_state_dict(model: Qwen3Model, state_dict: dict) -> None:
    """Load a Hugging Face Qwen3 checkpoint into Qwen3Model.

    The checkpoint keys are `model.<name>` plus an optional `lm_head.weight`.
    Our module names match `<name>`, so only the prefix needs to go. The 0.6B
    checkpoints tie the output head to the token embedding, which is why
    `lm_head.weight` can be absent.
    """
    renamed = {k.removeprefix("model."): v for k, v in state_dict.items()}
    if "lm_head.weight" not in renamed:
        renamed["lm_head.weight"] = renamed["embed_tokens.weight"]
    missing, unexpected = model.load_state_dict(renamed, strict=False)
    missing = [k for k in missing if not k.endswith((".cos", ".sin"))]
    if missing or unexpected:
        raise ValueError(f"checkpoint mismatch. missing={missing} unexpected={unexpected}")
    if "lm_head.weight" not in state_dict:
        model.lm_head.weight = model.embed_tokens.weight  # keep them tied


def load_qwen3(kind: str = "base", device=None, compile: bool = False, checkpoint: str | None = None):
    """Download (once) and load Qwen3-0.6B plus its tokenizer.

    kind: "base" for the pretrained model, "reasoning" for the post-trained one.
    checkpoint: an optional state dict saved by the GRPO notebook, loaded on top
    of the pretrained weights. Returns (model, tokenizer). The model is in eval
    mode on `device`.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    from .tokenizer import Qwen3Tokenizer

    if kind not in HF_REPOS:
        raise ValueError(f"kind must be one of {list(HF_REPOS)}, got {kind!r}")
    repo = HF_REPOS[kind]
    weights_path = hf_hub_download(repo, "model.safetensors")
    tokenizer_path = hf_hub_download(repo, "tokenizer.json")

    model = Qwen3Model(QWEN3_0_6B)
    load_hf_state_dict(model, load_file(weights_path))
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)  # a plain state dict, no pickled code
        model.load_state_dict(state)  # copies in place, so the tied output head stays tied
    model.eval()
    if device is not None:
        model.to(device)
    if compile:
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        # Training runs the model in several modes (cached sampling, scoring with and without gradients, a
        # float32 copy), and each needs its own compiled graph; the default limit of 8 falls back to eager.
        torch._dynamo.config.recompile_limit = 32
        model = torch.compile(model)

    tokenizer = Qwen3Tokenizer(tokenizer_path, chat_template=(kind == "reasoning"))
    return model, tokenizer
