"""Group relative policy optimization (GRPO) with verifiable rewards, one function per stage.

Built in the GRPO notebook. A training step samples a group of responses to
one problem, rewards each with the math verifier, turns the rewards into
group-relative advantages, scores every response with its sequence
log-probability under the model, and combines the two into the policy
gradient loss. `train_grpo` wraps that step in the usual PyTorch loop.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from .evaluate import MATH_PROMPT, render_math_prompt
from .generate import scale_by_temperature, top_p_filter
from .model import KVCache
from .verify import grade, last_boxed


@torch.no_grad()
def sample_response(model, tokenizer, prompt, device, max_new_tokens=512, temperature=0.8, top_p=0.9):
    """One sampled response. Returns the prompt plus response token ids, the prompt length, and the text."""
    prompt_ids = torch.tensor(tokenizer.encode(prompt), device=device)
    cache = KVCache(n_layers=len(model.layers))
    model.reset_kv_cache()
    logits = model(prompt_ids.unsqueeze(0), cache=cache)[:, -1]  # prefill, as in the decoding loop

    generated = []
    for _ in range(max_new_tokens):
        probs = torch.softmax(scale_by_temperature(logits.float(), temperature), dim=-1)
        probs = top_p_filter(probs, top_p)
        next_token = torch.multinomial(probs, num_samples=1)
        generated.append(next_token.item())  # the end token is kept too: the model has to learn when to stop
        if generated[-1] == tokenizer.eos_token_id:
            break
        logits = model(next_token, cache=cache)[:, -1]

    token_ids = torch.cat([prompt_ids, torch.tensor(generated, device=device)])
    text = tokenizer.decode(generated).removesuffix(tokenizer.eos_token)
    return token_ids, len(prompt_ids), text


def correctness_reward(text, answer):
    """1.0 for a boxed final answer that the verifier accepts, otherwise 0.0."""
    boxed = last_boxed(text)
    if boxed is None:
        return 0.0  # no box, no reward: the format is part of the task
    return float(grade(boxed, answer))


def group_advantages(rewards):
    """Each reward minus the group mean, divided by the group standard deviation."""
    rewards_t = torch.tensor(rewards)
    return (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-4)


def sequence_logprob(model, token_ids, prompt_len):
    """The summed log-probability of the response tokens given the prompt, with gradients attached."""
    logits = model(token_ids.unsqueeze(0)).squeeze(0).float()
    logprobs = torch.log_softmax(logits, dim=-1)
    positions = torch.arange(prompt_len - 1, len(token_ids) - 1, device=token_ids.device)
    next_ids = token_ids[prompt_len:]
    return logprobs[positions, next_ids].sum()


def grpo_loss(advantages, logps):
    """The policy gradient loss: minus the advantage-weighted sequence log-probabilities, averaged over the group."""
    return -(advantages * logps).mean()


def grpo_step(
    model, tokenizer, example, device, n_rollouts=4, max_new_tokens=512, temperature=0.8, top_p=0.9
):
    """Sample, reward, compare, and score one problem, and build the loss. The caller runs backward."""
    prompt = render_math_prompt(example["problem"], MATH_PROMPT)
    rollouts, rewards = [], []
    for _ in range(n_rollouts):
        token_ids, prompt_len, text = sample_response(
            model, tokenizer, prompt, device, max_new_tokens, temperature, top_p
        )
        rollouts.append((token_ids, prompt_len, text))
        rewards.append(correctness_reward(text, example["answer"]))
    advantages = group_advantages(rewards).to(device)

    out = {
        "texts": [text for _, _, text in rollouts],
        "lengths": [len(token_ids) - prompt_len for token_ids, prompt_len, _ in rollouts],
        "rewards": rewards,
        "advantages": advantages.tolist(),
        "loss": None,
    }
    if all(r == rewards[0] for r in rewards):
        return out  # every rollout got the same reward: nothing to compare, nothing to learn here

    logps = torch.stack([sequence_logprob(model, token_ids, prompt_len) for token_ids, prompt_len, _ in rollouts])
    out["loss"] = grpo_loss(advantages, logps)
    return out


def train_grpo(
    model,
    tokenizer,
    train_data,
    device,
    steps=50,
    n_rollouts=4,
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.9,
    lr=1e-6,
    log_path=None,
    checkpoint_every=0,
    checkpoint_dir="checkpoints",
    sample_every=10,
):
    """The GRPO training loop. Returns one metrics dict per step and optionally logs them as JSON lines."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text("", encoding="utf-8")

    start = time.time()
    for step in range(1, steps + 1):
        example = train_data[(step - 1) % len(train_data)]
        optimizer.zero_grad()
        out = grpo_step(model, tokenizer, example, device, n_rollouts, max_new_tokens, temperature, top_p)
        if out["loss"] is not None:
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # one outsized gradient must not derail the run
            optimizer.step()

        metrics = {
            "step": step,
            "reward": sum(out["rewards"]) / n_rollouts,
            "length": sum(out["lengths"]) / n_rollouts,
            "loss": out["loss"].item() if out["loss"] is not None else 0.0,
            "updated": out["loss"] is not None,
            "seconds": time.time() - start,
        }
        history.append(metrics)
        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics) + "\n")
        print(
            f"step {step}/{steps}",
            f"reward {metrics['reward']:.2f}",
            f"length {metrics['length']:.0f}",
            f"loss {metrics['loss']:.3f}",
        )
        if sample_every and step % sample_every == 0:
            print("sample:", repr(out["texts"][0][:160]))
        if checkpoint_every and step % checkpoint_every == 0:
            save_checkpoint(model, Path(checkpoint_dir) / f"grpo-step{step:04d}.pt")
    return history


def save_checkpoint(model, path):
    """Write the model weights to `path`; `load_qwen3(checkpoint=path)` loads them back."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    module = getattr(model, "_orig_mod", model)  # a compiled model wraps the real one; save the real one's keys
    torch.save(module.state_dict(), path)
    print("saved", path)
    return path
