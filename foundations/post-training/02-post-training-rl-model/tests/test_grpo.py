"""The GRPO stages, offline on a tiny random model: sampling, rewards, advantages, log-probs, the loss, one update."""

import math

import torch

from reasoning_utils.grpo import (
    correctness_reward,
    group_advantages,
    grpo_loss,
    grpo_step,
    sample_response,
    save_checkpoint,
    sequence_logprob,
    train_grpo,
)
from reasoning_utils.model import Qwen3Config, Qwen3Model
from reasoning_utils.scoring import token_logprobs

TINY = Qwen3Config(
    vocab_size=50,
    context_length=256,
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
    eos_token = "<eos>"

    def encode(self, text, chat_template=None):
        return [1 + (ord(c) % (TINY.vocab_size - 1)) for c in text]

    def decode(self, ids):
        return "".join("<eos>" if i == 0 else chr(97 + (i % 26)) for i in ids)


def test_sample_response_returns_prompt_plus_response_ids():
    model = Qwen3Model(TINY).eval()
    tok = FakeTokenizer()
    torch.manual_seed(0)
    ids, prompt_len, text = sample_response(model, tok, "abc", "cpu", max_new_tokens=6)
    assert prompt_len == 3
    assert ids.shape[0] <= prompt_len + 6
    assert torch.equal(ids[:3], torch.tensor(tok.encode("abc")))
    assert not text.endswith(tok.eos_token)  # the end token stays in the ids but leaves the text
    assert not ids.requires_grad


def test_correctness_reward_needs_a_box():
    assert correctness_reward(r"so \boxed{7}", "7") == 1.0
    assert correctness_reward(r"so \boxed{8}", "7") == 0.0
    assert correctness_reward("the answer is 7", "7") == 0.0
    assert correctness_reward(r"\boxed{\frac{7}{2}}", "3.5") == 1.0


def test_group_advantages_center_and_scale():
    adv = group_advantages([1.0, 1.0, 0.0, 0.0])
    assert torch.allclose(adv, torch.tensor([0.866, 0.866, -0.866, -0.866]), atol=1e-3)
    assert torch.all(group_advantages([1.0, 1.0, 1.0]) == 0)  # a group that agrees carries no signal
    assert math.isclose(group_advantages([1.0, 0.0, 0.0, 0.0])[0].item(), 1.5, abs_tol=1e-3)


def test_sequence_logprob_sums_the_response_tokens_with_grad():
    model = Qwen3Model(TINY).eval()
    tok = FakeTokenizer()
    ids = torch.tensor(tok.encode("abcxyz"))
    total = sequence_logprob(model, ids, prompt_len=3)
    assert total.requires_grad
    reference = token_logprobs(model, tok, "abcxyz", "cpu")[-3:].sum()
    assert torch.allclose(total.detach(), reference, atol=1e-5)


def test_grpo_loss_sign_and_average():
    adv = torch.tensor([1.0, -1.0])
    logps = torch.tensor([-2.0, -4.0])
    assert math.isclose(grpo_loss(adv, logps).item(), -((1.0 * -2.0) + (-1.0 * -4.0)) / 2)


def test_grpo_step_skips_a_unanimous_group_and_moves_logprobs_the_right_way():
    model = Qwen3Model(TINY)
    tok = FakeTokenizer()
    torch.manual_seed(0)
    out = grpo_step(model, tok, {"problem": "ab", "answer": "7"}, "cpu", n_rollouts=3, max_new_tokens=4)
    assert out["loss"] is None and len(out["rewards"]) == 3  # a random model never boxes 7
    assert len(out["texts"]) == 3 and all(n <= 4 for n in out["lengths"])

    # a hand-made group: one response favored, one disfavored; an SGD step must raise the first and lower the second
    ids_a, ids_b = torch.tensor(tok.encode("abzz")), torch.tensor(tok.encode("abqq"))
    adv = group_advantages([1.0, 0.0])
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    before = torch.stack([sequence_logprob(model, ids_a, 2), sequence_logprob(model, ids_b, 2)]).detach()
    opt.zero_grad()
    grpo_loss(adv, torch.stack([sequence_logprob(model, ids_a, 2), sequence_logprob(model, ids_b, 2)])).backward()
    opt.step()
    after = torch.stack([sequence_logprob(model, ids_a, 2), sequence_logprob(model, ids_b, 2)]).detach()
    assert after[0] > before[0] and after[1] < before[1]


def test_train_grpo_logs_every_step_and_saves_a_checkpoint(tmp_path):
    model = Qwen3Model(TINY)
    model.lm_head.weight = model.embed_tokens.weight  # tied, as load_qwen3 ties the real model
    tok = FakeTokenizer()
    torch.manual_seed(0)
    data = [{"problem": "ab", "answer": "7"}, {"problem": "cd", "answer": "8"}]
    log = tmp_path / "train.jsonl"
    history = train_grpo(
        model, tok, data, "cpu", steps=3, n_rollouts=2, max_new_tokens=3, log_path=log, checkpoint_every=3, checkpoint_dir=tmp_path, sample_every=0
    )
    assert [h["step"] for h in history] == [1, 2, 3]
    assert len(log.read_text().splitlines()) == 3
    assert (tmp_path / "grpo-step0003.pt").exists()
    path = save_checkpoint(model, tmp_path / "final.pt")
    fresh = Qwen3Model(TINY)
    fresh.lm_head.weight = fresh.embed_tokens.weight  # tied, as load_qwen3 ties them
    fresh.load_state_dict(torch.load(path, weights_only=True))
    assert torch.equal(fresh.embed_tokens.weight, model.embed_tokens.weight)
    assert fresh.lm_head.weight is fresh.embed_tokens.weight  # load_state_dict copies in place, the tie survives
