"""Chart helpers used by the notebooks, so the lessons show data rather than matplotlib boilerplate.

Each function draws one figure with the track's palette; call it from a notebook cell and it renders inline.
"""

from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
import torch

SLATE = "#7799B6"
GOLD = "#ECAC45"
TEAL = "#395761"
YELLOW = "#CEEB13"
SAGE = "#99AE88"
GRAY = "#E4E2E0"
BLACK = "#000000"


def _tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)


def plot_benchmark_composition(problems: list[dict]) -> None:
    """Problems per subject and per difficulty level for a MATH-style dataset."""
    subjects = Counter(r["subject"] for r in problems)
    levels = Counter(r["level"] for r in problems)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [3, 2]})
    names, counts = zip(*sorted(subjects.items(), key=lambda kv: kv[1]))
    ax1.barh(names, counts, color=SLATE, height=0.6)
    for y, c in enumerate(counts):
        ax1.text(c + 1, y, str(c), va="center", fontsize=9)
    ax1.set_title("problems per subject", loc="left")
    _tidy(ax1)

    lv = sorted(levels)
    ax2.bar([str(k) for k in lv], [levels[k] for k in lv], color=GOLD, width=0.6, edgecolor=BLACK)
    for x, k in enumerate(lv):
        ax2.text(x, levels[k] + 2, str(levels[k]), ha="center", fontsize=9)
    ax2.set_title("problems per difficulty level", loc="left")
    ax2.set_xlabel("level (5 is hardest)")
    _tidy(ax2)
    plt.tight_layout()
    plt.show()


def plot_next_token_probs(probs: torch.Tensor, tokenizer, prompt: str, k: int = 8) -> None:
    """The k most likely next tokens after `prompt`, from one row of softmax probabilities."""
    top = torch.topk(probs.detach().float().flatten(), k=k)
    labels = [repr(tokenizer.decode([i])) for i in top.indices.tolist()]
    values = top.values.tolist()

    fig, ax = plt.subplots(figsize=(8, 3.6))
    bars = ax.barh(range(k)[::-1], values, color=TEAL, height=0.6)
    ax.set_yticks(range(k)[::-1], labels, family="monospace")
    for bar, v in zip(bars, values):
        ax.text(v + 0.004, bar.get_y() + bar.get_height() / 2, f"{v:.1%}", va="center", fontsize=9)
    ax.set_xlim(0, max(values) * 1.25)
    ax.set_xlabel("probability of being the next token")
    ax.set_title(f"what follows {prompt!r}?", loc="left")
    _tidy(ax)
    plt.tight_layout()
    plt.show()


def plot_accuracy_comparison(results: dict, title: str) -> None:
    """Accuracy per named run, annotated with average tokens per problem. Values are evaluate_math summaries."""
    names = list(results)
    accs = [results[n]["accuracy"] for n in names]
    toks = [results[n]["avg_tokens"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 0.9 + 0.6 * len(names)))
    bars = ax.barh(names[::-1], accs[::-1], color=SLATE, height=0.6)
    for bar, a, t in zip(bars, accs[::-1], toks[::-1]):
        ax.text(
            a + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{a:.0%}   ({t:.0f} tokens avg)",
            va="center",
            fontsize=10,
        )
    ax.set_xlim(0, 1)
    ax.set_xlabel("accuracy")
    ax.set_title(title, loc="left")
    _tidy(ax)
    plt.tight_layout()
    plt.show()


def plot_template_comparison(template_results: dict, title: str) -> None:
    """Accuracy per prompt template. Same chart as plot_accuracy_comparison, kept under its original name."""
    plot_accuracy_comparison(template_results, title)


def plot_temperature_effect(logits: torch.Tensor, tokenizer, temperatures=(0.3, 1.0, 2.0), k: int = 6) -> None:
    """The top-k next-token probabilities after dividing one row of logits by each temperature."""
    logits = logits.detach().float().flatten()
    order = torch.topk(logits, k).indices  # same k tokens in every panel, ranked by the raw logits
    labels = [repr(tokenizer.decode([i])) for i in order.tolist()]

    fig, axes = plt.subplots(1, len(temperatures), figsize=(4.0 * len(temperatures), 3.4), sharey=True)
    for ax, t in zip(axes, temperatures):
        probs = torch.softmax(logits / t, dim=-1)[order].tolist()
        ax.bar(range(k), probs, color=GOLD if t == 1.0 else SLATE, edgecolor=BLACK, width=0.7)
        for x, p in enumerate(probs):
            ax.text(x, p + 0.01, f"{p:.0%}", ha="center", fontsize=8)
        ax.set_xticks(range(k), labels, family="monospace", fontsize=7.5)
        ax.set_title(f"temperature {t}", loc="left")
        ax.set_ylim(0, 1.05)
        _tidy(ax)
    axes[0].set_ylabel("probability")
    plt.tight_layout()
    plt.show()


def plot_top_p(probs: torch.Tensor, top_p: float) -> None:
    """A probability vector sorted high to low, its cumulative sum, and where a top-p cutoff falls."""
    sorted_probs, _ = torch.sort(probs.detach().float().flatten(), descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=0)
    n_kept = int((cumulative - sorted_probs < top_p).sum())
    x = range(len(sorted_probs))

    fig, ax = plt.subplots(figsize=(8, 3.4))
    colors = [YELLOW if i < n_kept else GRAY for i in x]
    ax.bar(x, sorted_probs.tolist(), color=colors, edgecolor=BLACK, width=0.7, label="probability (sorted)")
    ax.step(x, cumulative.tolist(), where="mid", color=TEAL, lw=2, label="cumulative sum")
    ax.axhline(top_p, color=BLACK, ls="--", lw=1, label=f"top_p = {top_p}")
    ax.axvline(n_kept - 0.5, color=TEAL, ls=":", lw=1.5)
    ax.text(n_kept - 0.4, 1.0, f"keep {n_kept} tokens", va="top", fontsize=9, color=TEAL)
    ax.set_xlabel("token rank")
    ax.set_ylabel("probability")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="center right", frameon=False)
    _tidy(ax)
    plt.tight_layout()
    plt.show()


def plot_brevity_bonus(brevity_scale: float = 500.0, max_len: int = 2500) -> None:
    """The length term of the rule-based score: a bonus that decays with the number of characters."""
    lengths = torch.arange(1, max_len)
    bonus = 1.5 * torch.exp(-lengths / brevity_scale)

    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.plot(lengths, bonus, color=TEAL, lw=2)
    ax.axvline(brevity_scale, color=GOLD, ls="--", lw=1.5)
    ax.text(brevity_scale + 30, 1.4, f"brevity_scale = {brevity_scale:.0f}", fontsize=9, color=BLACK)
    ax.set_xlabel("response length (characters)")
    ax.set_ylabel("bonus")
    ax.set_title("shorter responses earn a larger bonus", loc="left")
    _tidy(ax)
    plt.tight_layout()
    plt.show()


def plot_logits_to_logprobs(logits: torch.Tensor) -> None:
    """One row of logits next to its softmax and its log-softmax, so the three scales can be compared."""
    logits = logits.detach().float().flatten()
    probs = torch.softmax(logits, dim=-1)
    logprobs = torch.log_softmax(logits, dim=-1)
    x = range(len(logits))

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, values, title, color in zip(
        axes,
        (logits, probs, logprobs),
        ("logits", "softmax: probabilities", "log_softmax: log-probabilities"),
        (SLATE, GOLD, TEAL),
    ):
        ax.bar(x, values.tolist(), color=color, edgecolor=BLACK, width=0.7)
        ax.axhline(0, color=BLACK, lw=0.8)
        ax.set_xticks(list(x))
        ax.set_xlabel("token index")
        ax.set_title(title, loc="left")
        _tidy(ax)
    axes[1].set_ylim(0, 1)
    plt.tight_layout()
    plt.show()


def plot_token_logprobs(tokens: list[str], logprobs: torch.Tensor, title: str) -> None:
    """The log-probability of each token in a text, one bar per token, the lowest one highlighted."""
    values = logprobs.detach().float().cpu().tolist()
    worst = min(range(len(values)), key=lambda i: values[i])

    fig, ax = plt.subplots(figsize=(8, 3.2))
    colors = [GOLD if i == worst else SLATE for i in range(len(values))]
    ax.bar(range(len(values)), values, color=colors, edgecolor=BLACK, width=0.7)
    ax.set_xticks(range(len(values)), [repr(t) for t in tokens], family="monospace", fontsize=8)
    ax.axhline(0, color=BLACK, lw=0.8)
    for i, v in enumerate(values):
        ax.text(i, v - 0.3, f"{v:.1f}", ha="center", va="top", fontsize=8)
    ax.set_ylim(min(values) - 2.5, 0.5)
    ax.set_ylabel("log-probability")
    ax.set_title(title, loc="left")
    _tidy(ax)
    plt.tight_layout()
    plt.show()


def plot_bf16_resolution(weight: float = 0.02, update: float = 1e-5) -> None:
    """A weight on the bfloat16 number line next to the update a training step wants to apply to it."""
    spacing = 2.0 ** (torch.floor(torch.log2(torch.tensor(weight))) - 7)  # 8 bits of mantissa, so 2^(exponent - 7)
    spacing = spacing.item()
    grid = torch.tensor(weight, dtype=torch.bfloat16).float().item()
    ticks = [grid + k * spacing for k in range(-2, 3)]

    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.axhline(0, color=BLACK, lw=1)
    for t in ticks:
        ax.plot([t, t], [-0.25, 0.25], color=SLATE, lw=2)
        ax.text(t, 0.42, f"{t:.6f}", ha="center", fontsize=8, family="monospace")
    ax.plot(grid, 0, "o", color=TEAL, markersize=8, zorder=3)
    ax.annotate(
        "",
        xy=(grid + update, -0.5),
        xytext=(grid, -0.5),
        arrowprops={"arrowstyle": "-|>", "color": GOLD, "lw": 3, "shrinkA": 0, "shrinkB": 0},
    )
    ax.plot([grid, grid], [-0.3, -0.6], color=GOLD, lw=1, ls=":")
    ax.text(grid + 0.12 * spacing, -0.5, f"the update a step wants to add: +{update:g}", va="center", fontsize=9, color=BLACK)
    ax.annotate(
        "",
        xy=(ticks[1], -0.5),
        xytext=(ticks[0], -0.5),
        arrowprops={"arrowstyle": "<->", "color": SLATE, "lw": 1.5, "shrinkA": 0, "shrinkB": 0},
    )
    ax.text((ticks[0] + ticks[1]) / 2, -0.78, f"gap between bfloat16 values: {spacing:.1e}", ha="center", fontsize=9, color=TEAL)
    ax.text(ticks[0], -1.05, "the update is shorter than half a gap, so rounding puts the weight back where it was", fontsize=9, color=BLACK)
    ax.set_xlim(ticks[0] - 0.4 * spacing, ticks[-1] + 0.4 * spacing)
    ax.set_ylim(-1.25, 0.7)
    ax.axis("off")
    ax.set_title(f"a weight near {weight} stored in bfloat16 (blue ticks are the representable values)", loc="left", fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_rewards_and_advantages(rewards: list[float], labels: list[str] | None = None) -> None:
    """The rewards of one group of rollouts beside the advantages computed from them."""
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-4)
    labels = labels or [f"rollout {i + 1}" for i in range(len(rewards))]
    x = range(len(rewards))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2))
    ax1.bar(x, rewards, color=[GOLD if r > 0 else GRAY for r in rewards], edgecolor=BLACK, width=0.6)
    ax1.axhline(rewards_t.mean().item(), color=TEAL, ls="--", lw=1.5)
    ax1.text(len(rewards) - 0.5, rewards_t.mean().item() + 0.03, f"mean {rewards_t.mean():.2f}", ha="right", fontsize=9, color=TEAL)
    ax1.set_xticks(list(x), labels, fontsize=9)
    ax1.set_ylim(0, 1.15)
    ax1.set_title("rewards", loc="left")
    _tidy(ax1)

    ax2.bar(x, advantages.tolist(), color=[YELLOW if a > 0 else GRAY for a in advantages], edgecolor=BLACK, width=0.6)
    ax2.axhline(0, color=BLACK, lw=1)
    for i, a in enumerate(advantages.tolist()):
        ax2.text(i, a + (0.06 if a >= 0 else -0.06), f"{a:+.2f}", ha="center", va="bottom" if a >= 0 else "top", fontsize=9)
    ax2.set_xticks(list(x), labels, fontsize=9)
    ax2.set_ylim(min(-1.2, advantages.min().item() - 0.3), max(1.2, advantages.max().item() + 0.3))
    ax2.set_title("advantages", loc="left")
    _tidy(ax2)
    plt.tight_layout()
    plt.show()


def plot_advantage_patterns(n_rollouts: int = 4) -> None:
    """The advantage of a correct and of a wrong rollout as the number of correct rollouts in the group changes."""
    counts = list(range(n_rollouts + 1))
    correct, wrong = [], []
    for k in counts:
        rewards = torch.tensor([1.0] * k + [0.0] * (n_rollouts - k))
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
        correct.append(adv[0].item() if k > 0 else 0.0)
        wrong.append(adv[-1].item() if k < n_rollouts else 0.0)

    fig, ax = plt.subplots(figsize=(8, 3.4))
    w = 0.36
    ax.bar([c - w / 2 for c in counts], correct, width=w, color=YELLOW, edgecolor=BLACK, label="a correct rollout")
    ax.bar([c + w / 2 for c in counts], wrong, width=w, color=GRAY, edgecolor=BLACK, label="a wrong rollout")
    for c, a in zip(counts, correct):
        if a:
            ax.text(c - w / 2, a + 0.05, f"{a:+.2f}", ha="center", fontsize=8)
    for c, a in zip(counts, wrong):
        if a:
            ax.text(c + w / 2, a - 0.05, f"{a:+.2f}", ha="center", va="top", fontsize=8)
    ax.axhline(0, color=BLACK, lw=1)
    ax.set_xticks(counts, [f"{k} of {n_rollouts}" for k in counts])
    ax.set_xlabel("correct rollouts in the group")
    ax.set_ylabel("advantage")
    ax.set_ylim(-2.1, 2.1)
    ax.set_title("a rare success earns a large advantage; a group that agrees earns none", loc="left", fontsize=10)
    ax.legend(loc="upper right", frameon=False)
    _tidy(ax)
    plt.tight_layout()
    plt.show()


def plot_sequence_logprob(tokens: list[str], logprobs: torch.Tensor, prompt_len: int, title: str, show_prompt: int = 8) -> None:
    """Per-token log-probabilities of a prompt plus response; prompt tokens are grayed out, response tokens summed.

    `tokens` are the decoded tokens from position 1 onward and `logprobs` their log-probabilities,
    so entry i belongs to token i + 1 of the sequence. The response starts at token `prompt_len`.
    """
    values = logprobs.detach().float().cpu().tolist()
    first = prompt_len - 1  # the first entry that belongs to a response token
    lo = max(0, first - show_prompt)
    shown_tokens, shown_values = tokens[lo:], values[lo:]
    colors = [GRAY if lo + i < first else TEAL for i in range(len(shown_values))]
    response_sum = sum(values[first:])

    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.bar(range(len(shown_values)), shown_values, color=colors, edgecolor=BLACK, width=0.7)
    ax.set_xticks(range(len(shown_values)), [repr(t) for t in shown_tokens], family="monospace", fontsize=7.5, rotation=45, ha="right")
    ax.axhline(0, color=BLACK, lw=0.8)
    ax.axvline(first - lo - 0.5, color=GOLD, ls="--", lw=1.5)
    ax.text(first - lo - 0.4, min(shown_values) - 0.6, "response starts", fontsize=9, color=BLACK)
    ax.text(len(shown_values) - 0.5, min(shown_values) - 0.6, f"sum over the response: {response_sum:.2f}", ha="right", fontsize=9, color=TEAL)
    ax.set_ylim(min(shown_values) - 1.6, 0.5)
    ax.set_ylabel("log-probability")
    ax.set_title(title, loc="left")
    _tidy(ax)
    plt.tight_layout()
    plt.show()


def plot_logprob_shift(labels: list[str], before: list[float], after: list[float], advantages: list[float], title: str) -> None:
    """The change in each rollout's sequence log-probability after an update, colored by the sign of its advantage."""
    deltas = [a - b for a, b in zip(after, before)]
    fig, ax = plt.subplots(figsize=(8.5, 0.9 + 0.7 * len(labels)))
    ys = list(range(len(labels)))[::-1]
    colors = [GOLD if adv > 0 else SLATE for adv in advantages]
    ax.barh(ys, deltas, color=colors, edgecolor=BLACK, height=0.55)
    ax.axvline(0, color=BLACK, lw=1)
    span = max(abs(d) for d in deltas) or 1.0
    for y, d, b, a in zip(ys, deltas, before, after):
        ax.text(d + (0.03 if d >= 0 else -0.03) * span, y, f"{b:.2f} -> {a:.2f}", va="center", ha="left" if d >= 0 else "right", fontsize=9, family="monospace")
    ax.set_yticks(ys, [f"{lb}  (advantage {adv:+.2f})" for lb, adv in zip(labels, advantages)], fontsize=9)
    ax.set_xlim(-span * 2.2, span * 2.2)
    ax.set_xlabel("change in sequence log-probability (right: more likely, left: less likely)")
    ax.set_title(title, loc="left", fontsize=10)
    _tidy(ax)
    plt.tight_layout()
    plt.show()


def plot_training_metrics(history: list[dict], window: int = 10, title: str = "GRPO training") -> None:
    """Reward, response length, and loss per training step, with a moving average over `window` steps."""
    steps = [h["step"] for h in history]

    def smooth(values):
        out = []
        for i in range(len(values)):
            chunk = values[max(0, i - window + 1) : i + 1]
            out.append(sum(chunk) / len(chunk))
        return out

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, key, color, label in zip(
        axes,
        ("reward", "length", "loss"),
        (GOLD, SLATE, TEAL),
        ("average reward per step", "average response length (tokens)", "loss"),
    ):
        values = [h[key] for h in history]
        ax.plot(steps, values, color=color, lw=1, alpha=0.55)
        ax.plot(steps, smooth(values), color=BLACK, lw=2, label=f"moving average, {window} steps")
        ax.set_xlabel("step")
        ax.set_title(label, loc="left", fontsize=10)
        _tidy(ax)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(loc="lower right", frameon=False, fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_accuracy_vs_samples(n_samples: list[int], accuracies: list[float], baseline: float, title: str) -> None:
    """Majority-vote accuracy as the number of sampled chains grows, against a greedy baseline."""
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(n_samples, accuracies, marker="o", color=TEAL, lw=2, markersize=8, label="majority vote")
    for n, a in zip(n_samples, accuracies):
        ax.text(n, a + 0.02, f"{a:.0%}", ha="center", fontsize=9)
    ax.axhline(baseline, color=GOLD, ls="--", lw=2, label=f"greedy, one chain ({baseline:.0%})")
    ax.set_xticks(n_samples)
    ax.set_xlabel("sampled chains per problem")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, min(1.0, max(*accuracies, baseline) + 0.25))
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right", frameon=False)
    _tidy(ax)
    plt.tight_layout()
    plt.show()
