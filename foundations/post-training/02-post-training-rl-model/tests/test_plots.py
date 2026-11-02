"""Smoke tests: the chart helpers draw without error on synthetic data (headless backend)."""

import matplotlib

matplotlib.use("Agg")

import torch  # noqa: E402

from reasoning_utils import plots  # noqa: E402


class FakeTokenizer:
    def decode(self, ids):
        return f"tok{ids[0]}"


def test_plot_benchmark_composition():
    problems = [
        {"subject": "Algebra", "level": 1},
        {"subject": "Algebra", "level": 3},
        {"subject": "Geometry", "level": 3},
    ]
    plots.plot_benchmark_composition(problems)


def test_plot_accuracy_comparison_and_template_alias():
    results = {"x": {"accuracy": 0.2, "avg_tokens": 90}, "y": {"accuracy": 0.3, "avg_tokens": 120}}
    plots.plot_accuracy_comparison(results, "t")
    plots.plot_template_comparison(results, "t")


def test_plot_next_token_probs():
    plots.plot_next_token_probs(torch.softmax(torch.randn(40), dim=-1), FakeTokenizer(), "hello", k=5)


def test_plot_temperature_effect():
    plots.plot_temperature_effect(torch.randn(1, 40), FakeTokenizer(), temperatures=(0.5, 1.0), k=5)


def test_plot_top_p():
    plots.plot_top_p(torch.softmax(torch.randn(10), dim=-1), top_p=0.8)


def test_plot_accuracy_vs_samples():
    plots.plot_accuracy_vs_samples([1, 3, 5], [0.2, 0.3, 0.35], baseline=0.2, title="t")


def test_plot_brevity_bonus():
    plots.plot_brevity_bonus(500.0)


def test_plot_logits_to_logprobs():
    plots.plot_logits_to_logprobs(torch.linspace(-2, 2, 7))


def test_plot_token_logprobs():
    plots.plot_token_logprobs(["a", " b", " c"], torch.tensor([-0.5, -3.0, -0.1]), title="t")


def test_plot_bf16_resolution():
    plots.plot_bf16_resolution(0.02, 1e-5)


def test_plot_rewards_and_advantages():
    plots.plot_rewards_and_advantages([1.0, 1.0, 0.0, 0.0])
    plots.plot_rewards_and_advantages([0.0, 0.0], labels=["a", "b"])


def test_plot_advantage_patterns():
    plots.plot_advantage_patterns(4)


def test_plot_sequence_logprob():
    tokens = ["t1", "t2", "t3", "t4", "t5"]
    plots.plot_sequence_logprob(tokens, torch.tensor([-1.0, -2.0, -0.5, -3.0, -0.2]), prompt_len=3, title="t")


def test_plot_logprob_shift():
    plots.plot_logprob_shift(["a", "b"], [-8.0, -20.0], [-7.0, -22.0], [0.9, -0.9], title="t")


def test_plot_training_metrics():
    history = [{"step": i, "reward": i / 10, "length": 10 * i, "loss": 0.1 * i} for i in range(1, 12)]
    plots.plot_training_metrics(history, window=3)
