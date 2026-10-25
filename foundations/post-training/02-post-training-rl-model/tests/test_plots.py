"""Smoke tests: the chart helpers draw without error on synthetic data (headless backend)."""

import matplotlib

matplotlib.use("Agg")

from reasoning_utils import plots  # noqa: E402


def test_plot_benchmark_composition():
    problems = [
        {"subject": "Algebra", "level": 1},
        {"subject": "Algebra", "level": 3},
        {"subject": "Geometry", "level": 3},
    ]
    plots.plot_benchmark_composition(problems)


def test_plot_template_comparison():
    plots.plot_template_comparison(
        {"x": {"accuracy": 0.2, "avg_tokens": 90}, "y": {"accuracy": 0.3, "avg_tokens": 120}}, "t"
    )
