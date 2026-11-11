"""Comparison charts in the Crusoe palette."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import matplotlib.pyplot as plt

from .bench import LABELS, BenchResult

#: Crusoe brand tokens (tokens/colors.json in the brand kit).
C = {
    "black": "#000000",
    "teal": "#395761",
    "slate": "#7799B6",
    "cloud": "#C3D6E7",
    "sage": "#99AE88",
    "moss": "#323C2A",
    "gold": "#ECAC45",
    "sunset": "#CD881A",
    "yellow": "#CEEB13",
    "gray": "#E4E2E0",
    "stone": "#A3A3A3",
}

#: One color per series; vendors first, then tuning variants.
SERIES = [C["slate"], C["gold"], C["sage"], C["teal"], C["sunset"], C["moss"], C["stone"]]
MONO = "monospace"


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, family=MONO, fontsize=12, loc="left")
    ax.set_xlabel(xlabel, family=MONO)
    ax.set_ylabel(ylabel, family=MONO)
    ax.grid(True, color=C["gray"], linewidth=1)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=9)


def compare_bars(
    results: Mapping[str, BenchResult],
    metrics: Iterable[str] = ("output_throughput", "median_ttft_ms", "median_tpot_ms"),
    title: str | None = None,
):
    """Bar charts of a single run per label, side by side per metric."""
    metrics = list(metrics)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.4 * len(metrics), 4), dpi=110)
    axes = [axes] if len(metrics) == 1 else list(axes)
    labels = list(results)
    for ax, metric in zip(axes, metrics):
        values = [results[k].metrics.get(metric, float("nan")) for k in labels]
        bars = ax.bar(labels, values, color=SERIES[: len(labels)], edgecolor=C["black"], linewidth=1.5)
        for b, v in zip(bars, values):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                f"{v:.0f}",
                ha="center",
                va="bottom",
                family=MONO,
                fontsize=9,
            )
        _style(ax, LABELS.get(metric, metric), "", "")
        ax.tick_params(axis="x", labelrotation=15)
    if title:
        fig.suptitle(title, family=MONO, fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    return fig
