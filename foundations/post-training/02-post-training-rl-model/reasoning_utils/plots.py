"""Chart helpers used by the notebooks, so the lessons show data rather than matplotlib boilerplate.

Each function draws one figure with the track's palette; call it from a notebook cell and it renders inline.
"""

from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt

SLATE = "#7799B6"
GOLD = "#ECAC45"
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


def plot_template_comparison(template_results: dict, title: str) -> None:
    """Accuracy per prompt template, annotated with average response length. Values are evaluate_math summaries."""
    names = list(template_results)
    accs = [template_results[n]["accuracy"] for n in names]
    toks = [template_results[n]["avg_tokens"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 3.2))
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
