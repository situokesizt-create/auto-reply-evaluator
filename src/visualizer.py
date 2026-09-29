"""Headless, English-label charts with no platform-specific font dependency."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import METRICS

LABELS = ["Correctness", "Relevance", "Completeness", "Clarity", "Service"]


def create_charts(results: list[dict], summary: dict, validation: dict,
                  output: Path, mode: str) -> None:
    """Render means, full distributions, per-case scores and audit coverage."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 160})
    valid = [r for r in results if r.get("overall_score") is not None and not r.get("error")]
    frame = pd.DataFrame([r["scores"] for r in valid], columns=list(METRICS))
    suffix = "MOCK - pipeline demo only" if mode == "mock" else "DeepSeek API evaluation"

    def save(fig: plt.Figure, name: str) -> None:
        fig.text(0.99, 0.01, suffix, ha="right", fontsize=9, color="#64748b")
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(output / name, facecolor="white")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.25]})
    if valid:
        averages = frame.mean()
        bars = axes[0].bar(LABELS, averages, color="#2563eb", width=0.6)
        axes[0].bar_label(bars, fmt="%.2f", padding=4)
        axes[0].set_ylim(0, 5.6)
        distribution = np.array([summary["metrics"][key]["distribution"] for key in METRICS])
        im = axes[1].imshow(distribution, cmap="Blues", aspect="auto")
        for i in range(len(METRICS)):
            for j in range(6):
                axes[1].text(j, i, str(distribution[i, j]), ha="center", va="center",
                             color="white" if distribution[i, j] > distribution.max() / 2 else "#0f172a")
        axes[1].set_yticks(range(5), LABELS)
        axes[1].set_xticks(range(6), ["[0,1)", "[1,2)", "[2,3)", "[3,4)", "[4,5)", "5"])
        fig.colorbar(im, ax=axes[1], label="Case count", shrink=0.8)
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "No successful evaluations", ha="center", transform=ax.transAxes)
    axes[0].set_title("Average scores by metric")
    axes[0].set_ylabel("Score (0-5)")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].set_title("Score distribution by metric")
    axes[1].set_xlabel("Score interval")
    save(fig, "metric_distribution.png")

    fig, ax = plt.subplots(figsize=(13, 4.6))
    lowest = {r["id"] for r in summary["lowest"]}
    positions = range(len(results))
    values = [r["overall_score"] if r.get("overall_score") is not None else 0 for r in results]
    bars = ax.bar(positions, values, color=["#dc2626" if r["id"] in lowest else "#2563eb" for r in results])
    for bar, result in zip(bars, results):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.06,
                "ERR" if result.get("error") else f"{result['overall_score']:.2f}",
                ha="center", fontsize=8)
    ax.set_xticks(list(positions), [r["id"] for r in results], rotation=45, ha="right")
    ax.set_ylim(0, 5.7)
    ax.set_ylabel("Weighted score (0-5)")
    ax.set_xlabel("Case ID (red = lowest three; ERR = failed, not a zero score)")
    ax.set_title(f"Reply quality | {summary['successful']}/{summary['total']} successfully evaluated")
    save(fig, "overall_scores.png")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    counts = validation["counts"]
    bars = ax.bar([k.replace("_", " ").title() for k in counts], list(counts.values()),
                  color=["#16a34a", "#f59e0b", "#dc2626", "#8b5cf6", "#94a3b8", "#475569"])
    ax.bar_label(bars, padding=4)
    ax.set_ylim(0, max(counts.values(), default=0) + 3)
    ax.set_ylabel("Case count")
    ax.set_xlabel("Qualitative audit category (not gold-label accuracy)")
    ax.set_title(f"Human annotation audit | assessed {validation['assessed']}/{validation['total']}")
    save(fig, "validation.png")
