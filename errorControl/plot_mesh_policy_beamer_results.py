#!/usr/bin/env python3
"""Create the compact policy-comparison figure used by the SCF beamer."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SUMMARY = Path("experiment_results/mesh_policy_summary.tsv")
OUTPUT = Path("mesh_policy_beamer_results.png")


def main() -> None:
    with SUMMARY.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))

    wanted = [
        "FROZEN",
        "EVERY_H",
        "PERIODIC_H(H=4)",
        "RESIDUAL_H(tau=0.18)",
        "MOVING_R(omega=0.10)",
        "HYBRID_HR(tau=0.18,omega=0.10)",
    ]
    labels = ["Frozen", "Every-$h$", "Periodic\n$H=4$", "Residual\n$\\tau=0.18$",
              "Moving\n$\\omega=0.10$", "Hybrid"]
    by_variant = {row["variant"]: row for row in rows}
    selected = [by_variant[name] for name in wanted]
    times = np.array([float(row["T_sequence_s"]) for row in selected])
    errors = np.array([float(row["E_mean_post0"]) for row in selected])

    colors = ["#65727C", "#3E8E63", "#087E8B", "#E07A3F", "#B94A48", "#183A5A"]
    x = np.arange(len(labels))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelcolor": "#27323A",
        "xtick.color": "#27323A",
        "ytick.color": "#27323A",
        "axes.edgecolor": "#65727C",
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.65), constrained_layout=True)

    bars = axes[0].bar(x, times, color=colors, width=0.72)
    axes[0].set_ylabel("20-step sequence time (s)")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 63)
    axes[0].grid(axis="y", color="#D9E0E4", linewidth=0.7)
    axes[0].set_axisbelow(True)
    for bar, value in zip(bars, times):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 1.1,
                     f"{value:.1f}", ha="center", va="bottom", fontsize=9)

    axes[1].scatter(x, errors, s=90, c=colors, edgecolor="white", linewidth=0.8, zorder=3)
    axes[1].plot(x, errors, color="#AAB4BA", linewidth=1.0, zorder=1)
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"Mean $L^2$ error, steps 1--19")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(4.0e-5, 6.0e-4)
    axes[1].grid(axis="y", which="both", color="#D9E0E4", linewidth=0.7)
    axes[1].set_axisbelow(True)
    for index, value in enumerate(errors):
        axes[1].annotate(f"{value:.1e}", (index, value), xytext=(0, 8),
                         textcoords="offset points", ha="center", fontsize=8)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.savefig(OUTPUT, dpi=220, facecolor="white")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
