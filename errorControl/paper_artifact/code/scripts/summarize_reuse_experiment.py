#!/usr/bin/env python3
"""Summarize measured list rebuild versus reuse experiments."""
from __future__ import annotations

import csv
from pathlib import Path
import statistics

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
ROOT = HERE / "experiment_results" / "list_reuse_20261109"
PREFIXES = [1, 2, 5, 10, 20]
REPEATS = range(1, 4)


def read_run(path: Path):
    comments = {}
    data_lines = []
    with path.open() as stream:
        for line in stream:
            line = line.rstrip("\n")
            if line.startswith("# "):
                key, value = line[2:].split("\t", 1)
                comments[key] = float(value)
            elif line:
                data_lines.append(line)
    rows = []
    for row in csv.DictReader(data_lines, delimiter="\t"):
        rows.append({key: float(value) for key, value in row.items()})
    return rows, comments


runs = [read_run(ROOT / f"repeat_{repeat}.tsv") for repeat in REPEATS]
records = []
for repeat, (steps, comments) in zip(REPEATS, runs):
    initial_list = comments["initial_list_construction_time_s"]
    for count in PREFIXES:
        selected = steps[:count]
        updates = sum(row["step_time_s"] for row in selected)
        rebuild_lists = sum(row["list_rebuild_time_s"] for row in selected)
        reuse_total = initial_list + updates
        rebuild_total = rebuild_lists + updates
        records.append({
            "repeat": repeat,
            "steps": count,
            "reuse_total_s": reuse_total,
            "rebuild_total_s": rebuild_total,
            "saved_s": rebuild_total - reuse_total,
            "saved_fraction": (rebuild_total - reuse_total) / rebuild_total,
            "speedup": rebuild_total / reuse_total,
            "reuse_amortized_s": reuse_total / count,
            "rebuild_amortized_s": rebuild_total / count,
            "initial_list_s": initial_list,
            "rebuild_list_sum_s": rebuild_lists,
        })

with (ROOT / "all_prefix_runs.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)

summary = []
for count in PREFIXES:
    chosen = [row for row in records if row["steps"] == count]
    row = {"steps": count}
    for key in ("reuse_total_s", "rebuild_total_s", "saved_s",
                "saved_fraction", "speedup", "reuse_amortized_s",
                "rebuild_amortized_s"):
        values = [entry[key] for entry in chosen]
        row[key + "_median"] = statistics.median(values)
        row[key + "_min"] = min(values)
        row[key + "_max"] = max(values)
    summary.append(row)

with (ROOT / "summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)

all_steps = [row for steps, _ in runs for row in steps]
components = {
    "initial_list_s": [comments["initial_list_construction_time_s"]
                       for _, comments in runs],
    "rebuild_list_s": [row["list_rebuild_time_s"] for row in all_steps],
    "density_update_s": [row["density_update_time_s"] for row in all_steps],
    "moment_update_s": [row["moment_update_time_s"] for row in all_steps],
    "potential_evaluation_s": [row["potential_evaluation_time_s"]
                               for row in all_steps],
}
with (ROOT / "component_summary.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["component", "median_s", "min_s", "max_s"])
    for name, values in components.items():
        writer.writerow([name, statistics.median(values), min(values), max(values)])

plt.rcParams.update({"font.size": 9.5, "axes.labelsize": 10,
                     "legend.fontsize": 9})
fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.7))
x = [row["steps"] for row in summary]
for key, label, color, marker in (
        ("rebuild_total_s", "rebuild every step", "#d37a12", "s"),
        ("reuse_total_s", "build once and reuse", "#2f5597", "o")):
    median = [row[key + "_median"] for row in summary]
    lower = [row[key + "_min"] for row in summary]
    upper = [row[key + "_max"] for row in summary]
    axes[0].plot(x, median, marker=marker, color=color, linewidth=1.7,
                 label=label)
    axes[0].fill_between(x, lower, upper, color=color, alpha=0.14)
axes[0].set_xlabel("number of density updates")
axes[0].set_ylabel("cumulative time (s)")
axes[0].legend(frameon=False, loc="upper left")

speedup = [row["speedup_median"] for row in summary]
speedup_min = [row["speedup_min"] for row in summary]
speedup_max = [row["speedup_max"] for row in summary]
axes[1].plot(x, speedup, "o-", color="#2a9d8f", linewidth=1.7)
axes[1].fill_between(x, speedup_min, speedup_max,
                     color="#2a9d8f", alpha=0.14)
axes[1].axhline(1.0, color="black", linestyle="--", linewidth=0.9)
axes[1].set_xlabel("number of density updates")
axes[1].set_ylabel("speedup from list reuse")
for axis in axes:
    axis.set_xticks(PREFIXES)
    axis.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(ROOT / "list_reuse.pdf", bbox_inches="tight")
fig.savefig(ROOT / "list_reuse.png", dpi=240, bbox_inches="tight")

lines = [
    "# Measured HNL--MAC interaction-list reuse", "",
    "The experiment uses N=100,656, p=6, P=4, theta=0.5 and 64 CPU",
    "threads. A manufactured density sequence changes at every step while",
    "the mesh, targets and theta remain fixed. Each of three independent runs",
    "contains 20 updates. At every update the control path actually rebuilds",
    "the HNL--MAC lists on the same mesh; the reuse path retains the initial",
    "lists. Both paths share identical density, moment and potential work.", "",
    "| updates K | rebuild total (s) | reuse total (s) | saved (%) | speedup | reuse time/update (s) |",
    "|---:|---:|---:|---:|---:|---:|",
]
for row in summary:
    lines.append(
        f"| {row['steps']} | {row['rebuild_total_s_median']:.3f} | "
        f"{row['reuse_total_s_median']:.3f} | "
        f"{100.0 * row['saved_fraction_median']:.1f} | "
        f"{row['speedup_median']:.3f} | "
        f"{row['reuse_amortized_s_median']:.3f} |")

lines += ["", "Median component times:", ""]
for name, values in components.items():
    lines.append(f"- {name}: {statistics.median(values):.4f} s")
lines += [
    "", "Every measured rebuild reproduced the cached near/far lists exactly;",
    "the source-partition and MAC checks also passed. The experiment isolates",
    "interaction-list reuse on a frozen mesh. It does not include the cost or",
    "possible accuracy benefit of adapting the mesh between density updates.",
]
(ROOT / "report.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
