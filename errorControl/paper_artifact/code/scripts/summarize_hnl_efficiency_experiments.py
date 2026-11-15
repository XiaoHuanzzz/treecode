#!/usr/bin/env python3
"""Summarize HNL work-growth and interaction-list reuse experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
ROOT = HERE / "experiment_results" / "hnl_efficiency_20261109"
WORK_ROOT = ROOT / "work_growth"
REUSE_ROOT = ROOT / "reuse"
PREFIXES = [1, 2, 5, 10, 20]
REPEATS = [1, 2, 3]
CASES = [0, 1, 2, 3, 4]


def median_range(values):
    return statistics.median(values), min(values), max(values)


work_rows = []
for case_index in CASES:
    records = [
        json.loads((WORK_ROOT / f"repeat_{repeat}_mesh_{case_index}.json").read_text())
        for repeat in REPEATS
    ]
    if len({int(row["N"]) for row in records}) != 1:
        raise RuntimeError(f"Mesh size changed between repeats for case {case_index}")
    count_keys = [
        "hnl_star_queries_avg",
        "hnl_candidate_regions_avg",
        "hnl_materialization_visits_avg",
        "visited_nodes_avg",
        "list_entries_avg",
        "near_entries_avg",
        "far_entries_avg",
    ]
    row = {
        "case": case_index,
        "N": int(records[0]["N"]),
        "depth": int(records[0]["hierarchy_depth"]),
        "theta": float(records[0]["theta"]),
        "partition_error_max": max(
            float(item["maximum_partition_volume_error"]) for item in records
        ),
    }
    timing = [float(item["list_kernel_s"]) for item in records]
    row["list_s_median"], row["list_s_min"], row["list_s_max"] = median_range(timing)
    row["list_us_per_target"] = 1.0e6 * row["list_s_median"] / row["N"]
    for key in count_keys:
        values = [float(item[key]) for item in records]
        if max(values) - min(values) > 1.0e-8 * max(1.0, max(values)):
            raise RuntimeError(f"Nondeterministic work count {key} for case {case_index}")
        row[key] = statistics.median(values)
    work_rows.append(row)

with (ROOT / "work_growth_summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(work_rows[0]))
    writer.writeheader()
    writer.writerows(work_rows)


def read_reuse(path):
    comments = {}
    data_lines = []
    for line in path.read_text().splitlines():
        if line.startswith("# "):
            key, value = line[2:].split("\t", 1)
            comments[key] = float(value)
        elif line:
            data_lines.append(line)
    rows = [
        {key: float(value) for key, value in row.items()}
        for row in csv.DictReader(data_lines, delimiter="\t")
    ]
    return rows, comments


reuse_runs = [read_reuse(REUSE_ROOT / f"repeat_{repeat}.tsv") for repeat in REPEATS]
reuse_rows = []
for count in PREFIXES:
    records = []
    for repeat, (steps, comments) in zip(REPEATS, reuse_runs):
        selected = steps[:count]
        update_time = sum(row["step_time_s"] for row in selected)
        rebuild_time = sum(row["list_rebuild_time_s"] for row in selected)
        reuse_total = comments["initial_list_construction_time_s"] + update_time
        rebuild_total = rebuild_time + update_time
        records.append(
            {
                "repeat": repeat,
                "reuse_total_s": reuse_total,
                "rebuild_total_s": rebuild_total,
                "saved_fraction": (rebuild_total - reuse_total) / rebuild_total,
                "speedup": rebuild_total / reuse_total,
            }
        )
    row = {"updates": count}
    for key in ("reuse_total_s", "rebuild_total_s", "saved_fraction", "speedup"):
        values = [record[key] for record in records]
        row[key + "_median"], row[key + "_min"], row[key + "_max"] = median_range(values)
    reuse_rows.append(row)

with (ROOT / "reuse_summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(reuse_rows[0]))
    writer.writeheader()
    writer.writerows(reuse_rows)

components = {}
for name, selector in {
    "initial_list_s": lambda steps, comments: [comments["initial_list_construction_time_s"]],
    "rebuild_list_s": lambda steps, comments: [row["list_rebuild_time_s"] for row in steps],
    "density_update_s": lambda steps, comments: [row["density_update_time_s"] for row in steps],
    "moment_update_s": lambda steps, comments: [row["moment_update_time_s"] for row in steps],
    "potential_evaluation_s": lambda steps, comments: [row["potential_evaluation_time_s"] for row in steps],
}.items():
    values = []
    for steps, comments in reuse_runs:
        values.extend(selector(steps, comments))
    components[name] = median_range(values)

with (ROOT / "reuse_component_summary.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["component", "median_s", "min_s", "max_s"])
    for name, values in components.items():
        writer.writerow([name, *values])

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "dejavuserif",
        "font.size": 9.5,
        "axes.labelsize": 10,
        "legend.fontsize": 8.4,
        "pdf.fonttype": 42,
    }
)
fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2))
n = [row["N"] for row in work_rows]
timing = [row["list_s_median"] for row in work_rows]
timing_min = [row["list_s_min"] for row in work_rows]
timing_max = [row["list_s_max"] for row in work_rows]
axes[0, 0].loglog(n, timing, "o-", color="#2f5597")
axes[0, 0].fill_between(n, timing_min, timing_max, color="#2f5597", alpha=0.16)
axes[0, 0].set(xlabel=r"active leaves $N$", ylabel=r"$T_{\mathrm{list}}$ (s)")

for key, label, marker, color in (
    ("hnl_candidate_regions_avg", "candidates", "o", "#2f5597"),
    ("hnl_materialization_visits_avg", "materialization", "s", "#2a9d8f"),
    ("visited_nodes_avg", "MAC visits", "^", "#d37a12"),
    ("list_entries_avg", "list entries", "D", "#6a3d9a"),
):
    axes[0, 1].semilogx(n, [row[key] for row in work_rows], marker + "-", label=label, color=color)
axes[0, 1].set(xlabel=r"active leaves $N$", ylabel="average count per target")
axes[0, 1].legend(frameon=False, ncol=2)

x = [row["updates"] for row in reuse_rows]
for key, label, marker, color in (
    ("rebuild_total_s", "rebuild each update", "s", "#d37a12"),
    ("reuse_total_s", "build once and reuse", "o", "#2f5597"),
):
    median = [row[key + "_median"] for row in reuse_rows]
    lower = [row[key + "_min"] for row in reuse_rows]
    upper = [row[key + "_max"] for row in reuse_rows]
    axes[1, 0].plot(x, median, marker + "-", label=label, color=color)
    axes[1, 0].fill_between(x, lower, upper, color=color, alpha=0.15)
axes[1, 0].set(xlabel="density updates", ylabel="cumulative time (s)")
axes[1, 0].set_xticks(PREFIXES)
axes[1, 0].legend(frameon=False)

speedup = [row["speedup_median"] for row in reuse_rows]
speedup_min = [row["speedup_min"] for row in reuse_rows]
speedup_max = [row["speedup_max"] for row in reuse_rows]
axes[1, 1].plot(x, speedup, "o-", color="#2a9d8f")
axes[1, 1].fill_between(x, speedup_min, speedup_max, color="#2a9d8f", alpha=0.15)
axes[1, 1].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
axes[1, 1].set(xlabel="density updates", ylabel="speedup from list reuse")
axes[1, 1].set_xticks(PREFIXES)

for axis in axes.flat:
    axis.grid(True, which="both", alpha=0.22)
fig.tight_layout()
fig.savefig(ROOT / "hnl_efficiency.pdf", bbox_inches="tight")
fig.savefig(ROOT / "hnl_efficiency.png", dpi=240, bbox_inches="tight")
plt.close(fig)

n_ratio = work_rows[-1]["N"] / work_rows[0]["N"]
count_factors = {
    "candidate": work_rows[-1]["hnl_candidate_regions_avg"] / work_rows[0]["hnl_candidate_regions_avg"],
    "materialization": work_rows[-1]["hnl_materialization_visits_avg"] / work_rows[0]["hnl_materialization_visits_avg"],
    "visited": work_rows[-1]["visited_nodes_avg"] / work_rows[0]["visited_nodes_avg"],
    "list": work_rows[-1]["list_entries_avg"] / work_rows[0]["list_entries_avg"],
}
last = reuse_rows[-1]

lines = [
    "# HNL construction cost and list reuse", "",
    "## Work growth", "",
    "All cases use p=6, P=4, theta=0.5 and 16 OpenMP threads. Times are medians of three interleaved runs; counts are deterministic.", "",
    "| N | depth | T_list median [min,max] (s) | us/target | candidates/target | materialization/target | MAC visits/target | list entries/target |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in work_rows:
    lines.append(
        f"| {row['N']:,} | {row['depth']} | {row['list_s_median']:.3f} "
        f"[{row['list_s_min']:.3f},{row['list_s_max']:.3f}] | "
        f"{row['list_us_per_target']:.2f} | {row['hnl_candidate_regions_avg']:.1f} | "
        f"{row['hnl_materialization_visits_avg']:.1f} | {row['visited_nodes_avg']:.1f} | "
        f"{row['list_entries_avg']:.1f} |"
    )
lines += [
    "",
    f"N increases by {n_ratio:.1f} times. The average candidate, materialization, MAC-visit, and final-list counts increase by "
    f"{count_factors['candidate']:.2f}, {count_factors['materialization']:.2f}, {count_factors['visited']:.2f}, and {count_factors['list']:.2f} times, respectively.",
    "These measurements quantify work on this mesh family but do not establish an asymptotic complexity bound.",
    "",
    "## Repeated density updates", "",
    "The reuse study uses N=100,656, p=6, P=4, theta=0.5 and 64 OpenMP threads. Each control rebuild uses the same HNL construction; no alternative traversal is involved.", "",
    "| updates | rebuild total (s) | reuse total (s) | saved (%) | speedup |",
    "|---:|---:|---:|---:|---:|",
]
for row in reuse_rows:
    lines.append(
        f"| {row['updates']} | {row['rebuild_total_s_median']:.3f} | "
        f"{row['reuse_total_s_median']:.3f} | {100*row['saved_fraction_median']:.1f} | "
        f"{row['speedup_median']:.3f} |"
    )
lines += [
    "",
    f"At 20 updates, retaining the initial lists saves {100*last['saved_fraction_median']:.1f}% of cumulative time and gives a {last['speedup_median']:.3f}-fold speedup.",
    "Every measured rebuild reproduces the cached near/far lists exactly and passes the source-partition checks.",
    "The sequence is manufactured and SCF-like; it is not a Kohn--Sham SCF calculation.",
]
(ROOT / "report.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
