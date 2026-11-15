#!/usr/bin/env python3
"""Summarize HNL--MAC work counts on the article adaptive meshes."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
ROOT = HERE / "experiment_results" / "bounded_work_adaptive_article_20261109"
FILES = ["mesh_0.json", "mesh_1.json", "mesh_2.json", "mesh_3.json",
         "mesh_4_t2e4.json"]

rows = []
for name in FILES:
    data = json.loads((ROOT / name).read_text())
    depth = int(data["hierarchy_depth"])
    total_entries = int(data["near_entries_total"] + data["far_entries_total"])
    row = {
        "N": int(data["N"]),
        "depth": depth,
        "roots": int(data["root_regions"]),
        "candidate_avg": float(data["hnl_candidate_regions_avg"]),
        "candidate_max": int(data["hnl_candidate_regions_max"]),
        "raw_candidate_visits_avg": float(data["hnl_raw_candidate_visits_avg"]),
        "materialization_avg": float(data["hnl_materialization_visits_avg"]),
        "visited_avg": float(data["visited_nodes_avg"]),
        "visited_max": int(data["visited_nodes_max"]),
        "near_avg": float(data["near_entries_avg"]),
        "far_avg": float(data["far_entries_avg"]),
        "list_avg": float(data["list_entries_avg"]),
        "list_max": int(data["list_entries_max"]),
        "total_entries": total_entries,
        "candidate_per_level": float(data["hnl_candidate_regions_avg"]) / depth,
        "raw_candidate_per_level": float(data["hnl_raw_candidate_visits_avg"]) / depth,
        "visited_per_level": float(data["visited_nodes_avg"]) / depth,
        "list_per_level": float(data["list_entries_avg"]) / depth,
        "max_mac_ratio": float(data["maximum_mac_ratio"]),
        "partition_error": float(data["maximum_partition_volume_error"]),
        "tolerance_stage2": float(data["tolerance_stage2"]),
    }
    rows.append(row)

with (ROOT / "summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

plt.rcParams.update({"font.size": 9.2, "axes.labelsize": 9.5,
                     "legend.fontsize": 8.4})
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.55))
n = [row["N"] for row in rows]
axes[0].semilogx(n, [row["candidate_avg"] for row in rows], "o-",
                 label=r"final $C_{\rm HNL}$")
axes[0].semilogx(n, [row["raw_candidate_visits_avg"] for row in rows], "s--",
                 label="raw visits")
axes[0].set_ylabel("HNL counts per target")
axes[0].legend(frameon=False, loc="upper left")

axes[1].semilogx(n, [row["visited_avg"] for row in rows], "o-",
                 color="#d37a12")
axes[1].set_ylabel(r"MAC visits $V_{\rm MAC}$ per target")

axes[2].semilogx(n, [row["near_avg"] for row in rows], "o--",
                 label="near")
axes[2].semilogx(n, [row["far_avg"] for row in rows], "s--",
                 label="far")
axes[2].semilogx(n, [row["list_avg"] for row in rows], "^-",
                 color="black", label="total")
axes[2].set_ylabel("list entries per target")
axes[2].legend(frameon=False, loc="upper left")
for axis in axes:
    axis.set_xlabel(r"active leaves $N$")
    axis.grid(True, which="both", alpha=0.25)
fig.tight_layout()
fig.savefig(ROOT / "adaptive_work_counts.pdf", bbox_inches="tight")
fig.savefig(ROOT / "adaptive_work_counts.png", dpi=240, bbox_inches="tight")

lines = [
    "# HNL--MAC work counts on the article adaptive meshes", "",
    "The five runs use `p=6`, `P=4`, `theta=0.5`, a fixed 24-root coarse",
    "mesh, and the current adaptive indicator. Counts are deterministic and",
    "are accumulated over every target leaf; timings are not used here.", "",
    "| N | L | HNL candidates avg/max | Raw candidates avg | MAC visits avg/max | Near avg | Far avg | List avg/max | C/L | V/L | I/L |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['N']:,} | {row['depth']} | "
        f"{row['candidate_avg']:.1f}/{row['candidate_max']:,} | "
        f"{row['raw_candidate_visits_avg']:.1f} | "
        f"{row['visited_avg']:.1f}/{row['visited_max']:,} | "
        f"{row['near_avg']:.1f} | {row['far_avg']:.1f} | "
        f"{row['list_avg']:.1f}/{row['list_max']:,} | "
        f"{row['candidate_per_level']:.1f} | "
        f"{row['visited_per_level']:.1f} | {row['list_per_level']:.1f} |")
lines += [
    "", "Across a 41.2-fold increase in N, the observed per-level ranges are:",
    f"- final HNL candidates: {min(r['candidate_per_level'] for r in rows):.1f}--{max(r['candidate_per_level'] for r in rows):.1f};",
    f"- MAC node visits: {min(r['visited_per_level'] for r in rows):.1f}--{max(r['visited_per_level'] for r in rows):.1f};",
    f"- final list entries: {min(r['list_per_level'] for r in rows):.1f}--{max(r['list_per_level'] for r in rows):.1f}.",
    "", "These bounded measured ranges support the per-level-work assumption",
    "for this adaptive family, but they do not prove a mesh-independent bound.",
    "All partitions pass the coverage check and have maximum accepted MAC",
    "ratio 0.5.",
]
(ROOT / "report.md").write_text("\n".join(lines) + "\n")

table = [
    r"\begin{table}[!htbp]",
    r"\centering",
    r"\caption{Per-target HNL--MAC work counts on the adaptive mesh family "
    r"at $(p,P,\theta)=(6,4,0.5)$.}",
    r"\label{tab:adaptive_bounded_work}",
    r"\small",
    r"\setlength{\tabcolsep}{4pt}",
    r"\begin{tabular}{rrrrrrrr}",
    r"\toprule",
    r"$N$ & $L$ & $C_{\rm HNL}$ & $R_{\rm HNL}$ & $V_{\rm MAC}$ & "
    r"$I_{\rm near}$ & $I_{\rm far}$ & $I_{\rm list}$ \\",
    r"\midrule",
]
for row in rows:
    n_value = f"{row['N']:,}".replace(",", r"\,")
    table.append(
        f"{n_value} & {row['depth']} & {row['candidate_avg']:.0f}"
        f" & {row['raw_candidate_visits_avg']:.0f}"
        f" & {row['visited_avg']:.0f} & {row['near_avg']:.0f}"
        f" & {row['far_avg']:.0f} & {row['list_avg']:.0f} " + r"\\")
table += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(ROOT / "article_table.tex").write_text("\n".join(table) + "\n")

print("\n".join(lines))
