#!/usr/bin/env python3
"""Summarize the four-level irregular-domain scaling experiment."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent / "runs" / "irregular_scaling_20261108"
CASES = [("sphere", "Sphere"), ("l_prism", "L-shaped prism"),
         ("perforated_cube", "Perforated cube")]
LEVELS = range(4)
ORDER_CORRECTIONS = [("l_prism", "L-shaped prism", "L3_P4", "3, order corrected")]


def read_values(path: Path):
    groups, values = [], []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            groups.append(row["group"])
            values.append(float(row["potential"]))
    return groups, values


def rel_rms(values, reference, groups, selected):
    ids = [i for i, group in enumerate(groups) if group == selected]
    numerator = sum((values[i] - reference[i]) ** 2 for i in ids)
    denominator = sum(reference[i] ** 2 for i in ids)
    return math.sqrt(numerator / denominator)


rows = []
for case, title in CASES:
    ref_groups, final_direct = read_values(ROOT / case / "L3.direct.csv")
    for level in LEVELS:
        prefix = ROOT / case / f"L{level}"
        stats = json.loads(prefix.with_suffix(".json").read_text())
        groups, tree = read_values(prefix.with_suffix(".values.csv"))
        direct_groups, direct = read_values(prefix.with_suffix(".direct.csv"))
        if groups != ref_groups or direct_groups != ref_groups:
            raise RuntimeError(f"Target mismatch for {case} level {level}")
        row = {
            "case": case, "domain": title, "level": level,
            "N": int(stats["N"]), "p": int(stats["p"]),
            "P": int(stats["P"]), "theta": stats["theta"],
            "threads": int(stats["threads"]),
            "Tlist_s": stats["list_s"], "Teval_s": stats["evaluation_s"],
            "Ttotal_s": stats["total_solver_s"],
            "Evolume_ref": rel_rms(tree, final_direct, groups, "volume"),
            "Ecore_ref": rel_rms(tree, final_direct, groups, "core"),
            "Evolume_same": rel_rms(tree, direct, groups, "volume"),
            "Ecore_same": rel_rms(tree, direct, groups, "core"),
            "reference_change_volume": rel_rms(direct, final_direct, groups, "volume"),
            "reference_change_core": rel_rms(direct, final_direct, groups, "core"),
            "charge": stats["charge"], "volume": stats["volume"],
        }
        rows.append(row)

for case, title, name, label in ORDER_CORRECTIONS:
    prefix = ROOT / case / name
    if not prefix.with_suffix(".json").exists():
        continue
    ref_groups, final_direct = read_values(ROOT / case / "L3.direct.csv")
    stats = json.loads(prefix.with_suffix(".json").read_text())
    groups, tree = read_values(prefix.with_suffix(".values.csv"))
    direct_groups, direct = read_values(prefix.with_suffix(".direct.csv"))
    if groups != ref_groups or direct_groups != ref_groups:
        raise RuntimeError(f"Target mismatch for {case}/{name}")
    rows.append({
        "case": case, "domain": title, "level": label,
        "N": int(stats["N"]), "p": int(stats["p"]),
        "P": int(stats["P"]), "theta": stats["theta"],
        "threads": int(stats["threads"]),
        "Tlist_s": stats["list_s"], "Teval_s": stats["evaluation_s"],
        "Ttotal_s": stats["total_solver_s"],
        "Evolume_ref": rel_rms(tree, final_direct, groups, "volume"),
        "Ecore_ref": rel_rms(tree, final_direct, groups, "core"),
        "Evolume_same": rel_rms(tree, direct, groups, "volume"),
        "Ecore_same": rel_rms(tree, direct, groups, "core"),
        "reference_change_volume": rel_rms(direct, final_direct, groups, "volume"),
        "reference_change_core": rel_rms(direct, final_direct, groups, "core"),
        "charge": stats["charge"], "volume": stats["volume"],
    })

columns = list(rows[0])
with (ROOT / "summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

colors = {"sphere": "#2f5597", "l_prism": "#d37a12",
          "perforated_cube": "#2a9d8f"}
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.15))
for case, title in CASES:
    selected = [row for row in rows
                if row["case"] == case and isinstance(row["level"], int)]
    n = [row["N"] for row in selected]
    t = [row["Ttotal_s"] for row in selected]
    e = [row["Evolume_ref"] for row in selected]
    axes[0].loglog(n, t, "o-", color=colors[case], label=title)
    axes[1].loglog(n, e, "o-", color=colors[case], label=title)

corrected = [row for row in rows if not isinstance(row["level"], int)]
for row in corrected:
    axes[0].loglog(row["N"], row["Ttotal_s"], marker="D", markersize=6,
                   markerfacecolor="white", markeredgewidth=1.5,
                   color=colors[row["case"]], linestyle="none")
    axes[1].loglog(row["N"], row["Evolume_ref"], marker="D", markersize=6,
                   markerfacecolor="white", markeredgewidth=1.5,
                   color=colors[row["case"]], linestyle="none",
                   label=r"L-shaped prism, $P=4$")

base = next(row for row in rows if row["case"] == "sphere" and row["level"] == 1)
n_line = [base["N"], max(row["N"] for row in rows)]
t_line = [base["Ttotal_s"]]
t_line.append(t_line[0] * (n_line[1] * math.log(n_line[1])) /
              (n_line[0] * math.log(n_line[0])))
axes[0].loglog(n_line, t_line, "k--", linewidth=1.2, label=r"$O(N\log N)$")
for ax in axes:
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlabel(r"active leaves $N$")
axes[0].set_ylabel("total time (s)")
axes[1].set_ylabel("sampled relative error")
axes[0].legend(frameon=False, loc="upper left")
axes[1].legend(frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig(ROOT / "scaling_results.pdf", bbox_inches="tight")
fig.savefig(ROOT / "scaling_results.png", dpi=220, bbox_inches="tight")

lines = [
    "# Four-level treecode study on irregular domains", "",
    "The four mesh levels use p=2, P=2, theta=0.5 and 16 OpenMP threads. An",
    "additional finest-mesh L-shaped run changes only P from 2 to 4. Each starter",
    "mesh is globally refined three times, so every sequence spans 512x in N.",
    "Errors use the same 128 volume targets and 32 source-core targets.",
    "The level-3 direct values are numerical references; same-mesh errors isolate",
    "the tree approximation. Timings exclude all direct validation calculations.", "",
    "| Domain | level | p | P | N | Tlist (s) | Teval (s) | Ttotal (s) | Evolume | Ecore |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['domain']} | {row['level']} | {row['p']} | {row['P']} | {row['N']:,} | "
        f"{row['Tlist_s']:.4f} | {row['Teval_s']:.4f} | "
        f"{row['Ttotal_s']:.4f} | {row['Evolume_ref']:.4e} | "
        f"{row['Ecore_ref']:.4e} |")
lines += ["", "## Tree-approximation check", "",
          "| Domain | level | Evolume,same | Ecore,same |",
          "|---|---:|---:|---:|"]
for row in rows:
    lines.append(f"| {row['domain']} | {row['level']} | "
                 f"{row['Evolume_same']:.4e} | {row['Ecore_same']:.4e} |")
lines += ["", "The finest direct calculation is a numerical reference, not an exact",
          "solution. A quadrature/order sensitivity check is still required before",
          "the combined errors are used as final publication data."]
(ROOT / "report.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
