#!/usr/bin/env python3
"""Summarize a fixed-order irregular-domain scaling directory."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

CASES = (("sphere", "Sphere"), ("l_prism", "L-shaped prism"),
         ("perforated_cube", "Perforated cube"))


def values(path: Path):
    with path.open(newline="") as stream:
        records = list(csv.DictReader(stream))
    return [row["group"] for row in records], [float(row["potential"]) for row in records]


def relative_rms(a, b, groups, name):
    ids = [i for i, group in enumerate(groups) if group == name]
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in ids)
                     / sum(b[i] ** 2 for i in ids))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    rows = []
    for case, title in CASES:
        ref_groups, reference = values(root / case / "L3.direct.csv")
        for level in range(4):
            prefix = root / case / f"L{level}"
            stats = json.loads(prefix.with_suffix(".json").read_text())
            groups, tree = values(prefix.with_suffix(".values.csv"))
            direct_groups, direct = values(prefix.with_suffix(".direct.csv"))
            if groups != ref_groups or direct_groups != ref_groups:
                raise RuntimeError(f"Target mismatch for {case}/L{level}")
            rows.append({
                "case": case, "domain": title, "level": level,
                "N": int(stats["N"]), "p": int(stats["p"]), "P": int(stats["P"]),
                "theta": stats["theta"], "threads": int(stats["threads"]),
                "Tlist_s": stats["list_s"], "Teval_s": stats["evaluation_s"],
                "Ttotal_s": stats["total_solver_s"],
                "Evolume_ref": relative_rms(tree, reference, groups, "volume"),
                "Ecore_ref": relative_rms(tree, reference, groups, "core"),
                "Evolume_same": relative_rms(tree, direct, groups, "volume"),
                "Ecore_same": relative_rms(tree, direct, groups, "core"),
                "reference_change_volume": relative_rms(direct, reference, groups, "volume"),
                "reference_change_core": relative_rms(direct, reference, groups, "core"),
                "charge": stats["charge"], "volume": stats["volume"],
            })
    p = rows[0]["p"]
    P = rows[0]["P"]
    if any(row["p"] != p or row["P"] != P for row in rows):
        raise RuntimeError("Expected fixed p and P")
    with (root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    colors = {"sphere": "#2f5597", "l_prism": "#d37a12",
              "perforated_cube": "#2a9d8f"}
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.15))
    for case, title in CASES:
        selected = [row for row in rows if row["case"] == case]
        n = [row["N"] for row in selected]
        axes[0].loglog(n, [row["Ttotal_s"] for row in selected], "o-",
                       color=colors[case], label=title)
        axes[1].loglog(n, [row["Evolume_ref"] for row in selected], "o-",
                       color=colors[case], label=title)
    base = next(row for row in rows if row["case"] == "sphere" and row["level"] == 1)
    nline = [base["N"], max(row["N"] for row in rows)]
    tline = [base["Ttotal_s"]]
    tline.append(tline[0] * nline[1] * math.log(nline[1])
                 / (nline[0] * math.log(nline[0])))
    axes[0].loglog(nline, tline, "k--", linewidth=1.2, label=r"$O(N\log N)$")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
        axis.set_xlabel(r"active leaves $N$")
    axes[0].set_ylabel("total time (s)")
    axes[1].set_ylabel("sampled relative error")
    axes[0].legend(frameon=False, loc="upper left")
    axes[1].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(root / "scaling_results.pdf", bbox_inches="tight")
    fig.savefig(root / "scaling_results.png", dpi=220, bbox_inches="tight")

    lines = [f"# Irregular-domain scaling study with p={p} and P={P}", "",
             f"All runs use p={p}, P={P}, theta=0.5 and 16 OpenMP threads.", "",
             "| Domain | level | N | Tlist (s) | Teval (s) | Ttotal (s) | Evolume | Ecore |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['domain']} | {row['level']} | {row['N']:,} | "
                     f"{row['Tlist_s']:.4f} | {row['Teval_s']:.4f} | "
                     f"{row['Ttotal_s']:.4f} | {row['Evolume_ref']:.4e} | "
                     f"{row['Ecore_ref']:.4e} |")
    lines += ["", "Timings were collected on a heavily loaded server and must be",
              "repeated on an idle machine before a performance claim is made."]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
