#!/usr/bin/env python3
"""Summarize the recorded SCF mesh-policy TSV files.

The per-step TSV files remain the authoritative raw results.  This script
creates compact TSV and Markdown tables containing the parameters and the
quantities needed for a policy-level comparison.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path


RESULT_DIR = Path("experiment_results")

RUNS = [
    {
        "variant": "FROZEN",
        "file": "mesh_policy_FROZEN.tsv",
        "period": "-",
        "residual_trigger": "-",
        "max_reuse": "-",
        "move_relax": "-",
        "move_iters": "-",
    },
    {
        "variant": "EVERY_H",
        "file": "mesh_policy_EVERY_H.tsv",
        "period": "1",
        "residual_trigger": "-",
        "max_reuse": "-",
        "move_relax": "-",
        "move_iters": "-",
    },
    {
        "variant": "PERIODIC_H(H=2)",
        "file": "mesh_policy_PERIODIC_H_p2.tsv",
        "period": "2",
        "residual_trigger": "-",
        "max_reuse": "-",
        "move_relax": "-",
        "move_iters": "-",
    },
    {
        "variant": "PERIODIC_H(H=4)",
        "file": "mesh_policy_PERIODIC_H_p4.tsv",
        "period": "4",
        "residual_trigger": "-",
        "max_reuse": "-",
        "move_relax": "-",
        "move_iters": "-",
    },
    {
        "variant": "PERIODIC_H(H=8)",
        "file": "mesh_policy_PERIODIC_H_p8.tsv",
        "period": "8",
        "residual_trigger": "-",
        "max_reuse": "-",
        "move_relax": "-",
        "move_iters": "-",
    },
    {
        "variant": "RESIDUAL_H(tau=0.18)",
        "file": "mesh_policy_RESIDUAL_H_tau0p18.tsv",
        "period": "-",
        "residual_trigger": "0.18",
        "max_reuse": "4",
        "move_relax": "-",
        "move_iters": "-",
    },
    {
        "variant": "MOVING_R(omega=0.05)",
        "file": "mesh_policy_MOVING_R_w0p05.tsv",
        "period": "-",
        "residual_trigger": "-",
        "max_reuse": "-",
        "move_relax": "0.05",
        "move_iters": "2",
    },
    {
        "variant": "MOVING_R(omega=0.10)",
        "file": "mesh_policy_MOVING_R_w0p1.tsv",
        "period": "-",
        "residual_trigger": "-",
        "max_reuse": "-",
        "move_relax": "0.10",
        "move_iters": "2",
    },
    {
        "variant": "MOVING_R(omega=0.20)",
        "file": "mesh_policy_MOVING_R_w0p2.tsv",
        "period": "-",
        "residual_trigger": "-",
        "max_reuse": "-",
        "move_relax": "0.20",
        "move_iters": "2",
    },
    {
        "variant": "HYBRID_HR(tau=0.18,omega=0.10)",
        "file": "mesh_policy_HYBRID_HR_tau0p18_w0p1.tsv",
        "period": "-",
        "residual_trigger": "0.18",
        "max_reuse": "4",
        "move_relax": "0.10",
        "move_iters": "2",
    },
]


def fsum(rows: list[dict[str, str]], key: str) -> float:
    return math.fsum(float(row[key]) for row in rows)


def read_result(path: Path) -> tuple[list[dict[str, str]], dict[str, str]]:
    data_lines: list[str] = []
    footer: dict[str, str] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("# "):
                key, value = line[2:].rstrip("\n").split("\t", 1)
                footer[key] = value
            elif line.strip():
                data_lines.append(line)
    rows = list(csv.DictReader(data_lines, delimiter="\t"))
    if not rows:
        raise ValueError(f"no result rows in {path}")
    return rows, footer


def summarize(run: dict[str, str]) -> dict[str, str]:
    path = RESULT_DIR / run["file"]
    rows, footer = read_result(path)
    first = rows[0]
    post_initial = rows[1:] or rows
    n_values = [int(row["n_elements"]) for row in rows]
    errors = [float(row["L2_error"]) for row in post_initial]
    mesh_time = fsum(rows, "mesh_update_time_s")
    mapping_time = fsum(rows, "mapping_time_s")
    validation_time = fsum(rows, "hierarchy_validation_time_s")
    list_time = fsum(rows, "list_time_s")
    quadrature_time = fsum(rows, "quadrature_cache_time_s")
    density_time = fsum(rows, "density_update_time_s")
    moment_time = fsum(rows, "moment_time_s")
    evaluation_time = fsum(rows, "potential_time_s")
    sequence_time = float(footer["sequence_time_s"])
    tree_time = mapping_time + validation_time + list_time
    source_time = quadrature_time + density_time
    control_time = sequence_time - (
        mesh_time + tree_time + source_time + moment_time + evaluation_time
    )
    return {
        "variant": run["variant"],
        "mode": first["mode"],
        "p": first["p"],
        "P": first["P"],
        "theta": first["theta"],
        "steps": str(len(rows)),
        "threads": first["omp_threads"],
        "period": run["period"],
        "residual_trigger": run["residual_trigger"],
        "max_reuse": run["max_reuse"],
        "move_relax": run["move_relax"],
        "move_iters": run["move_iters"],
        "n_initial": str(n_values[0]),
        "n_min": str(min(n_values)),
        "n_max": str(max(n_values)),
        "h_updates": footer.get("h_update_count", "?"),
        "r_updates": footer.get("move_update_count", "?"),
        "T_mesh_update_s": f"{mesh_time:.6f}",
        "T_mapping_s": f"{mapping_time:.6f}",
        "T_validation_s": f"{validation_time:.6f}",
        "T_list_s": f"{list_time:.6f}",
        "T_tree_list_s": f"{tree_time:.6f}",
        "T_quadrature_s": f"{quadrature_time:.6f}",
        "T_density_s": f"{density_time:.6f}",
        "T_source_s": f"{source_time:.6f}",
        "T_moment_s": f"{moment_time:.6f}",
        "T_eval_s": f"{evaluation_time:.6f}",
        "T_control_other_s": f"{control_time:.6f}",
        "T_sequence_s": f"{sequence_time:.6f}",
        "T_wall_s": f"{float(footer['wall_time_including_initial_setup_s']):.6f}",
        "E_mean_post0": f"{math.fsum(errors) / len(errors):.8e}",
        "E_max_post0": f"{max(errors):.8e}",
        "E_final": f"{float(rows[-1]['L2_error']):.8e}",
    }


def write_tsv(summaries: list[dict[str, str]]) -> None:
    path = RESULT_DIR / "mesh_policy_summary.tsv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(summaries)


def write_markdown(summaries: list[dict[str, str]]) -> None:
    path = RESULT_DIR / "mesh_policy_summary.md"
    lines = [
        "# SCF mesh-policy experiment summary",
        "",
        "Common numerical parameters: `D.mesh`, two initial global refinements, "
        "`p=6`, `P=4`, `theta=0.5`, 20 steps, density amplitude decay `0.95`, "
        "initial amplitude `0.5`, initial center shift `0.8`, adaptive tolerance "
        "`1e-6`, one h-adaptation pass, and 128 OpenMP threads.",
        "",
        "Moving-mesh runs use two movement iterations, maximum displacement "
        "fraction `0.1`, and minimum tetrahedron quality `0.05`. Residual and "
        "hybrid runs use `max_reuse=4`. Error statistics exclude the common "
        "initial step (step 0); `T_wall` includes the common initial setup.",
        "The moving monitor uses strength `10`, exponent `0.5`, floor `0.1`, "
        "and three graph-smoothing passes. Base radial adaptation is enabled; "
        "subsequent h-adaptation permits both refinement and coarsening.",
        "Timing definitions: `T_mesh` is h/r mesh modification; `T_tree/list` "
        "combines hierarchy mapping, validation, and MAC-list construction; "
        "`T_source` combines quadrature-cache construction and source-density "
        "refresh; `T_moment` updates Cartesian moments; `T_eval` evaluates the "
        "near, singular, and far potential at one barycenter per leaf and "
        "accumulates the L2 error. `T_control/other` is the measured remainder, "
        "dominated by density-change and source-residual indicators.",
        "",
        "| Variant | n range | h/r updates | T_mesh | T_tree/list | T_source | "
        "T_moment | T_eval | T_control/other | T_sequence | T_wall | mean L2 | max L2 | final L2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['variant']} | {row['n_min']}--{row['n_max']} | "
            f"{row['h_updates']}/{row['r_updates']} | {row['T_mesh_update_s']} | "
            f"{row['T_tree_list_s']} | {row['T_source_s']} | {row['T_moment_s']} | "
            f"{row['T_eval_s']} | {row['T_control_other_s']} | {row['T_sequence_s']} | "
            f"{row['T_wall_s']} | {row['E_mean_post0']} | "
            f"{row['E_max_post0']} | {row['E_final']} |"
        )
    lines.extend(
        [
            "",
            "The complete step-by-step parameters, update decisions, mesh sizes, "
            "timings, errors, displacement, and quality measurements remain in "
            "the corresponding `mesh_policy_*.tsv` files.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    missing = [run["file"] for run in RUNS if not (RESULT_DIR / run["file"]).exists()]
    if missing:
        raise FileNotFoundError("missing result files: " + ", ".join(missing))
    summaries = [summarize(run) for run in RUNS]
    write_tsv(summaries)
    write_markdown(summaries)
    print(f"Wrote {RESULT_DIR / 'mesh_policy_summary.tsv'}")
    print(f"Wrote {RESULT_DIR / 'mesh_policy_summary.md'}")


if __name__ == "__main__":
    main()
