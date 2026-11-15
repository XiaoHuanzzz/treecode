#!/usr/bin/env python3
"""Validate fixed-parameter adaptive/uniform meshes on common target rules."""

from __future__ import annotations

import csv
import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiment_results" / "fixed_adaptive_uniform_20261109"
MESH = OUT / "D20.mesh"
CASES = [
    ("coarse", 3, "1.0", "0.14"),
    ("medium", 4, "0.1", "0.018"),
    ("fine", 5, "0.01", "0.0022"),
]


def number(text: str, label: str) -> float:
    match = re.search(re.escape(label) + r"\s*([0-9.eE+\-]+)", text)
    if match is None:
        raise RuntimeError(f"missing field: {label}")
    return float(match.group(1))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    output_file = OUT / "common_validation.csv"
    rows: list[dict] = []
    for quadrature_order in (8, 10):
        target_file = OUT / f"common_targets_q{quadrature_order}.txt"
        for label, level, tol1, tol2 in CASES:
            for mesh_type in ("adaptive", "uniform"):
                work = OUT / "common_validation" / (
                    f"{label}_{mesh_type}_q{quadrature_order}")
                work.mkdir(parents=True, exist_ok=True)
                executable = ROOT / (
                    "tree_exact" if mesh_type == "adaptive" else "tree_uniform")
                refinement = "0" if mesh_type == "adaptive" else str(level)
                command = [
                    str(executable), str(MESH), refinement,
                    "6", "4", "0.5", "1", "CPU", tol1, tol2,
                ]
                environment = os.environ.copy()
                environment.update({
                    "OMP_NUM_THREADS": "128",
                    "OMP_DYNAMIC": "FALSE",
                    "OMP_PROC_BIND": "close",
                    "OMP_PLACES": "cores",
                    "TREECODE_COMMON_TARGET_FILE": str(target_file),
                    "AFEPACK_TEMPLATE_PATH":
                        "/usr/local/include/AFEPack/template/triangle/:"
                        "/usr/local/include/AFEPack/template/twin_triangle/:"
                        "/usr/local/include/AFEPack/template/tetrahedron/:"
                        "/usr/local/include/AFEPack/template/twin_tetrahedron/:"
                        "/usr/local/include/AFEPack/template/four_tetrahedron/",
                })
                completed = subprocess.run(
                    command, cwd=work, env=environment, text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    check=False,
                )
                log = work / "console.log"
                log.write_text(completed.stdout)
                if completed.returncode != 0:
                    raise RuntimeError(f"failed: {work}; see {log}")
                mesh_output = work / (
                    "D.dx" if mesh_type == "adaptive" else "D_uniform.dx")
                mesh_output.unlink()
                row = {
                    "scale": label,
                    "mesh_type": mesh_type,
                    "target_rule_order": quadrature_order,
                    "target_count": int(number(
                        completed.stdout, "Common-target count:")),
                    "target_weight_sum": number(
                        completed.stdout, "Common-target weight sum:"),
                    "N": int(number(
                        completed.stdout, "Exact-comparison leaf barycenters:")),
                    "p": 6,
                    "P": 4,
                    "theta": 0.5,
                    "E_common_abs": number(
                        completed.stdout,
                        "Common-target L2 error against exact solution:"),
                    "E_common_rel": number(
                        completed.stdout, "Common-target relative L2 error:"),
                    "exact_norm_common": number(
                        completed.stdout,
                        "Common-target L2 norm of exact solution:"),
                    "validation_time_s": number(
                        completed.stdout, "Common-target validation time:"),
                    "log": str(log.relative_to(ROOT)),
                }
                rows.append(row)
                write_csv(output_file, rows)
                print(
                    f"DONE q{quadrature_order} {label} {mesh_type}: "
                    f"N={row['N']} E={row['E_common_abs']:.6e}",
                    flush=True,
                )
    print(f"COMPLETE: {output_file}", flush=True)


if __name__ == "__main__":
    main()
