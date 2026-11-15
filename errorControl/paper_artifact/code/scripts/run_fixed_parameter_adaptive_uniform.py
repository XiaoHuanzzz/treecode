#!/usr/bin/env python3
"""Run the fixed-parameter adaptive-versus-uniform article study.

The C++ executable performs every numerical solve.  This driver only runs
interleaved repetitions, extracts the reported quantities, and writes a
reproducible CSV summary.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiment_results" / "fixed_adaptive_uniform_20261109"
MESH = OUT / "D20.mesh"
PARAMETERS = {"p": 6, "P": 4, "theta": 0.5, "threads": 128}
CASES = [
    # label, uniform level, stage-1 tolerance, stage-2 tolerance
    ("coarse", 3, "1.0", "0.14"),
    ("medium", 4, "0.1", "0.018"),
    ("fine", 5, "0.01", "0.0022"),
]
REPEATS = 3


def extract(text: str, label: str) -> float:
    match = re.search(re.escape(label) + r"\s*([0-9.eE+\-]+)", text)
    if match is None:
        raise RuntimeError(f"missing output field: {label}")
    return float(match.group(1))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_one(label: str, level: int, tol1: str, tol2: str,
            mesh_type: str, repeat: int) -> dict:
    work = OUT / "production" / f"{label}_{mesh_type}_r{repeat}"
    work.mkdir(parents=True, exist_ok=True)
    executable = ROOT / ("tree_exact" if mesh_type == "adaptive" else "tree_uniform")
    refinement = "0" if mesh_type == "adaptive" else str(level)
    command = [
        str(executable), str(MESH), refinement,
        str(PARAMETERS["p"]), str(PARAMETERS["P"]),
        str(PARAMETERS["theta"]), "1", "CPU", tol1, tol2,
    ]
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": str(PARAMETERS["threads"]),
        "OMP_DYNAMIC": "FALSE",
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
        "AFEPACK_TEMPLATE_PATH":
            "/usr/local/include/AFEPack/template/triangle/:"
            "/usr/local/include/AFEPack/template/twin_triangle/:"
            "/usr/local/include/AFEPack/template/tetrahedron/:"
            "/usr/local/include/AFEPack/template/twin_tetrahedron/:"
            "/usr/local/include/AFEPack/template/four_tetrahedron/",
    })
    completed = subprocess.run(
        command, cwd=work, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    log = work / "console.log"
    log.write_text(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(f"{work.name} failed; see {log}")
    expected_marker = f"mesh={mesh_type}"
    if expected_marker not in completed.stdout:
        raise RuntimeError(f"{work.name} did not use {mesh_type} mesh mode")
    if "Interaction search: HNL" not in completed.stdout:
        raise RuntimeError(f"{work.name} did not use the HNL interaction path")
    mesh_output = work / ("D.dx" if mesh_type == "adaptive" else "D_uniform.dx")
    mesh_hash = digest(mesh_output)
    # The mesh is deterministic and can be regenerated.  Keeping eighteen DX
    # copies would dominate the storage required by this timing study.
    mesh_output.unlink()
    return {
        "scale": label,
        "mesh_type": mesh_type,
        "repeat": repeat,
        "uniform_level": level,
        "tolerance_stage1": tol1 if mesh_type == "adaptive" else "",
        "tolerance_stage2": tol2 if mesh_type == "adaptive" else "",
        "N": int(extract(completed.stdout, "Exact-comparison leaf barycenters:")),
        "p": PARAMETERS["p"],
        "P": PARAMETERS["P"],
        "theta": PARAMETERS["theta"],
        "threads": PARAMETERS["threads"],
        "T_list_s": extract(completed.stdout, "Filling neighbor_elements uses:"),
        "T_eval_s": extract(completed.stdout, "Potential evaluation time:"),
        "T_total_s": extract(completed.stdout, "Total execution time:"),
        "E_abs_barycenter": extract(
            completed.stdout, "L2 error against exact solution:"),
        "exact_norm_barycenter": extract(
            completed.stdout, "L2 norm of exact solution:"),
        "mesh_sha256": mesh_hash,
        "log": str(log.relative_to(ROOT)),
    }


def main() -> None:
    if not MESH.exists():
        raise RuntimeError(f"missing expanded-domain mesh: {MESH}")
    if not (ROOT / "tree_exact").exists() or not (ROOT / "tree_uniform").exists():
        raise RuntimeError("build tree_exact and tree_uniform before running")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "[-20,20]^3",
        "mesh": str(MESH.relative_to(ROOT)),
        "mesh_sha256": digest(MESH),
        "parameters": PARAMETERS,
        "cases": CASES,
        "repeats": REPEATS,
        "timing": "three interleaved runs; median reported",
        "production_error":
            "volume-weighted leaf-barycenter norm against analytic solution; "
            "recorded for diagnostics but not used in the article comparison",
        "reported_error":
            "common-target weighted norm against analytic solution; see "
            "common_validation.csv",
        "common_target_rule":
            "tensor product of 10-point Gauss rules on the seven intervals "
            "separated by -20,-2,-1,-0.5,0.5,1,2,20 in each coordinate",
        "common_target_check":
            "order-8 to order-10 error change is at most 1.1 percent",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    runs: list[dict] = []
    for label, level, tol1, tol2 in CASES:
        for repeat in range(1, REPEATS + 1):
            order = ("adaptive", "uniform") if repeat != 2 else ("uniform", "adaptive")
            for mesh_type in order:
                row = run_one(label, level, tol1, tol2, mesh_type, repeat)
                runs.append(row)
                write_csv(OUT / "runs.csv", runs)
                print(
                    f"DONE {label} {mesh_type} r{repeat}: "
                    f"N={row['N']} E={row['E_abs_barycenter']:.6e} "
                    f"T={row['T_total_s']:.3f}s",
                    flush=True,
                )

    summary: list[dict] = []
    for label, level, _, _ in CASES:
        for mesh_type in ("adaptive", "uniform"):
            selected = [
                row for row in runs
                if row["scale"] == label and row["mesh_type"] == mesh_type
            ]
            if len(selected) != REPEATS:
                raise RuntimeError(f"missing repetitions for {label} {mesh_type}")
            if len({row["N"] for row in selected}) != 1:
                raise RuntimeError(f"mesh count changed for {label} {mesh_type}")
            if len({row["mesh_sha256"] for row in selected}) != 1:
                raise RuntimeError(f"mesh changed for {label} {mesh_type}")
            if max(row["E_abs_barycenter"] for row in selected) - min(
                    row["E_abs_barycenter"] for row in selected) > 1.0e-11:
                raise RuntimeError(f"error changed for {label} {mesh_type}")
            summary.append({
                "scale": label,
                "mesh_type": mesh_type,
                "uniform_level": level,
                "N": selected[0]["N"],
                "p": PARAMETERS["p"],
                "P": PARAMETERS["P"],
                "theta": PARAMETERS["theta"],
                "threads": PARAMETERS["threads"],
                "E_abs_barycenter": selected[0]["E_abs_barycenter"],
                "T_list_median_s": statistics.median(
                    row["T_list_s"] for row in selected),
                "T_eval_median_s": statistics.median(
                    row["T_eval_s"] for row in selected),
                "T_total_median_s": statistics.median(
                    row["T_total_s"] for row in selected),
                "T_total_min_s": min(row["T_total_s"] for row in selected),
                "T_total_max_s": max(row["T_total_s"] for row in selected),
                "mesh_sha256": selected[0]["mesh_sha256"],
            })
    write_csv(OUT / "summary.csv", summary)
    print(f"COMPLETE: {OUT / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
