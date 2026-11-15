#!/usr/bin/env python3
"""Rerun every 128-thread manuscript timing study on one 64-core socket.

The existing data are never overwritten.  Raw logs, commands, hashes, machine
load, individual timings, and medians are written below
``experiment_results/article_64thread_20261111``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiment_results" / "article_64thread_20261111"
THREADS = 64
REPEATS = 3
TEMPLATES = ":".join([
    "/usr/local/include/AFEPack/template/triangle/",
    "/usr/local/include/AFEPack/template/twin_triangle/",
    "/usr/local/include/AFEPack/template/tetrahedron/",
    "/usr/local/include/AFEPack/template/twin_tetrahedron/",
    "/usr/local/include/AFEPack/template/four_tetrahedron/",
])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_float(text: str, label: str) -> float:
    match = re.search(re.escape(label) + r"\s*([0-9.eE+\-]+)", text)
    if match is None:
        raise RuntimeError(f"missing output field: {label}")
    return float(match.group(1))


def extract_int(text: str, label: str) -> int:
    return int(round(extract_float(text, label)))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "AFEPACK_TEMPLATE_PATH": TEMPLATES,
        "OMP_NUM_THREADS": str(THREADS),
        "OMP_DYNAMIC": "FALSE",
        "OMP_PLACES": "cores",
        "OMP_PROC_BIND": "close",
    })
    return env


def run_one(*, study: str, case: str, repeat: int, executable: Path,
            mesh: Path, refinement: int, p: int, P: int, theta: float,
            tol1: str, tol2: str, mesh_type: str) -> dict:
    work = OUT / "raw" / study / f"{case}_r{repeat}"
    work.mkdir(parents=True, exist_ok=True)
    command = [
        "/usr/bin/numactl", "--physcpubind=0-63", str(executable),
        str(mesh), str(refinement), str(p), str(P), f"{theta:.12g}",
        "1", "CPU", tol1, tol2,
    ]
    load_before = list(os.getloadavg())
    started = datetime.now(timezone.utc).isoformat()
    wall_start = time.monotonic()
    completed = subprocess.run(
        command, cwd=work, env=environment(), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    wall_s = time.monotonic() - wall_start
    load_after = list(os.getloadavg())
    log = work / "console.log"
    log.write_text(completed.stdout)
    if completed.returncode:
        raise RuntimeError(f"{study}/{case}/r{repeat} failed; see {log}")
    if "Interaction search: HNL" not in completed.stdout:
        raise RuntimeError(f"{study}/{case}/r{repeat} did not use HNL")
    if f"mesh={mesh_type}" not in completed.stdout:
        raise RuntimeError(f"{study}/{case}/r{repeat} mesh mode mismatch")
    actual_threads = extract_int(
        completed.stdout, "OpenMP threads actually used:")
    if actual_threads != THREADS:
        raise RuntimeError(
            f"{study}/{case}/r{repeat} used {actual_threads} threads")

    mesh_name = "D_uniform.dx" if mesh_type == "uniform" else "D.dx"
    generated_mesh = work / mesh_name
    mesh_hash = sha256(generated_mesh) if generated_mesh.exists() else ""
    if generated_mesh.exists():
        generated_mesh.unlink()

    row = {
        "study": study,
        "case": case,
        "repeat": repeat,
        "mesh_type": mesh_type,
        "N": extract_int(completed.stdout, "Exact-comparison leaf barycenters:"),
        "p": p,
        "P": P,
        "theta": theta,
        "tol1": tol1 if mesh_type == "adaptive" else "",
        "tol2": tol2 if mesh_type == "adaptive" else "",
        "threads": THREADS,
        "actual_threads": actual_threads,
        "T_list_s": extract_float(
            completed.stdout, "Filling neighbor_elements uses:"),
        "T_eval_s": extract_float(
            completed.stdout, "Potential evaluation time:"),
        "T_total_s": extract_float(
            completed.stdout, "Total execution time:"),
        "E_abs_barycenter": extract_float(
            completed.stdout, "L2 error against exact solution:"),
        "load1_before": load_before[0],
        "load5_before": load_before[1],
        "load15_before": load_before[2],
        "load1_after": load_after[0],
        "wall_s": wall_s,
        "started_utc": started,
        "executable_sha256": sha256(executable),
        "mesh_sha256": mesh_hash,
        "command": " ".join(command),
        "log": str(log.relative_to(ROOT)),
    }
    (work / "run.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    keys = []
    for row in rows:
        key = (row["study"], row["case"], row["mesh_type"])
        if key not in keys:
            keys.append(key)
    for study, case, mesh_type in keys:
        selected = [
            row for row in rows
            if (row["study"], row["case"], row["mesh_type"])
            == (study, case, mesh_type)
        ]
        if len(selected) != REPEATS:
            raise RuntimeError(f"expected {REPEATS} runs for {study}/{case}")
        for field in ("N", "p", "P", "theta", "E_abs_barycenter",
                      "actual_threads", "mesh_sha256"):
            if len({row[field] for row in selected}) != 1:
                raise RuntimeError(f"non-deterministic {field}: {study}/{case}")
        first = selected[0]
        summary.append({
            "study": study,
            "case": case,
            "mesh_type": mesh_type,
            "N": first["N"],
            "p": first["p"],
            "P": first["P"],
            "theta": first["theta"],
            "threads": first["actual_threads"],
            "E_abs_barycenter": first["E_abs_barycenter"],
            "T_list_median_s": statistics.median(
                row["T_list_s"] for row in selected),
            "T_eval_median_s": statistics.median(
                row["T_eval_s"] for row in selected),
            "T_total_median_s": statistics.median(
                row["T_total_s"] for row in selected),
            "T_total_min_s": min(row["T_total_s"] for row in selected),
            "T_total_max_s": max(row["T_total_s"] for row in selected),
            "max_load1": max(row["load1_before"] for row in selected),
            "executable_sha256": first["executable_sha256"],
            "mesh_sha256": first["mesh_sha256"],
        })
    return summary


def main() -> None:
    exact = ROOT / "tree_exact"
    uniform = ROOT / "tree_uniform"
    mesh = ROOT / "D.mesh"
    mesh20 = ROOT / "experiment_results" / "fixed_adaptive_uniform_20261109" / "D20.mesh"
    for path in (exact, uniform, mesh, mesh20):
        if not path.exists():
            raise RuntimeError(f"missing required input: {path}")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "replace all 128-thread manuscript production timings",
        "threads": THREADS,
        "cpu_binding": "numactl --physcpubind=0-63 (socket 0)",
        "openmp": {
            "OMP_NUM_THREADS": str(THREADS), "OMP_DYNAMIC": "FALSE",
            "OMP_PLACES": "cores", "OMP_PROC_BIND": "close",
        },
        "repeats": REPEATS,
        "tree_exact_sha256": sha256(exact),
        "tree_uniform_sha256": sha256(uniform),
        "compare_exact_cpp_sha256": sha256(ROOT / "compare_exact.cpp"),
        "D_mesh_sha256": sha256(mesh),
        "D20_mesh_sha256": sha256(mesh20),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    rows: list[dict] = []
    raw_csv = OUT / "runs.csv"

    def execute(spec: dict, repeat: int) -> None:
        row = run_one(repeat=repeat, **spec)
        rows.append(row)
        write_csv(raw_csv, rows)
        print(
            f"DONE {row['study']} {row['case']} r{repeat}: "
            f"N={row['N']} T={row['T_total_s']:.3f}s "
            f"E={row['E_abs_barycenter']:.6e}", flush=True)

    # Fixed-mesh parameter sweeps.  The baseline is run once and shared by all
    # three curves, exactly as in the manuscript.
    parameter_specs = []
    unique_parameters = [
        ("p2", 2, 4, 0.5), ("p4", 4, 4, 0.5),
        ("baseline", 6, 4, 0.5), ("p8", 8, 4, 0.5),
        ("p12", 12, 4, 0.5), ("P2", 6, 2, 0.5),
        ("P6", 6, 6, 0.5), ("P8", 6, 8, 0.5),
        ("theta02", 6, 4, 0.2), ("theta03", 6, 4, 0.3),
        ("theta07", 6, 4, 0.7),
    ]
    for case, p, P, theta in unique_parameters:
        parameter_specs.append(dict(
            study="parameters", case=case, executable=exact, mesh=mesh,
            refinement=2, p=p, P=P, theta=theta, tol1="0.01",
            tol2="0.001", mesh_type="adaptive"))

    # Mesh refinement; N=100656 is the baseline parameter case above.
    mesh_specs = []
    for case, tol2 in [
        ("N10160", "0.01"), ("N35920", "0.002"),
        ("N192888", "0.0003"), ("N423034", "0.0001"),
    ]:
        mesh_specs.append(dict(
            study="mesh_refinement", case=case, executable=exact, mesh=mesh,
            refinement=2, p=6, P=4, theta=0.5, tol1="0.01",
            tol2=tol2, mesh_type="adaptive"))

    # Fixed-parameter adaptive/uniform pairs on [-20,20]^3.
    paired_specs = []
    for scale, level, tol1, tol2 in [
        ("coarse", 3, "1.0", "0.14"),
        ("medium", 4, "0.1", "0.018"),
        ("fine", 5, "0.01", "0.0022"),
    ]:
        paired_specs.extend([
            dict(study="adaptive_uniform", case=f"{scale}_adaptive",
                 executable=exact, mesh=mesh20, refinement=0, p=6, P=4,
                 theta=0.5, tol1=tol1, tol2=tol2, mesh_type="adaptive"),
            dict(study="adaptive_uniform", case=f"{scale}_uniform",
                 executable=uniform, mesh=mesh20, refinement=level, p=6, P=4,
                 theta=0.5, tol1=tol1, tol2=tol2, mesh_type="uniform"),
        ])

    groups = [parameter_specs, mesh_specs, paired_specs]
    for repeat in range(1, REPEATS + 1):
        for specs in groups:
            order = specs if repeat == 1 else (
                list(reversed(specs)) if repeat == 2 else specs[::2] + specs[1::2])
            for spec in order:
                execute(spec, repeat)

    # Add the baseline runs to the mesh-refinement summary without rerunning.
    baseline = [row for row in rows if row["case"] == "baseline"]
    for row in baseline:
        copy = dict(row)
        copy["study"] = "mesh_refinement"
        copy["case"] = "N100656"
        rows.append(copy)
    write_csv(raw_csv, rows)
    summary = summarize(rows)
    write_csv(OUT / "summary.csv", summary)
    print(f"COMPLETE: {OUT / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
