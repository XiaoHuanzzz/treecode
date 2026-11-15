#!/usr/bin/env python3
"""Matched HNL versus root-recursive MAC ablation on adaptive meshes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


HERE = Path(__file__).resolve().parent
OUT = HERE / "experiment_results" / "hnl_root_ablation_20261110"
LIST_OUT = OUT / "list_runs"
EVAL_OUT = OUT / "evaluation_runs"
METHODS = {
    "HNL": HERE / "tree_exact",
    "ROOT": HERE / "tree_exact_root",
}
STAGE2 = [1.0e-2, 2.0e-3, 1.0e-3, 3.0e-4, 1.0e-4]
REPEATS = 3
THREADS = 16
TEMPLATES = ":".join([
    "/usr/local/include/AFEPack/template/triangle/",
    "/usr/local/include/AFEPack/template/twin_triangle/",
    "/usr/local/include/AFEPack/template/tetrahedron/",
    "/usr/local/include/AFEPack/template/twin_tetrahedron/",
    "/usr/local/include/AFEPack/template/four_tetrahedron/",
])


def base_environment() -> dict[str, str]:
    return dict(
        os.environ,
        AFEPACK_TEMPLATE_PATH=TEMPLATES,
        OMP_NUM_THREADS=str(THREADS),
        OMP_DYNAMIC="FALSE",
        OMP_PLACES="cores",
        OMP_PROC_BIND="close",
    )


def execute(command: list[str], env: dict[str, str], log: Path, job: Path) -> None:
    metadata = {
        "command": command,
        "load_average_before": list(os.getloadavg()),
        "started_unix": time.time(),
    }
    start = time.monotonic()
    with log.open("w") as stream:
        process = subprocess.run(
            command, cwd=HERE, env=env, stdout=stream,
            stderr=subprocess.STDOUT,
        )
    metadata.update(
        returncode=process.returncode,
        wall_s=time.monotonic() - start,
        load_average_after=list(os.getloadavg()),
    )
    job.write_text(json.dumps(metadata, indent=2) + "\n")
    if process.returncode:
        raise RuntimeError(f"run failed: {log}")


def run_list_ablation() -> None:
    LIST_OUT.mkdir(parents=True, exist_ok=True)
    mesh_orders = [
        list(range(len(STAGE2))),
        list(reversed(range(len(STAGE2)))),
        [2, 0, 4, 1, 3],
    ]
    for repeat, mesh_order in enumerate(mesh_orders, 1):
        for mesh_index in mesh_order:
            tolerance = STAGE2[mesh_index]
            method_order = ["HNL", "ROOT"]
            if (repeat + mesh_index) % 2:
                method_order.reverse()
            for method in method_order:
                stem = f"repeat_{repeat}_mesh_{mesh_index}_{method.lower()}"
                result = LIST_OUT / f"{stem}.json"
                if result.exists():
                    print("REUSE", stem, flush=True)
                    continue
                env = dict(
                    base_environment(),
                    TREECODE_LIST_DIAGNOSTICS_JSON=str(result),
                )
                command = [
                    str(METHODS[method]), "D.mesh", "2", "6", "4", "0.5",
                    "0", "CPU", "0.01", f"{tolerance:.12g}",
                ]
                print("RUN", stem, "tol=", tolerance, flush=True)
                execute(
                    command, env, LIST_OUT / f"{stem}.log",
                    LIST_OUT / f"{stem}.job.json",
                )
                data = json.loads(result.read_text())
                print(
                    "DONE", stem, "N=", data["N"],
                    "T=", data["list_total_s"],
                    "visits=", data["visited_nodes_avg"], flush=True,
                )


def run_evaluation_check() -> None:
    """Compare final potentials/errors on the representative 100k mesh."""
    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    for repeat in range(1, REPEATS + 1):
        order = ["HNL", "ROOT"] if repeat % 2 else ["ROOT", "HNL"]
        for method in order:
            stem = f"repeat_{repeat}_{method.lower()}"
            log = EVAL_OUT / f"{stem}.log"
            if log.exists():
                print("REUSE EVAL", stem, flush=True)
                continue
            command = [
                str(METHODS[method]), "D.mesh", "2", "6", "4", "0.5",
                "1", "CPU", "0.01", "0.001",
            ]
            print("RUN EVAL", stem, flush=True)
            execute(
                command, base_environment(), log,
                EVAL_OUT / f"{stem}.job.json",
            )
            print("DONE EVAL", stem, flush=True)


def main() -> None:
    for method, executable in METHODS.items():
        if not executable.exists():
            raise RuntimeError(f"missing {method} executable: {executable}")
    OUT.mkdir(parents=True, exist_ok=True)
    run_list_ablation()
    run_evaluation_check()
    print("HNL/ROOT ABLATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
