#!/usr/bin/env python3
"""Run HNL work-growth and repeated-density list-reuse experiments."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


HERE = Path(__file__).resolve().parent
OUT = HERE / "experiment_results" / "hnl_efficiency_20261109"
WORK_OUT = OUT / "work_growth"
REUSE_OUT = OUT / "reuse"
EXACT_EXE = HERE / "tree_exact"
REUSE_EXE = HERE / "tree_reuse"
REPEATS = 3
STAGE2 = [1.0e-2, 2.0e-3, 1.0e-3, 3.0e-4, 1.0e-4]
TEMPLATES = ":".join([
    "/usr/local/include/AFEPack/template/triangle/",
    "/usr/local/include/AFEPack/template/twin_triangle/",
    "/usr/local/include/AFEPack/template/tetrahedron/",
    "/usr/local/include/AFEPack/template/twin_tetrahedron/",
    "/usr/local/include/AFEPack/template/four_tetrahedron/",
])


def run_command(command, env, log_path, metadata_path):
    metadata = {
        "command": [str(item) for item in command],
        "load_average_before": list(os.getloadavg()),
        "started_unix": time.time(),
    }
    start = time.monotonic()
    with log_path.open("w") as stream:
        process = subprocess.run(
            command,
            cwd=HERE,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    metadata.update(
        returncode=process.returncode,
        wall_s=time.monotonic() - start,
        load_average_after=list(os.getloadavg()),
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    if process.returncode:
        raise RuntimeError(f"Run failed; see {log_path}")
    return metadata


def run_work_growth():
    WORK_OUT.mkdir(parents=True, exist_ok=True)
    base_env = dict(
        os.environ,
        AFEPACK_TEMPLATE_PATH=TEMPLATES,
        OMP_NUM_THREADS="16",
        OMP_DYNAMIC="FALSE",
        OMP_PLACES="cores",
        OMP_PROC_BIND="close",
    )
    orders = [
        list(range(len(STAGE2))),
        list(reversed(range(len(STAGE2)))),
        [2, 0, 4, 1, 3],
    ]
    for repeat, order in enumerate(orders, 1):
        for case_index in order:
            tolerance = STAGE2[case_index]
            stem = f"repeat_{repeat}_mesh_{case_index}"
            result_path = WORK_OUT / f"{stem}.json"
            if result_path.exists():
                print("REUSE WORK", repeat, case_index, flush=True)
                continue
            env = dict(
                base_env,
                TREECODE_LIST_DIAGNOSTICS_JSON=str(result_path),
            )
            command = [
                str(EXACT_EXE), "D.mesh", "2", "6", "4", "0.5",
                "0", "CPU", "0.01", f"{tolerance:.12g}",
            ]
            print("RUN WORK", repeat, case_index, tolerance, flush=True)
            metadata = run_command(
                command,
                env,
                WORK_OUT / f"{stem}.log",
                WORK_OUT / f"{stem}.job.json",
            )
            result = json.loads(result_path.read_text())
            print(
                "DONE WORK", repeat, case_index,
                "N=", result["N"],
                "T_list=", result["list_kernel_s"],
                f"wall={metadata['wall_s']:.2f}s",
                flush=True,
            )


def run_reuse():
    REUSE_OUT.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        AFEPACK_TEMPLATE_PATH=TEMPLATES,
        OMP_NUM_THREADS="64",
        OMP_DYNAMIC="FALSE",
        OMP_PLACES="cores",
        OMP_PROC_BIND="close",
    )
    for repeat in range(1, REPEATS + 1):
        output_path = REUSE_OUT / f"repeat_{repeat}.tsv"
        if output_path.exists():
            print("REUSE DENSITY", repeat, flush=True)
            continue
        command = [
            str(REUSE_EXE), "D.mesh", "2", "6", "4", "0.5", "1",
            "CPU", "20", "0.8", "0.5", str(output_path),
        ]
        print("RUN DENSITY", repeat, flush=True)
        metadata = run_command(
            command,
            env,
            REUSE_OUT / f"repeat_{repeat}.log",
            REUSE_OUT / f"repeat_{repeat}.job.json",
        )
        print("DONE DENSITY", repeat, f"wall={metadata['wall_s']:.2f}s", flush=True)


def main():
    if not EXACT_EXE.exists() or not REUSE_EXE.exists():
        raise RuntimeError("Build tree_exact and tree_reuse before running")
    OUT.mkdir(parents=True, exist_ok=True)
    run_work_growth()
    run_reuse()
    print("ALL EXPERIMENTS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
