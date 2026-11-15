#!/usr/bin/env python3
"""Collect HNL--MAC work counters on the article's adaptive mesh family."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


HERE = Path(__file__).resolve().parent
OUT = HERE / "experiment_results" / "bounded_work_adaptive_article_20261109"
EXE = HERE / "tree_exact"
STAGE2 = [1.0e-2, 2.0e-3, 1.0e-3, 3.0e-4, 2.0e-4]
RESULT_STEMS = ["mesh_0", "mesh_1", "mesh_2", "mesh_3", "mesh_4_t2e4"]
TEMPLATES = ":".join([
    "/usr/local/include/AFEPack/template/triangle/",
    "/usr/local/include/AFEPack/template/twin_triangle/",
    "/usr/local/include/AFEPack/template/tetrahedron/",
    "/usr/local/include/AFEPack/template/twin_tetrahedron/",
    "/usr/local/include/AFEPack/template/four_tetrahedron/",
])


def reusable(path: Path, tolerance: float) -> bool:
    if not path.exists():
        return False
    data = json.loads(path.read_text())
    return (int(data.get("p", -1)) == 6
            and int(data.get("P", -1)) == 4
            and float(data.get("theta", -1)) == 0.5
            and abs(float(data.get("tolerance_stage2", -1)) - tolerance)
            < 1.0e-14
            and "hnl_candidate_regions_avg" in data
            and "visited_nodes_avg" in data
            and "list_entries_avg" in data)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base_env = dict(os.environ, AFEPACK_TEMPLATE_PATH=TEMPLATES,
                    OMP_NUM_THREADS="16", OMP_DYNAMIC="FALSE",
                    OMP_PLACES="cores", OMP_PROC_BIND="close")
    for index, (tolerance, stem) in enumerate(zip(STAGE2, RESULT_STEMS)):
        result_path = OUT / f"{stem}.json"
        log_path = OUT / f"{stem}.log"
        job_path = OUT / f"{stem}.job.json"
        if reusable(result_path, tolerance):
            print("REUSE", index, tolerance, flush=True)
            continue
        if result_path.exists():
            raise RuntimeError(f"Conflicting result: {result_path}")
        command = [str(EXE), "D.mesh", "2", "6", "4", "0.5",
                   "0", "CPU", "0.01", f"{tolerance:.12g}"]
        env = dict(base_env,
                   TREECODE_LIST_DIAGNOSTICS_JSON=str(result_path))
        metadata = {"command": command, "threads": 16,
                    "load_average_before": list(os.getloadavg()),
                    "started_unix": time.time()}
        print("RUN", index, tolerance, flush=True)
        start = time.monotonic()
        with log_path.open("w") as log:
            process = subprocess.run(command, cwd=HERE, env=env,
                                     stdout=log, stderr=subprocess.STDOUT)
        metadata.update(returncode=process.returncode,
                        wall_s=time.monotonic() - start,
                        load_average_after=list(os.getloadavg()))
        job_path.write_text(json.dumps(metadata, indent=2) + "\n")
        if process.returncode:
            raise RuntimeError(f"Run failed: {log_path}")
        data = json.loads(result_path.read_text())
        print("DONE", index, tolerance, "N=", data["N"],
              f"{metadata['wall_s']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
