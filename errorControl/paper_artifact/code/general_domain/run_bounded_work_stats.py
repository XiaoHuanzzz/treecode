#!/usr/bin/env python3
"""Collect list-construction work statistics on the article mesh sequences."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
TARGETS = HERE / "runs" / "irregular_scaling_20261108"
OUT = HERE / "runs" / "bounded_work_p2P4_20261109"
EXE = HERE / "compare_radial_test_domain"
TEMPLATES = ":".join([
    "/usr/local/include/AFEPack/template/triangle/",
    "/usr/local/include/AFEPack/template/twin_triangle/",
    "/usr/local/include/AFEPack/template/tetrahedron/",
    "/usr/local/include/AFEPack/template/twin_tetrahedron/",
    "/usr/local/include/AFEPack/template/four_tetrahedron/",
])


def reusable(prefix: Path, level: int) -> bool:
    path = prefix.with_suffix(".json")
    if not path.exists():
        return False
    data = json.loads(path.read_text())
    required = ("visited_nodes_avg", "hnl_candidate_regions_avg",
                "hnl_raw_candidate_visits_avg", "list_entries_avg")
    return (int(data.get("level", -1)) == level
            and int(data.get("p", -1)) == 2 and int(data.get("P", -1)) == 4
            and float(data.get("theta", -1)) == 0.5
            and int(data.get("lists_only", 0)) == 1
            and all(name in data for name in required))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, AFEPACK_TEMPLATE_PATH=TEMPLATES,
               OMP_NUM_THREADS="16", OMP_DYNAMIC="FALSE",
               OMP_PLACES="cores", OMP_PROC_BIND="close")
    for case in ("sphere", "l_prism", "perforated_cube"):
        folder = OUT / case
        folder.mkdir(parents=True, exist_ok=True)
        for level in range(4):
            prefix = folder / f"L{level}"
            if reusable(prefix, level):
                print("REUSE", case, level, flush=True)
                continue
            if prefix.with_suffix(".json").exists():
                raise RuntimeError(f"Conflicting diagnostic output: {prefix}")
            command = [
                str(EXE), "--mode", "solve", "--uniform", "--lists-only",
                "--mesh", str(WORKSPACE / "radial_shell_meshes" / "scaling"
                              / f"{case}.mesh"),
                "--level", str(level), "--p", "2", "--P", "4",
                "--theta", "0.5", "--targets", str(TARGETS / case / "targets.csv"),
                "--leaf-cap", "1000000", "--output", str(prefix),
            ]
            metadata = {"command": command, "threads": 16,
                        "load_average_before": list(os.getloadavg()),
                        "started_unix": time.time()}
            print("RUN", case, level, flush=True)
            start = time.monotonic()
            with prefix.with_suffix(".log").open("w") as log:
                result = subprocess.run(command, cwd=WORKSPACE, env=env,
                                        stdout=log, stderr=subprocess.STDOUT)
            metadata.update(returncode=result.returncode,
                            wall_s=time.monotonic() - start,
                            load_average_after=list(os.getloadavg()))
            prefix.with_suffix(".job.json").write_text(
                json.dumps(metadata, indent=2) + "\n")
            if result.returncode:
                raise RuntimeError(f"Diagnostic failed: {prefix.with_suffix('.log')}")
            print("DONE", case, level, f"{metadata['wall_s']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
