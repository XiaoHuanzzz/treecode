#!/usr/bin/env python3
"""Run repeated-density list-reuse experiments with measured rebuild controls."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


HERE = Path(__file__).resolve().parent
OUT = HERE / "experiment_results" / "list_reuse_20261109"
EXE = HERE / "tree_reuse"
REPEATS = 3
STEPS = 20
DECAY = 0.8
INITIAL_AMPLITUDE = 0.5
THREADS = 64
TEMPLATES = ":".join([
    "/usr/local/include/AFEPack/template/triangle/",
    "/usr/local/include/AFEPack/template/twin_triangle/",
    "/usr/local/include/AFEPack/template/tetrahedron/",
    "/usr/local/include/AFEPack/template/twin_tetrahedron/",
    "/usr/local/include/AFEPack/template/four_tetrahedron/",
])


def complete(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text()
    data_rows = [line for line in text.splitlines()
                 if line and not line.startswith("#")]
    return (len(data_rows) == STEPS + 1
            and "# measured_rebuild_sum_s\t" in text
            and "# rebuild_every_step_measured_time_s\t" in text)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, AFEPACK_TEMPLATE_PATH=TEMPLATES,
               OMP_NUM_THREADS=str(THREADS), OMP_DYNAMIC="FALSE",
               OMP_PLACES="cores", OMP_PROC_BIND="close")
    for repeat in range(1, REPEATS + 1):
        tsv = OUT / f"repeat_{repeat}.tsv"
        log = OUT / f"repeat_{repeat}.log"
        job = OUT / f"repeat_{repeat}.job.json"
        if complete(tsv):
            print("REUSE", repeat, flush=True)
            continue
        if tsv.exists():
            raise RuntimeError(f"Conflicting incomplete result: {tsv}")
        command = [str(EXE), "D.mesh", "2", "6", "4", "0.5", "1",
                   "CPU", str(STEPS), str(DECAY), str(INITIAL_AMPLITUDE),
                   str(tsv)]
        metadata = {"command": command, "threads": THREADS,
                    "load_average_before": list(os.getloadavg()),
                    "started_unix": time.time()}
        print("RUN", repeat, flush=True)
        start = time.monotonic()
        with log.open("w") as stream:
            process = subprocess.run(command, cwd=HERE, env=env,
                                     stdout=stream, stderr=subprocess.STDOUT)
        metadata.update(returncode=process.returncode,
                        wall_s=time.monotonic() - start,
                        load_average_after=list(os.getloadavg()))
        job.write_text(json.dumps(metadata, indent=2) + "\n")
        if process.returncode:
            raise RuntimeError(f"Run failed: {log}")
        if not complete(tsv):
            raise RuntimeError(f"Incomplete result: {tsv}")
        print("DONE", repeat, f"{metadata['wall_s']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
