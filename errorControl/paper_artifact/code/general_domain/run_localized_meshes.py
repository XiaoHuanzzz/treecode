#!/usr/bin/env python3
"""Generate three mesh-only illustrations; retain the numerical pilot unchanged."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from run_study import HERE, ROOT, TEMPLATES


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path,
                        default=HERE / "runs" / "localized_meshes_20261108")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--tol", type=float, default=.0002)
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--width", type=float, default=1.2)
    parser.add_argument("--floor", type=float, default=.0001)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    executable = HERE / "generate_localized_mesh"
    source_files = [HERE / name for name in ("generate_localized_mesh.cpp", "mesh_context.hpp")]
    source_files += [ROOT / name for name in ("compare_exact.cpp", "local_quadrature.hpp")]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in source_files}
    env = dict(os.environ, AFEPACK_TEMPLATE_PATH=TEMPLATES,
               OMP_NUM_THREADS=str(args.threads), OMP_DYNAMIC="FALSE",
               OMP_PLACES="cores", OMP_PROC_BIND="spread")
    for case in ("sphere", "l_prism", "perforated_cube"):
        directory = args.directory.resolve() / case
        directory.mkdir(parents=True, exist_ok=True)
        prefix = directory / "localized"
        mesh = ROOT / "domain_meshes" / f"{case}.mesh"
        command = [str(executable), "--mesh", str(mesh), "--output", str(prefix),
                   "--tol", str(args.tol), "--passes", str(args.passes),
                   "--width", str(args.width), "--floor", str(args.floor)]
        request = dict(command=command, threads=args.threads,
                       binary_sha256=digest(executable), mesh_sha256=digest(mesh),
                       source_sha256=hashes)
        job_path = prefix.with_suffix(".job.json")
        if job_path.exists():
            old = json.loads(job_path.read_text())
            if (all(old.get(k) == v for k, v in request.items()) and
                    old.get("returncode") == 0 and prefix.with_suffix(".json").exists() and
                    prefix.with_suffix(".dx").exists()):
                print("REUSE", case, flush=True)
                continue
            raise RuntimeError(f"Existing/conflicting job: {job_path}; use a new directory")
        if prefix.with_suffix(".json").exists() or prefix.with_suffix(".dx").exists():
            raise RuntimeError(f"Existing output without matching provenance: {prefix}")
        print("RUN", case, flush=True)
        start = time.perf_counter()
        with prefix.with_suffix(".log").open("x") as log:
            process = subprocess.run(command, env=env, cwd=HERE, stdout=log,
                                     stderr=subprocess.STDOUT)
        request.update(returncode=process.returncode, wall_s=time.perf_counter()-start)
        job_path.write_text(json.dumps(request, indent=2)+"\n")
        if process.returncode:
            raise RuntimeError(f"Mesh generation failed; see {prefix}.log")
        stats = json.loads(prefix.with_suffix(".json").read_text())
        print("DONE", case, f"N={int(stats['N'])}", flush=True)


if __name__ == "__main__":
    main()
