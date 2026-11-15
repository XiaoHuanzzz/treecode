#!/usr/bin/env python3
"""Evaluate the displayed localized meshes and rerun their radial baselines."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from run_study import HERE, ROOT, TEMPLATES

CASES = ("sphere", "l_prism", "perforated_cube")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path,
                        default=HERE / "runs" / "localized_accuracy_20261108")
    parser.add_argument("--baseline", type=Path,
                        default=HERE / "runs" / "pilot_20261108")
    parser.add_argument("--mesh-directory", type=Path,
                        default=HERE / "runs" / "localized_meshes_20261108")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    for key in ("directory", "baseline", "mesh_directory"):
        setattr(args, key, getattr(args, key).resolve())
    executable = HERE / "compare_localized_domain"
    files = [HERE / name for name in ("compare_general_domain.cpp", "mesh_context.hpp",
             "direct_reference.hpp", "Makefile", "run_localized_accuracy.py")]
    files += [ROOT / name for name in ("compare_exact.cpp", "local_quadrature.hpp",
                                      "cartesian_multipole.hpp")]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in files}
    binary_hash = digest(executable)
    env = dict(os.environ, AFEPACK_TEMPLATE_PATH=TEMPLATES, OMP_NUM_THREADS=str(args.threads),
               OMP_DYNAMIC="FALSE", OMP_PLACES="cores", OMP_PROC_BIND="spread")
    args.directory.mkdir(parents=True, exist_ok=True)
    snapshot = args.directory / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for path in files:
        destination = snapshot / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and digest(destination) != digest(path):
            raise RuntimeError(f"Source changed: {path}; select a new --directory")
        if not destination.exists():
            shutil.copy2(path, destination)
    manifest_path = args.directory / "manifest.json"
    manifest = dict(source_sha256=hashes, binary_sha256=binary_hash, threads=args.threads,
                    baseline=str(args.baseline), mesh_directory=str(args.mesh_directory),
                    p=6, P=4, theta=.5, reference="uniform_L2_p16_d4",
                    environment={k:env[k] for k in ("AFEPACK_TEMPLATE_PATH", "OMP_NUM_THREADS",
                                 "OMP_DYNAMIC", "OMP_PLACES", "OMP_PROC_BIND")})
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if any(old.get(k) != v for k,v in manifest.items()):
            raise RuntimeError("Conflicting study manifest; use a new --directory")
    else:
        manifest.update(load_average_at_start=os.getloadavg(),
                        timing_note="Fresh single-run pairs on a shared server; not isolated benchmarks")
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    for case in args.cases:
        directory = args.directory / case
        directory.mkdir(exist_ok=True)
        base = args.baseline / case
        mesh = ROOT / "domain_meshes" / f"{case}.mesh"
        targets = base / "targets.csv"
        reference = base / "uniform_L2_p16_d4.values.csv"
        for name, monitor, tolerance, passes, expected in (
            ("localized", "localized", ".0002", "4", args.mesh_directory / case / "localized"),
            ("radial_baseline", "radial", ".001", "3", base / "adaptive_large"),
        ):
            prefix = directory / name
            command = [str(executable), "--mesh", str(mesh), "--mode", "solve", "--targets", str(targets),
                       "--p", "6", "--P", "4", "--theta", ".5", "--tol", tolerance,
                       "--passes", passes, "--monitor", monitor, "--monitor-width", "1.2",
                       "--monitor-floor", ".0001", "--same-direct", "--output", str(prefix)]
            request = dict(command=command, threads=args.threads, binary_sha256=binary_hash,
                           source_sha256=hashes, mesh_sha256=digest(mesh), targets_sha256=digest(targets),
                           reference_sha256=digest(reference), expected_mesh_sha256=digest(expected.with_suffix(".dx")))
            job_path = prefix.with_suffix(".job.json")
            if job_path.exists():
                job = json.loads(job_path.read_text())
                complete = all(prefix.with_suffix(suffix).exists() for suffix in
                               (".json", ".values.csv", ".direct.csv", ".dx"))
                if (all(job.get(k) == v for k,v in request.items()) and
                        job.get("returncode") == 0 and job.get("mesh_hash_match") and complete):
                    print("REUSE", case, name, flush=True)
                    continue
                raise RuntimeError(f"Conflicting or incomplete job: {job_path}")
            if any(prefix.with_suffix(suffix).exists() for suffix in (".json", ".values.csv", ".direct.csv", ".dx", ".log")):
                raise RuntimeError(f"Existing output without matching completed job: {prefix}")
            print("RUN", case, name, flush=True)
            request["load_average_at_start"] = os.getloadavg()
            start = time.perf_counter()
            with prefix.with_suffix(".log").open("x") as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=HERE)
            request.update(returncode=result.returncode, wall_s=time.perf_counter()-start)
            if result.returncode == 0:
                request["mesh_hash_match"] = digest(prefix.with_suffix(".dx")) == request["expected_mesh_sha256"]
                stats = json.loads(prefix.with_suffix(".json").read_text())
                expected_stats = json.loads(expected.with_suffix(".json").read_text())
                request["element_count_match"] = int(stats["N"]) == int(expected_stats["N"])
                request["thread_count_match"] = int(stats["threads"]) == args.threads
            job_path.write_text(json.dumps(request, indent=2)+"\n")
            if result.returncode:
                raise RuntimeError(f"Solver failed; inspect {prefix}.log")
            if not all(request[k] for k in ("mesh_hash_match", "element_count_match", "thread_count_match")):
                raise RuntimeError(f"Reproduction check failed: {job_path}")
            print("DONE", case, name, f"N={int(stats['N'])}",
                  f"evaluation={stats['evaluation_s']:.4f}s; exact mesh hash matched", flush=True)


if __name__ == "__main__":
    main()
