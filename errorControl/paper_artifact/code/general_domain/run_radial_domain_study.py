#!/usr/bin/env python3
"""Run the radial-source, radial-indicator general-domain study."""
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


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path,
                        default=HERE / "runs" / "radial_domain_20261108")
    parser.add_argument("--targets-directory", type=Path,
                        default=HERE / "runs" / "pilot_20261108")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    args.directory = args.directory.resolve()
    args.targets_directory = args.targets_directory.resolve()
    executable = HERE / "compare_radial_test_domain"
    sources = [HERE / name for name in (
        "compare_radial_domain.cpp", "compare_general_domain.cpp", "mesh_context.hpp",
        "direct_reference.hpp", "run_radial_domain_study.py", "Makefile")]
    sources += [ROOT / name for name in
                ("compare_exact.cpp", "local_quadrature.hpp", "cartesian_multipole.hpp")]
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sources}
    binary_hash = sha256(executable)
    env = dict(os.environ, AFEPACK_TEMPLATE_PATH=TEMPLATES,
               OMP_NUM_THREADS=str(args.threads), OMP_DYNAMIC="FALSE",
               OMP_PLACES="cores", OMP_PROC_BIND="spread")
    args.directory.mkdir(parents=True, exist_ok=True)
    snapshot = args.directory / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for source in sources:
        destination = snapshot / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha256(destination) != sha256(source):
            raise RuntimeError(f"Source changed; use a new directory: {source}")
        if not destination.exists():
            shutil.copy2(source, destination)
    manifest = dict(
        source="exp(-4*|x-(-2,-2,0)|^2), restricted to each domain",
        indicator="integral_K [1+(gamma-1)r^2/(r^2+R^2)]^(-3) dx",
        center=[-2, -2, 0], width=1.5, gamma=4.0, tolerance=0.0002,
        passes=4, p=6, P=4, theta=0.5, threads=args.threads,
        target_groups={"volume": 128, "core": 32},
        references=["uniform_L2_p12_d3", "uniform_L2_p16_d4"],
        binary_sha256=binary_hash, source_sha256=source_hashes,
        environment={k: env[k] for k in
                     ("AFEPACK_TEMPLATE_PATH", "OMP_NUM_THREADS", "OMP_DYNAMIC",
                      "OMP_PLACES", "OMP_PROC_BIND")})
    manifest_path = args.directory / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if any(old.get(key) != value for key, value in manifest.items()):
            raise RuntimeError("Conflicting manifest; use a new --directory")
    else:
        manifest.update(load_average_at_start=os.getloadavg(),
                        timing_note="Single runs on a shared server")
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")

    def execute(case, name, command, prefix, inputs):
        job_path = prefix.with_suffix(".job.json")
        request = dict(command=list(map(str, command)), threads=args.threads,
                       binary_sha256=binary_hash,
                       input_sha256={str(path): sha256(path) for path in inputs})
        outputs = [prefix.with_suffix(suffix) for suffix in (".json", ".values.csv", ".dx")]
        if name == "adaptive_radial_graded":
            outputs.append(prefix.with_suffix(".direct.csv"))
        if job_path.exists():
            job = json.loads(job_path.read_text())
            if (all(job.get(key) == value for key, value in request.items()) and
                    job.get("returncode") == 0 and all(path.exists() for path in outputs)):
                print("REUSE", case, name, flush=True)
                return
            raise RuntimeError(f"Conflicting/incomplete job: {job_path}")
        if any(path.exists() for path in outputs+[prefix.with_suffix(".log")]):
            raise RuntimeError(f"Output exists without a matching job: {prefix}")
        print("RUN", case, name, flush=True)
        request["load_average_at_start"] = os.getloadavg()
        start = time.perf_counter()
        with prefix.with_suffix(".log").open("x") as log:
            process = subprocess.run(command, cwd=HERE, env=env, stdout=log,
                                     stderr=subprocess.STDOUT)
        request.update(returncode=process.returncode, wall_s=time.perf_counter()-start)
        if process.returncode == 0:
            stats = json.loads(prefix.with_suffix(".json").read_text())
            request["actual_N"] = int(stats["N"])
            request["actual_threads"] = int(stats["threads"])
            request["source_verified"] = stats["source"] == \
                "exp(-4*((x+2)^2+(y+2)^2+z^2)), truncated to the mesh domain"
            request["monitor_verified"] = (name != "adaptive_radial_graded" or
                (stats["monitor"] == "radial_graded" and
                 abs(stats["monitor_width"]-1.5) < 1e-15 and
                 abs(stats["monitor_gamma"]-4.0) < 1e-15))
        job_path.write_text(json.dumps(request, indent=2)+"\n")
        if process.returncode:
            raise RuntimeError(f"Failed: {prefix.with_suffix('.log')}")
        if request["actual_threads"] != args.threads or not request["source_verified"] or not request["monitor_verified"]:
            raise RuntimeError(f"Runtime provenance check failed: {job_path}")
        print("DONE", case, name, f"N={request['actual_N']}", flush=True)

    for case in args.cases:
        folder = args.directory / case
        folder.mkdir(exist_ok=True)
        mesh = ROOT / "domain_meshes" / f"{case}.mesh"
        targets = args.targets_directory / case / "targets.csv"
        common = [str(executable), "--mesh", str(mesh), "--targets", str(targets),
                  "--monitor", "radial_graded", "--monitor-width", "1.5",
                  "--monitor-gamma", "4", "--monitor-floor", "0"]
        adaptive = folder / "adaptive_radial_graded"
        execute(case, adaptive.name,
                common+["--mode", "solve", "--p", "6", "--P", "4", "--theta", ".5",
                        "--tol", ".0002", "--passes", "4", "--same-direct",
                        "--output", str(adaptive)], adaptive, [mesh, targets])
        for name, p, depth in (("uniform_L2_p12_d3", "12", "3"),
                               ("uniform_L2_p16_d4", "16", "4")):
            prefix = folder / name
            execute(case, name,
                    common+["--mode", "reference", "--uniform", "--level", "2",
                            "--p", p, "--P", "4", "--theta", ".5",
                            "--near-depth", depth, "--output", str(prefix)],
                    prefix, [mesh, targets])


if __name__ == "__main__":
    main()
