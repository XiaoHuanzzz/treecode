#!/usr/bin/env python3
"""Reproducible, resumable CPU general-domain pilot; no legacy outputs touched."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TEMPLATES = ":".join(
    str(Path("/usr/local/include/AFEPack/template") / name) + "/"
    for name in ("triangle", "twin_triangle", "tetrahedron", "twin_tetrahedron", "four_tetrahedron")
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=["sphere", "l_prism", "perforated_cube"])
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--directory", type=Path, default=HERE / "runs" / "pilot_20261108")
    parser.add_argument("--stages", nargs="+", default=["targets", "adaptive", "reference"])
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    binary = HERE / "compare_general_domain"
    env = dict(os.environ, AFEPACK_TEMPLATE_PATH=TEMPLATES,
               OMP_NUM_THREADS=str(args.threads), OMP_DYNAMIC="FALSE",
               OMP_PLACES="cores", OMP_PROC_BIND="spread")
    args.directory.mkdir(parents=True, exist_ok=True)
    snapshot = args.directory / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    sources = [HERE / name for name in (
        "compare_general_domain.cpp", "mesh_context.hpp", "direct_reference.hpp",
        "Makefile", "run_study.py")]
    sources += [ROOT / name for name in (
        "compare_exact.cpp", "local_quadrature.hpp", "cartesian_multipole.hpp")]
    hashes = {}
    for src in sources:
        digest = hashlib.sha256(src.read_bytes()).hexdigest()
        hashes[src.name] = digest
        dst = snapshot / src.name
        if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source changed since this study began: {src}; use a new directory")
        shutil.copy2(src, dst)
    hashes["binary"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    (args.directory / "manifest.json").write_text(json.dumps({
        "source_sha256": hashes, "threads": args.threads,
        "target_seed": 20261108, "volume_targets": 128, "core_targets": 32,
        "environment": {k:env[k] for k in ("AFEPACK_TEMPLATE_PATH", "OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_PLACES", "OMP_PROC_BIND")},
        "timing_note": "Pilot timings on shared server; not an isolated performance benchmark",
        "load_average_at_start": os.getloadavg(),
    }, indent=2) + "\n")

    def execute(name, command, sentinel):
        job_path = sentinel.parent / (name + ".job.json")
        requested = {"command": list(map(str, command)), "threads": args.threads,
                     "binary_sha256": hashes["binary"]}
        if sentinel.exists() and job_path.exists():
            previous = json.loads(job_path.read_text())
            if all(previous.get(k) == v for k,v in requested.items()) and previous.get("returncode") == 0:
                print(f"REUSE {name}", flush=True)
                return
            raise RuntimeError(f"Conflicting existing result {sentinel}")
        print(f"RUN {name}", flush=True)
        log = sentinel.parent / (name + ".log")
        start = time.time()
        with log.open("w") as stream:
            proc = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, cwd=HERE)
        requested.update(returncode=proc.returncode, wall_s=time.time()-start)
        job_path.write_text(json.dumps(requested, indent=2) + "\n")
        if proc.returncode:
            print(log.read_text()[-5000:], flush=True)
            raise RuntimeError(f"{name} failed; see {log}")
        if not sentinel.exists():
            raise RuntimeError(f"{name} did not produce {sentinel}")
        print(f"DONE {name} ({time.time()-start:.2f} s)", flush=True)

    for case in args.cases:
        mesh = ROOT / "domain_meshes" / (case + ".mesh")
        if not mesh.exists():
            raise FileNotFoundError(mesh)
        out = args.directory / case
        out.mkdir(exist_ok=True)
        shutil.copy2(mesh, out / "base.mesh")
        targets = out / "targets.csv"
        common = [str(binary), "--mesh", str(mesh)]
        if "targets" in args.stages:
            execute("targets", common + ["--mode", "targets", "--output", str(targets)], targets)
        if "adaptive" in args.stages:
            for label,tol in (("coarse", "0.03"), ("fine", "0.01")):
                name = "adaptive_" + label
                prefix = out / name
                execute(name, common + ["--mode", "solve", "--targets", str(targets),
                    "--p", "6", "--P", "4", "--theta", "0.5", "--tol", tol,
                    "--passes", "2", "--same-direct", "--output", str(prefix)], prefix.with_suffix(".json"))
        if "reference" in args.stages:
            for level,p in ((1,12), (2,8), (2,12)):
                name = f"uniform_L{level}_p{p}"
                prefix = out / name
                execute(name, common + ["--mode", "reference", "--targets", str(targets),
                    "--level", str(level), "--p", str(p), "--near-depth", "3",
                    "--output", str(prefix)], prefix.with_suffix(".json"))


if __name__ == "__main__":
    main()
