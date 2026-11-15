#!/usr/bin/env python3
"""Additional adaptive levels; preserves the initial pilot and its provenance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from run_study import HERE, ROOT, TEMPLATES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=HERE / "runs" / "pilot_20261108")
    parser.add_argument("--cases", nargs="+", default=["sphere", "l_prism", "perforated_cube"])
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    env = dict(os.environ, AFEPACK_TEMPLATE_PATH=TEMPLATES, OMP_NUM_THREADS=str(args.threads),
               OMP_DYNAMIC="FALSE", OMP_PLACES="cores", OMP_PROC_BIND="spread")
    binary = HERE / "compare_general_domain"
    binary_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
    for case in args.cases:
        out = args.directory / case
        for label,tolerance in (("medium", "0.003"), ("large", "0.001")):
            name = "adaptive_" + label
            prefix = out / name
            command = [str(binary), "--mesh", str(ROOT / "domain_meshes" / (case + ".mesh")),
                       "--mode", "solve", "--targets", str(out / "targets.csv"), "--p", "6",
                       "--P", "4", "--theta", "0.5", "--tol", tolerance, "--passes", "3",
                       "--same-direct", "--output", str(prefix)]
            meta_path = prefix.with_suffix(".job.json")
            requested = dict(command=command, threads=args.threads, binary_sha256=binary_hash)
            if prefix.with_suffix(".json").exists() and meta_path.exists():
                prior = json.loads(meta_path.read_text())
                if all(prior.get(k) == v for k,v in requested.items()) and prior.get("returncode") == 0:
                    print("REUSE", case, name, flush=True)
                    continue
                raise RuntimeError(f"Conflicting existing data {prefix}")
            print("RUN", case, name, flush=True)
            start = time.time()
            with prefix.with_suffix(".log").open("w") as log:
                proc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=HERE)
            requested.update(returncode=proc.returncode, wall_s=time.time()-start,
                             load_average=os.getloadavg())
            meta_path.write_text(json.dumps(requested, indent=2) + "\n")
            if proc.returncode:
                raise RuntimeError(f"Run failed: {prefix}.log")
            print("DONE", case, name, f"{time.time()-start:.2f} s", flush=True)


if __name__ == "__main__":
    main()
