#!/usr/bin/env python3
"""Run independent C++ validation without modifying historical results.

Python schedules and records experiments; all potential evaluations, including
the all-source direct reference, are performed by the C++ executables.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "article_error_validation"
THREADS = 128
SCALES = {"N10k": ("0.1", "0.01", 10496),
          "N100k": ("0.01", "0.001", 100432),
          "N800k": ("0.001", "0.0001", 823672)}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def csv_write(path, data):
    if data:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)


def number(text, label):
    match = re.search(re.escape(label) + r"\s*([-+\d.eE]+)", text)
    if not match or not math.isfinite(float(match[1])):
        raise RuntimeError("Missing/nonfinite: " + label)
    return float(match[1])


def base_environment():
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("ERROR_VALIDATION_", "H2O_VALIDATION_", "WATER10_VALIDATION_")):
            env.pop(key)
    env.update(
        OMP_NUM_THREADS=str(THREADS), OMP_DYNAMIC="FALSE",
        OMP_PROC_BIND="close", OMP_PLACES="cores",
        TREECODE_CPU_INTERACTIONS="CACHED", TREECODE_HNL_MAC_FAMILIES="0",
        TREECODE_HNL_SHARED_TARGETS="0",
        AFEPACK_TEMPLATE_PATH=":".join(
            "/usr/local/include/AFEPack/template/" + s + "/"
            for s in ("triangle", "twin_triangle", "tetrahedron",
                      "twin_tetrahedron", "four_tetrahedron")))
    return env


def execute(work, command, extra, dependencies):
    work.mkdir(parents=True, exist_ok=True)
    env = base_environment()
    env.update(extra)
    relevant = {k: v for k, v in env.items()
                if k.startswith(("OMP_", "TREECODE_", "ERROR_VALIDATION_",
                                 "H2O_VALIDATION_", "WATER10_",
                                 "AFEPACK_TEMPLATE_PATH"))}
    hashes = {name: digest(ROOT / name) for name in dependencies}
    key = dict(command=command, environment=relevant, hashes=hashes)
    metadata = work / "manifest.json"
    log = work / "console.log"
    if metadata.exists():
        saved = json.loads(metadata.read_text())
        if saved.get("key") != key or not saved.get("completed"):
            raise RuntimeError("Preserving incompatible/incomplete run: " + str(work))
        print("REUSE", work.name, flush=True)
        return log.read_text()
    dump(metadata, dict(key=key, completed=False, started_utc=stamp()))
    print("START", work.name, flush=True)
    with log.open("w") as f:
        result = subprocess.run(command, cwd=work, env=env, stdout=f,
                                stderr=subprocess.STDOUT, timeout=1800)
    if result.returncode:
        raise RuntimeError("C++ validation failed: " + str(log))
    if any(digest(ROOT / name) != value for name, value in hashes.items()):
        raise RuntimeError("Inputs changed during validation")
    text = log.read_text()
    if "HNL" not in text or int(number(text, "OpenMP threads actually used:")) != THREADS:
        raise RuntimeError("Unexpected algorithm or thread count")
    dump(metadata, dict(key=key, completed=True, finished_utc=stamp()))
    return text


GAUSS_DEPS = ["tree_error_validation", "compare_error_validation.cpp",
              "error_validation.hpp", "compare_exact.cpp", "D.mesh",
              "local_quadrature.hpp", "cartesian_multipole.hpp",
              "gaussian_test_problem.hpp", "hnl_candidates.hpp",
              "hnl_family_mac.hpp", "mac_geometry.hpp", "hnl_visualization.hpp"]


def gaussian_case(group, scale, p, order, theta, inner, outer, seed, direct):
    tol1, tol2, expected_n = SCALES[scale]
    name = f"{scale}_p{p}_P{order}_a{theta:g}_m{inner}_o{outer}_s{seed}"
    work = OUT / group / name
    command = [str(ROOT / "tree_error_validation"), str(ROOT / "D.mesh"), "0",
               str(p), str(order), str(theta), "1", "CPU", tol1, tol2, "RADIAL", "6", "0"]
    text = execute(work, command, dict(
        ERROR_VALIDATION_INNER=str(inner), ERROR_VALIDATION_OUTER=str(outer),
        ERROR_VALIDATION_SEED=str(seed), ERROR_VALIDATION_DIRECT=str(int(direct))),
        GAUSS_DEPS)
    if "Independent validation complete:" not in text:
        raise RuntimeError("Incomplete independent validation")
    result, = rows(work / "validation.csv")
    if int(result["N"]) != expected_n:
        raise RuntimeError("Unexpected validation mesh count")
    historical = ROOT / "article_hnl_study" / f"{scale}_p6_P4_theta0.5" / "work" / "D.dx"
    mesh_hash = digest(work / "D.dx")
    if mesh_hash != digest(historical):
        raise RuntimeError("Validation mesh differs from production mesh")
    for marker in ("HNL target sharing: OFF", "HNL MAC family packing: OFF",
                   "CPU interaction mode: CACHED", "construction failures=0"):
        if marker not in text:
            raise RuntimeError("Missing production policy: " + marker)
    result.update(scale=scale, mesh_sha256=mesh_hash,
                  target_file=str((work / "target_values.csv").relative_to(ROOT)),
                  log=str((work / "console.log").relative_to(ROOT)))
    dump(work / "validation_metadata.json", dict(
        mesh_sha256=mesh_hash,
        targets_sha256=digest(work / "target_values.csv"),
        validation_only=True, production_results_unchanged=True))
    print("DONE", name, "E=", result["E_total"],
          "fast=", result["E_fast"], "integration=", result["E_integration"], flush=True)
    return result


def components():
    settings = [(p, 4, .5, 7351, True) for p in (2, 4, 6, 8, 12)]
    settings += [(6, order, .5, 7351, False) for order in (2, 6, 8)]
    settings += [(6, 4, theta, 7351, False) for theta in (.2, .3, .7)]
    settings += [(6, 4, .5, 19477, True)]
    data = []
    for p, order, theta, seed, direct in settings:
        data.append(gaussian_case("components", "N100k", p, order, theta,
                                  12, 2, seed, direct))
        csv_write(OUT / "components_raw.csv", data)


def sampling():
    data = []
    for scale in SCALES:
        for inner in (16, 32, 64):
            for seed in (7351, 19477):
                data.append(gaussian_case("sampling", scale, 6, 4, .5,
                                          inner, 4, seed, False))
                csv_write(OUT / "sampling.csv", data)


def molecular():
    data = []
    cases = rows(ROOT / "single_h2o_hnl_refinement" / "summary.csv")
    for case in cases:
        for m in (64, 96):
            name = f"single_level{case['level']}_m{m}"
            work = OUT / "molecular" / name
            command = [str(ROOT / "tree_h2o"), str(ROOT / "D.mesh"), "2",
                       case["p"], case["P"], ".5", "1", "CPU", "SINGLE",
                       "QUADRATURE", case["tolerance"], "RADIAL", "CORE_ANALYTIC", ".25"]
            text = execute(work, command,
                           dict(H2O_VALIDATION_GRID=str(m), H2O_VALIDATION_FOCUSED="1"),
                           ["tree_h2o", "compare_h2o.cpp", "D.mesh",
                            "h2o_three_center_test_problem.hpp", "gaussian_test_problem.hpp",
                            "local_quadrature.hpp", "cartesian_multipole.hpp", "hnl_candidates.hpp"])
            n = int(re.search(r"Single all-atoms mesh: elements=(\d+)", text)[1])
            mesh = work / "h2o_all_atoms_afepack.dx"
            old_mesh = ROOT / case["mesh"]
            if n != int(case["N"]) or digest(mesh) != digest(old_mesh):
                raise RuntimeError("Single-H2O validation mesh changed")
            if abs(number(text,"Single-mesh L2 error:")-float(case["E_abs"])) > 1e-10:
                raise RuntimeError("Single-H2O production error changed")
            result = dict(model="single_h2o", level=case["level"], N=n,
                          p=case["p"], P=case["P"], theta=.5, grid=m,
                          targets=int(number(text,"Common validation targets:")),
                          E_relative=number(text,"Common validation relative L2 error:"),
                          validation_seconds=number(text,"Common validation time (excluded):"),
                          validation_threads=THREADS, mesh_sha256=digest(mesh),
                          log=str((work/"console.log").relative_to(ROOT)))
            data.append(result)
            csv_write(OUT / "molecular.csv", data)
            print("DONE", name, result["E_relative"], flush=True)

    for case in rows(ROOT / "water10_case" / "joint_refinement" / "summary.csv"):
        for m in (48, 64):
            name = f"ten_level{case['level']}_m{m}"
            work = OUT / "molecular" / name
            (work / "water10_case").mkdir(parents=True, exist_ok=True)
            command = [str(ROOT / "tree_water10"), str(ROOT / "D.mesh"), "1",
                       case["p"], case["P"], ".5", "1", "CPU",
                       str(float(case["tol2"])*10/3), case["tol2"], "RADIAL", "6", "0"]
            text = execute(work, command, dict(
                WATER10_VALIDATION_GRID=str(m), WATER10_PADDING="4",
                WATER10_ATOMS=str(ROOT/"water10_case"/"water10.centered.bohr.xyz")),
                ["tree_water10", "compare_water10.cpp", "compare_exact.cpp",
                 "water10_test_problem.hpp", "D.mesh", "local_quadrature.hpp",
                 "cartesian_multipole.hpp", "hnl_candidates.hpp", "mac_geometry.hpp",
                 "water10_case/water10.centered.bohr.xyz"])
            run_dir = Path(re.search(r"Output directory: (.+)", text)[1].strip().strip('"'))
            mesh = run_dir / "D.dx"
            n = int(number(text, "Exact-comparison leaf barycenters:"))
            if n != int(case["N"]) or digest(mesh) != digest(Path(case["mesh"])):
                raise RuntimeError("Ten-H2O validation mesh changed")
            if abs(number(text,"Relative L2 error against exact solution:")-float(case["E_bar"])) > 1e-10:
                raise RuntimeError("Ten-H2O production error changed")
            result = dict(model="ten_h2o", level=case["level"], N=n,
                          p=case["p"], P=case["P"], theta=.5, grid=m,
                          targets=int(number(text,"Common validation targets:")),
                          E_relative=number(text,"Common validation relative L2 error:"),
                          validation_seconds=number(text,"Common validation time (excluded):"),
                          validation_threads=THREADS, mesh_sha256=digest(mesh),
                          log=str((work/"console.log").relative_to(ROOT)))
            data.append(result)
            csv_write(OUT / "molecular.csv", data)
            print("DONE", name, result["E_relative"], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["components", "sampling", "molecular", "all"], default="all")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    snapshot = OUT / "source_snapshot"
    if not snapshot.exists():
        snapshot.mkdir()
        for name in set(GAUSS_DEPS + [
                "compare_h2o.cpp", "compare_water10.cpp", "water10_test_problem.hpp",
                "h2o_three_center_test_problem.hpp", "run_error_validation_study.py"]):
            if name.endswith((".cpp", ".hpp", ".py", ".mesh")):
                shutil.copy2(ROOT / name, snapshot / name)
    dump(OUT / "plan.json", dict(
        started_utc=stamp(), threads=THREADS, scope="independent validation only",
        direct_targets=4096, gaussian_sampling_inner=[16,32,64],
        gaussian_sampling_outer=4, gaussian_sampling_seeds=[7351,19477],
        single_h2o_focused=[64,96], ten_h2o_uniform=[48,64],
        component_meaning="tree-direct isolates acceleration of the SAME quadrature charges; "
        "direct-analytic contains integration and domain-tail error; neither "
        "separately isolates S_p or B_p",
        timing="all new timings excluded from production tables"))
    if args.stage in ("components", "all"): components()
    if args.stage in ("sampling", "all"): sampling()
    if args.stage in ("molecular", "all"): molecular()
    print("COMPLETE", args.stage, flush=True)


if __name__ == "__main__":
    main()
