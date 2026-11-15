#!/usr/bin/env python3
"""One further target refinement for the narrow single-H2O manufactured peaks."""
import re
from run_error_validation_study import ROOT, OUT, THREADS, execute, number, digest, csv_write, rows

data = []
for case in rows(ROOT / "single_h2o_hnl_refinement" / "summary.csv"):
    m = 128
    work = OUT / "molecular" / f"single_level{case['level']}_m{m}"
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
    if n != int(case["N"]) or digest(mesh) != digest(ROOT / case["mesh"]):
        raise RuntimeError("Single-H2O validation mesh changed")
    if abs(number(text, "Single-mesh L2 error:") - float(case["E_abs"])) > 1e-10:
        raise RuntimeError("Single-H2O production error changed")
    data.append(dict(model="single_h2o", level=case["level"], N=n,
                     p=case["p"], P=case["P"], theta=.5, grid=m,
                     targets=int(number(text, "Common validation targets:")),
                     E_relative=number(text, "Common validation relative L2 error:"),
                     validation_seconds=number(text, "Common validation time (excluded):"),
                     validation_threads=THREADS, mesh_sha256=digest(mesh),
                     log=str((work / "console.log").relative_to(ROOT))))
    csv_write(OUT / "molecular_fine.csv", data)
    print("DONE", work.name, data[-1]["E_relative"], flush=True)
