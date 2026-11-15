#!/usr/bin/env python3
"""Validate previously selected decamer configurations on finer targets."""
import re
from pathlib import Path
from run_error_validation_study import ROOT, OUT, THREADS, execute, number, digest, csv_write

data=[]
for tol,n,mesh in [
        (.05,7164,"water10_case/run_1792320680328677807/D.dx"),
        (.01,22298,"water10_case/run_1791966962061750350/D.dx")]:
    for m in (48,64):
        work=OUT/"selected"/f"N{n}_theta0.7_m{m}"
        (work/"water10_case").mkdir(parents=True,exist_ok=True)
        command=[str(ROOT/"tree_water10"),str(ROOT/"D.mesh"),"1","12","6",".7",
                 "1","CPU",str(tol*10/3),str(tol),"RADIAL","6","0"]
        text=execute(work,command,dict(
            WATER10_VALIDATION_GRID=str(m),WATER10_PADDING="4",
            WATER10_ATOMS=str(ROOT/"water10_case"/"water10.centered.bohr.xyz")),
            ["tree_water10","compare_water10.cpp","compare_exact.cpp","water10_test_problem.hpp",
             "D.mesh","local_quadrature.hpp","cartesian_multipole.hpp","hnl_candidates.hpp",
             "mac_geometry.hpp","water10_case/water10.centered.bohr.xyz"])
        run=Path(re.search(r"Output directory: (.+)",text)[1].strip().strip('"'))
        if (int(number(text,"Exact-comparison leaf barycenters:"))!=n
                or digest(run/"D.dx")!=digest(ROOT/mesh)):
            raise RuntimeError("Selected-case mesh changed")
        row=dict(N=n,p=12,P=6,theta=.7,tol2=tol,grid=m,
                 targets=int(number(text,"Common validation targets:")),
                 E_relative=number(text,"Common validation relative L2 error:"),
                 validation_seconds=number(text,"Common validation time (excluded):"),
                 validation_threads=THREADS,mesh_sha256=digest(run/"D.dx"),
                 log=str((work/"console.log").relative_to(ROOT)))
        data.append(row)
        csv_write(OUT/"selected.csv",data)
        print("DONE",work.name,row["E_relative"],flush=True)
