#!/usr/bin/env python3
"""Add the approximately 425k-leaf refinement case to the 64-thread study."""

import csv

import run_article_64thread_experiments as study


def read_rows(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    integer_fields = {"repeat", "N", "p", "P", "threads", "actual_threads"}
    float_fields = {
        "theta", "T_list_s", "T_eval_s", "T_total_s",
        "E_abs_barycenter", "load1_before", "load5_before",
        "load15_before", "load1_after", "wall_s",
    }
    for row in rows:
        for field in integer_fields:
            row[field] = int(row[field])
        for field in float_fields:
            row[field] = float(row[field])
    return rows


def main():
    runs_path = study.OUT / "runs.csv"
    rows = read_rows(runs_path)
    if any(row["case"] == "N425498" for row in rows):
        raise RuntimeError("N425498 extension already exists")
    spec = dict(
        study="mesh_refinement", case="N425498",
        executable=study.ROOT / "tree_exact", mesh=study.ROOT / "D.mesh",
        refinement=2, p=6, P=4, theta=0.5, tol1="0.01",
        tol2="0.0002", mesh_type="adaptive",
    )
    for repeat in range(1, study.REPEATS + 1):
        row = study.run_one(repeat=repeat, **spec)
        rows.append(row)
        study.write_csv(runs_path, rows)
        print(
            f"DONE mesh_refinement N425498 r{repeat}: "
            f"N={row['N']} T={row['T_total_s']:.3f}s "
            f"E={row['E_abs_barycenter']:.6e}", flush=True)
    study.write_csv(study.OUT / "summary.csv", study.summarize(rows))


if __name__ == "__main__":
    main()
