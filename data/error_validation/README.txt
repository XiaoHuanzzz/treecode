Independent error validation for article.tex
============================================

Purpose
-------
Separate fast summation from direct-quadrature error, and test sensitivity
of analytical-reference errors to the validation targets. This is not a
new production-timing benchmark or a certified continuous-L2 computation.

All potential calculations are in C++. Python schedules runs, verifies
provenance, summarizes results, and plots them. No historical results are
deleted. article_before_validation.tex/pdf preserve the previous manuscript.

Reproduce from /home/hxiao/treecode/errorControl
----------------------------------------------
make error-validation h2o water10
python3 run_error_validation_study.py
python3 run_selected_error_validation.py
python3 run_fine_h2o_validation.py
MPLCONFIGDIR=/tmp/article-error-mpl python3 summarize_error_validation_study.py

The runner reuses completed runs only if command, environment, and input
hashes match. It refuses to overwrite incomplete or incompatible records.
For a deliberately different study, use a different output directory.
The scripts set 128 threads with dynamic OpenMP teams disabled, cached
per-leaf HNL -> MAC lists, and no shared-target/family-packing optimization.

Studies
-------
components_raw.csv / components.csv:
  12 diagnostic runs on the N=100432 mesh; six include an all-source
  direct reference. Direct and tree potentials use identical source
  quadrature and singular target-element rules. Tree-only configurations
  reuse a direct reference only after exact target/weight agreement checks.
  The 4096-point target set is for paired diagnostics, not a converged norm.
  A second predetermined seed checks the baseline.

sampling.csv / sampling_summary.csv:
  18 runs: three source meshes, three independent validation resolutions,
  two seeds. The finest resolution contains 373248 points per seed.
  Pooling halves each target weight, preserving total volume 1000.
  Seed ranges are sensitivity checks, not confidence intervals.

molecular.csv / molecular_fine.csv:
  20 runs: single-H2O at inner resolutions 64, 96, 128 (four source meshes),
  ten-H2O at uniform resolutions 48, 64 (four source meshes).
  Source-mesh hashes and production barycenter errors are checked against
  historical HNL production results. The finer single-H2O follow-up was
  added after the first two resolutions showed substantial sensitivity.

selected.csv:
  Four runs revalidate two previously selected ten-H2O configurations at
  resolutions 48 and 64. These candidates were not chosen using these
  finer target sets. Report max(E_48,E_64), not the more favorable value.

There are 54 new validation runs in total. New validation timings are
excluded from article production times. The original single-center timing
tables use 128 threads; molecular production tables retain four threads.
The additional validation runs use 128 threads. Do not infer speedup from
different thread counts in these separate tasks.

Regression checks
-----------------
make quadrature-test multipole-test hnl-test mac-geometry-test
make exact

The quadrature test covers algebraic accuracies through 20 on tetrahedron,
twin, and four-tetrahedron elements. The multipole test covers orders
through 20. The HNL test checks disjoint source coverage and agreement
with the original layer constructor; the MAC test checks geometric
distance and threshold decisions. These are separate from error-norm
validation and are not included in production timings.

Evidence
--------
Each run retains console.log and manifest.json with command, environment,
binary/input hashes, and completion status. Gaussian runs also save all
per-target coordinates, weights, tree values, analytical values, and direct
values when requested. Source meshes are checked against production files.
The source_snapshot directory records the validation implementation.

report.txt contains numerical summaries and limitations. Plots are saved
as PDF and PNG. Historical sparse-target errors remain in their original
directories and are not silently replaced there.

Implementation scope
--------------------
compare_error_validation.cpp enables two compile-time hooks in
compare_exact.cpp. The production quadrature, singular integration, HNL,
MAC, moment, and potential kernels are unchanged by this validation task.
The validation executable exits before production-result logging.
Molecular validation uses existing optional validation paths, in isolated
working directories. Root results.txt and results_h2o.txt are not modified.
