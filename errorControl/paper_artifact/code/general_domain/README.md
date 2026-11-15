# General-domain Hartree experiments

This is a separate C++ experiment for the sphere, L-shaped prism, and
perforated cube in `../domain_meshes/`. It reuses the current CPU
HNL → MAC → near-field integration / far-field multipole evaluation from
`../compare_exact.cpp`. The density substitution is restricted to this
translation unit. The existing single-center model, molecular models,
results files, and `article.tex` are not changed by this experiment.

## Relation to Cui's experiments

Cui and Yang's accessible Poisson-treecode preprint uses leaf barycenters as
targets. Section 4.2 compares its treecode with direct integration using the
same sixth-degree Gaussian quadrature to separate treecode error from domain
truncation and discretization effects. Its mesh-refinement experiment uses
uniform tetrahedral meshes in a cube; its “p-adaptive” method adapts the
multipole order, not the spatial mesh. See
[Cui and Yang, Sections 3.2 and 4](https://arxiv.org/html/2412.20434v1#S4).

The earlier
[Cui, Yang, Wu, and Hu general-domain paper](https://doi.org/10.1007/s11075-024-01888-8)
is directly relevant to unstructured-domain treecodes, but its experimental
body was not accessible during this check. We therefore do not claim to
reproduce its exact domains, functions, or reference settings. The present
adaptive-domain study and converged uniform-mesh reference are an extension
inspired by this validation approach.

## Problem and domains

For each mesh domain D, compute

```text
rho(x,y,z) = exp(-(x+2)^2 - 2*(y+2)^2 - 3*z^2)
V_D(x)    = integral over D of rho(y)/|x-y| dy
```

The density is positive and centered at `(-2,-2,0)`. Its definition is the
same in all three cases and is not renormalized separately for each shape.
This is a prescribed density model, not an SCF-derived molecular density.
The source is restricted to D: density outside D is not part of the modeled
integral. In particular, the cavity is genuinely excluded from the source.
No full-space analytical Gaussian potential is used as an exact answer.
The reported charge is the quadrature approximation to `integral_D rho`.

| Starter mesh | Domain |
|---|---|
| `sphere.mesh` | Straight-sided polyhedral approximation of the radius-5 ball |
| `l_prism.mesh` | `[-5,5]^3` minus `[0,5] x [0,5] x [-5,5]` |
| `perforated_cube.mesh` | `[-5,5]^3` minus the internal cube `[-1,1]^3` |

Adaptive and reference meshes are derived from the **same starter mesh**.
Their boundary geometry therefore remains the same. In particular, globally
refining the sphere mesh subdivides its existing flat boundary faces; it does
not project new vertices onto the exact sphere. Errors in this experiment
refer to the fixed faceted domain, not to the exact curved ball. The initial
sphere volume discrepancy must be assessed separately for exact-sphere claims.

## Mesh construction and indicator

AFEPack reads each starter mesh and builds its hierarchy. A requested number
of global red-refinement steps is applied first. Each ordinary tetrahedron
has eight children per step. “Uniform reference mesh” means uniform/global
refinement of this unstructured tetrahedral mesh; it does **not** mean a
Cartesian grid or tetrahedra of identical size and shape.

Adaptive solves subsequently use `MeshAdaptor` with the radial monitor

```text
c       = (-2,-2,0)
delta   = 0.5
m(x)    = 0.02 + 1/sqrt(|x-c|^2 + delta^2)
eta_K   = integral_K m(x) dx
```

The indicator is recomputed using code-owned order-6 quadrature at each
adaptation pass. It requests finer resolution around the source center and
retains a nonzero background monitor. It is a prescribed radial mesh monitor,
not a residual-based or certified error estimator. The Gaussian density is
anisotropic, whereas this monitor is radial; no optimality claim is made.
The controls are adaptation tolerance, number of passes, and refinement steps
per pass. AFEPack's tetrahedron, twin-tetrahedron, and four-tetrahedron elements
are handled by the existing local quadrature decomposition.

## Targets and reported error

Generate the target file once per domain and reuse it for every mesh and
parameter setting. Defaults are:

- 128 volume-uniform targets, obtained by selecting a starter tetrahedron
  with probability proportional to volume and sampling uniformly inside it;
- 32 extra source-core targets in the radius-0.9 ball around `c`, rejecting
  points outside the mesh domain.

The random seed is fixed at `20261108`. The two groups remain separate.
The core targets must not be added with equal weights to a domain-volume
error estimate. They test the more demanding near-source region separately.

For the volume group, a sampled absolute domain error can be estimated by

```text
E_sample = sqrt(volume(D)/M * sum_i |V_tree(x_i)-V_ref(x_i)|^2)
E_relative_sample = sqrt(sum_i |V_tree(x_i)-V_ref(x_i)|^2
                         / sum_i |V_ref(x_i)|^2)
```

These are finite-sample estimates, **not certified continuum L2 norms**.
Report the volume-group error and core-group diagnostics separately. Check
target-sample stability before treating small differences between runs as
significant. All reference and treecode values must use identical target
coordinates within a comparison; no interpolation from reference barycenters
is required.

The timed production evaluation is different: it evaluates one barycenter
per active leaf over the entire solve mesh. The fixed validation targets are
evaluated separately and their additional cost is recorded separately.

## Uniform reference and same-mesh direct check

`reference` mode globally refines the starter mesh and directly integrates
over all source tetrahedra. It does not use HNL, MAC, multipole expansions, or
treecode interaction lists. Code-owned quadrature samples are processed in
bounded-size tiles. Target-containing tetrahedra use the existing singular
tetrahedron/Duffy integration. Nearby nonsingular tetrahedra receive local
quadrature subdivision controlled by `--near-depth`.

A reference is not reliable merely because it uses more elements. Compare
successive global refinement levels, increase its quadrature order, and check
the near-subdivision depth. Only designate a result as the reference after
these changes are small compared with the reported solve errors. Keep the
reference-discretization differences in the report; do not label a numerical
reference “exact.” Reference calculation and validation are excluded from
production solver timing.

The solve flag `--same-direct` additionally performs a direct integration on
the **same source mesh and at the same p** as the treecode. This produces a
separate `.direct.csv` file and checks that production and direct source
charges agree. It diagnoses the error introduced by replacing source
integrals with multipole interactions. Agreement in this check does not
establish that the shared quadrature, source resolution, or geometry is
accurate. The comparison against a converged uniform reference assesses the
combined numerical difference instead. A failure shared by the quadrature
and singular-integration helpers would also require an independent test.

## Build and individual runs

Requirements: AFEPack and its tetrahedral templates, deal.II, an MPI C++
compiler, and OpenMP. From the project root:

```bash
make -C general_domain
export AFEPACK_TEMPLATE_PATH=/usr/local/include/AFEPack/template/triangle/:/usr/local/include/AFEPack/template/twin_triangle/:/usr/local/include/AFEPack/template/tetrahedron/:/usr/local/include/AFEPack/template/twin_tetrahedron/:/usr/local/include/AFEPack/template/four_tetrahedron/
export OMP_NUM_THREADS=16
export OMP_DYNAMIC=FALSE

# Fixed validation points; --output is the target CSV filename in this mode.
general_domain/compare_general_domain --mode targets \
  --mesh domain_meshes/l_prism.mesh \
  --volume-targets 128 --core-targets 32 \
  --output general_domain/example/l_prism.targets.csv

# Adaptive production solve and separate same-mesh direct check.
general_domain/compare_general_domain --mode solve \
  --mesh domain_meshes/l_prism.mesh --level 0 \
  --p 6 --P 4 --theta 0.5 --tol 0.03 --passes 2 --refine-steps 1 \
  --targets general_domain/example/l_prism.targets.csv --same-direct \
  --output general_domain/example/l_prism_adaptive

# Direct uniform-mesh reference candidate, not automatically converged.
general_domain/compare_general_domain --mode reference \
  --mesh domain_meshes/l_prism.mesh --level 1 --p 10 --near-depth 3 \
  --targets general_domain/example/l_prism.targets.csv \
  --output general_domain/example/l_prism_reference_L1_p10
```

Repeat the reference with a finer `--level`, larger `--p`, or larger
`--near-depth` as appropriate, keeping the target file unchanged. A uniform
**treecode solve**, distinct from a direct reference, uses `--mode solve
--uniform`; `--level` then sets the global mesh refinement.

The completed pilot is in `runs/pilot_20261108/`. It contains four adaptive
runs per shape, the uniform reference ladder, and all raw fixed-target
potentials. The main tables select tolerances 0.01, 0.003, and 0.001; the
initial 0.03 run remains available in the raw files. The accepted pilot
reference uses global level 2, quadrature accuracy 16, and near-subdivision
depth 4, after comparisons with levels/orders 1/12, 2/8, and 2/12.

The drivers are resumable and do not silently overwrite conflicting results:

```bash
python3 general_domain/run_study.py --threads 16
python3 general_domain/run_refinements.py --threads 16
python3 general_domain/run_reference_checks.py --threads 16 --validation
python3 general_domain/summarize_study.py
python3 general_domain/plot_mesh_comparison.py
```

Use a new `--directory` with each script for a fresh study after editing the
solver. `results_general_domain.txt` is the compact result table;
`summary.csv` retains all runs and additional error statistics. `report.pdf`
contains the numerical results and `mesh_comparison.pdf` shows actual mesh
sections. The all-near consistency test agrees with direct integration to
5.33e-15; the final reference sensitivity is below 2.6% of the corresponding
smallest production relative errors. These are numerical validation checks,
not rigorous reference-error bounds.

| Control | Meaning |
|---|---|
| `--mode targets/solve/reference` | Generate targets, treecode solve, or direct reference |
| `--mesh FILE` | AFEPack starter mesh |
| `--level 0..3` | Global refinement count; target-generation mode uses level 0 |
| `--p 2..20` | Source quadrature algebraic accuracy |
| `--P 1..20` | Cartesian multipole order in solve mode |
| `--theta` | MAC threshold in `(0,1)` for solve mode |
| `--tol`, `--passes`, `--refine-steps` | Radial-monitor adaptation controls |
| `--uniform` | Disable adaptation for a treecode solve |
| `--near-depth 0..6` | Maximum local subdivision depth for reference integration |
| `--same-direct` | Additional same-mesh, same-p direct validation |
| `--targets FILE` | Previously generated fixed target CSV |
| `--output PREFIX` | Output prefix; target mode instead takes an exact CSV path |

The executable limits solve meshes to 300,000 active elements and reference
meshes to 1,000,000. A global step that would exceed the configured cap is
rejected. These safety caps and the very different initial root counts mean
that the same global level does not imply the same N across domains.

## Files, timing, and interpretation

Each solve/reference prefix produces `.values.csv` (fixed-target potentials),
`.json` (parameters and measurements), and `.dx` (the actual source mesh).
The optional same-mesh check adds `.direct.csv`. The JSON records actual
OpenMP team size, N, root count, quadrature count, source charge, and mesh
volume. No result is inferred from a requested element count.

Timing fields distinguish:

- `mesh_s`: mesh reading, global refinement, and adaptation;
- `list_s`: complete cache/list-construction call; `list_kernel_s` is its
  separately instrumented list kernel and is not an additional time to add;
- `quadrature_s`, `moments_s`: source quadrature and multipole preparation;
- `evaluation_s`: potential evaluation at all active-leaf barycenters;
- `total_solver_s`: mesh, list, quadrature, moments, and leaf evaluation;
- `validation_target_evaluation_s`, `same_mesh_direct_s`, and reference
  timing fields: separate validation/reference work, excluded from solver time.

Direct-reference timing at 160 targets must not be presented as a speedup
baseline for production evaluation at N leaf barycenters: the target workloads
are different.

The initial pilot uses 16 CPU threads, and the server is shared/loaded. Treat
its timings as functional pilot measurements, not publication-quality
performance data. Rerun on an otherwise idle machine with a fixed affinity
and repeated timings before drawing speedup conclusions. These Gmsh starter
meshes also introduce many independent hierarchy roots. Record `root_count`:
list preparation depends on the initial forest as well as the active leaf
count, so its timing is not directly comparable with an older cube experiment
built from a much smaller initial root mesh. This issue must be audited before
claiming the same N-scaling constant across geometries.

Numerical summaries and the final report are generated after the runs and
reference-stability checks. This README specifies the experiment; it does not
assert convergence, speedup, or any unmeasured result.

## More concentrated mesh illustrations

The separate `generate_localized_mesh` utility uses the same AFEPack starter
meshes and mesh adaptor with the monitor

\[
\eta_K=\int_K\left[10^{-4}+
  \left(1+\frac{|\mathbf{x}-(-2,-2,0)|^2}{1.2^2}\right)^{-3}\right]d\mathbf{x}.
\]

This is a prescribed smooth monitor, not an error estimator. The mesh-only
run uses tolerance `0.0002`, four adaptation passes, and order-6 quadrature
for the monitor. It does not construct interaction lists or evaluate a
potential. The original `adapt()` monitor, pilot results, and existing solver
binary are retained.

```bash
make -C general_domain generate_localized_mesh
python3 general_domain/run_localized_meshes.py
python3 general_domain/plot_localized_meshes.py
```

Outputs are in `runs/localized_meshes_20261108/`: actual `.dx` meshes with
metadata and commands, `localized_meshes.png`, `localized_meshes.pdf`, and
the three-page `mesh_before_after.pdf`. Use a new `--directory` with both
Python commands when changing generation parameters; existing mesh outputs
are never overwritten.

| Domain | Active elements | Previous far/core size ratio | New ratio |
|---|---:|---:|---:|
| Sphere | 12,178 | 1.79 | 3.75 |
| L-shaped prism | 8,626 | 1.75 | 4.61 |
| Perforated cube | 9,522 | 1.97 | 6.38 |

Here size is the cube root of tetrahedron volume; the ratio compares the
unweighted medians for centroids at radius greater than 3 and less than 1
from the source center. These statistics refer to the exported ordinary
tetrahedra, including decompositions of twin/four elements. The figures are
actual `z=0` sections with identical plotting coordinates and line widths.
Source-core median sizes remain similar to the previous meshes, while much
of the outer mesh is coarser. Counts are not matched. The mesh-only outputs
above contain no potential calculations; the subsequent accuracy test is
stored separately as described below.

## Potential accuracy on the localized meshes

The paired accuracy test is in `runs/localized_accuracy_20261108/`, with
`report.pdf`, `report.md`, `results_localized_accuracy.txt`, and `summary.csv`.
It uses `p=6`, `P=4`, `theta=0.5`, and 16 OpenMP threads, retaining the
pilot's 128 volume targets, 32 separate core targets, and uniform L2/p16
direct reference. The previous radial meshes are rerun with the same new
executable. Every generated DX file must match the previously displayed
mesh byte for byte; all six checks passed. Fresh radial potential values
also match the original pilot values exactly.

```bash
make -C general_domain compare_localized_domain
python3 general_domain/run_localized_accuracy.py
python3 general_domain/summarize_localized_accuracy.py
```

The new executable accepts `--monitor radial|localized`,
`--monitor-width 1.2`, and `--monitor-floor 0.0001`. The default monitor
remains radial. The runner records commands, source/input hashes, actual
thread counts, and separate preprocessing/evaluation timings, without
overwriting the original solver executable or pilot results. Use a new
`--directory` for runs after changing the solver or execution settings.

The measured localized/radial relative RMS error ratios are:

| Domain | Volume targets | Core targets |
|---|---:|---:|
| Sphere | 1.538 | 1.320 |
| L-shaped prism | 0.982 | 0.871 |
| Perforated cube | 1.407 | 0.608 |

A ratio above one means worse accuracy. Thus concentrating refinement is
not uniformly more accurate: the sphere worsens in both groups, and the
perforated-domain volume error increases even though its core error falls.
The localized meshes use 76–90% fewer elements. Fresh total times are
3.048/1.313/1.715 seconds versus 14.837/10.678/17.331 seconds for the radial
meshes, respectively. These are single-run shared-server observations,
with different numbers of production targets; they are not an equal-workload
acceleration benchmark. The reference-sensitivity checks pass, but the
reported errors remain sampled diagnostics rather than certified continuum
L2 bounds. The report also records maximum absolute errors and the separate
same-mesh direct comparisons.

## Consolidated report

After generating the pilot, mesh illustrations, and localized-accuracy
results, build the complete report with

```bash
python3 general_domain/build_general_domain_report.py
```

This verifies the recorded mesh, element-count, thread-count, and reference
guards before writing `runs/general_domain_report_20261108.pdf` and the
matching Markdown summary. The PDF contains an executive summary, the
original pilot, actual mesh sections, radial-versus-localized mesh views,
the potential-accuracy report, and the separate radial-source/radial-indicator
study. It preserves unfavorable results and does not present the single-run
timings as an equal-workload speedup measurement.

## Radial source and radial indicator

The separate radial study uses

\[
\rho(\mathbf x)=\exp\!\left[-4|\mathbf x-(-2,-2,0)|^2\right]
\]

and the radial size profile

\[
h_{\mathrm{rel}}(r)=1+(\gamma-1)\frac{r^2}{r^2+R^2},
\qquad
\eta_K=\int_K h_{\mathrm{rel}}(r)^{-3}\,d\mathbf x,
\]

with `R=1.5`, `gamma=4`, tolerance `0.0002`, and four adaptation
passes. The monitor prescribes a smooth radial grading and is not an error
estimator. Build and reproduce it with

```bash
make -C general_domain compare_radial_test_domain
python3 general_domain/run_radial_domain_study.py
python3 general_domain/summarize_radial_domain.py
python3 general_domain/plot_mesh_comparison.py \
  --run-dir general_domain/runs/radial_domain_20261108 \
  --adaptive adaptive_radial_graded --reference uniform_L2_p16_d4
```

The complete outputs are in `runs/radial_domain_20261108/`. The study uses
the same common target files as the pilot and new direct references for the
radial density. The final p12/depth3-to-p16/depth4 sensitivity guards pass
for all domain/target groups. Results remain sampled diagnostics rather than
certified continuous norms.
