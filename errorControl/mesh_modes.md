# SCF mesh-update experiment

`tree_mesh_modes` compares six policies for reusing or updating the adaptive
tetrahedral mesh during a manufactured density sequence.  The same treecode
quadrature, hierarchical neighborhood/MAC lists, Cartesian multipoles, and
leaf-barycenter potential evaluation are used by every policy.

## Policies

| Mode | Mesh action after the initial step |
|---|---|
| `FROZEN` | Keep the initial mesh and interaction lists. Update only the density samples and multipole moments. |
| `EVERY_H` | Perform source-residual-driven local refinement/coarsening at every step, then rebuild all geometry-dependent treecode data. |
| `PERIODIC_H` | Perform the same `h`-adaptation every `--period` steps. Reuse the mesh between updates. |
| `RESIDUAL_H` | Perform `h`-adaptation when the normalized source-projection residual exceeds `--residual-trigger`, with `--max-reuse` as a safety limit. |
| `MOVING_R` | Keep mesh connectivity fixed and move admissible interior vertices at every step. Rebuild geometry-dependent treecode data after a nonzero movement. |
| `HYBRID_HR` | Use the `RESIDUAL_H` decision for topology changes and also attempt a safeguarded vertex movement at every step. |

For `HYBRID_HR`, every accepted post-`h` mesh is stored as a geometry
checkpoint. Before a later topology update, the code restores that checkpoint
and applies AFEPack refinement/coarsening there; the new checkpoint is then
captured before the next `r`-movement. This prevents stale parent geometry
from being reactivated after vertices have moved and preserves the
truncated-domain volume.

The residual used here is

\[
\eta_\rho^2 =
\frac{\sum_K h_K^2\|\rho-\overline{\rho}_K\|_{L^2(K)}^2}
     {\|\rho\|_{L^2(\Omega)}^2}.
\]

It is a source-resolution residual for the integral Hartree solver.  It is
not the jump residual of a finite-element Poisson solve.

The moving-mesh monitor is based on the current source magnitude,

\[
M_i = \left(m_0 + \alpha
\frac{|\rho(\mathbf{x}_i)|}{\max_j|\rho(\mathbf{x}_j)|}\right)^q.
\]

Boundary vertices and vertices constrained by a hanging interface are held
fixed.  Each candidate movement is limited by a fraction of the local edge
length and accepted only if tetrahedron orientation and minimum quality are
preserved.  Internal tree clusters are then rebuilt from child volumes and
conservative child bounding spheres, so the MAC remains valid after
`r`-adaptation.

## Main parameters

- `--p`: algebraic accuracy of local tetrahedral quadrature.
- `--P`: Cartesian multipole expansion order.
- `--theta`: multipole acceptance threshold.
- `--steps`: number of density/SCF-like steps.
- `--period`: update interval for `PERIODIC_H`.
- `--residual-trigger`: threshold for `RESIDUAL_H` and `HYBRID_HR`.
- `--max-reuse`: maximum number of consecutive reuse steps for a
  residual-controlled policy.
- `--adapt-tol`, `--refine-steps`, `--refine-only`: AFEPack `h`-adaptation
  controls.
- `--monitor-strength`, `--monitor-exponent`, `--monitor-floor`: moving-mesh
  monitor controls \(\alpha\), \(q\), and \(m_0\).
- `--monitor-smooth`: graph-smoothing passes applied to the monitor.
- `--move-relax`, `--move-iters`: movement relaxation and iteration count.
- `--max-displacement`: maximum movement as a fraction of the shortest local
  edge.
- `--min-quality`: minimum accepted tetrahedron quality.
- `--output`: TSV result path.

## Build and run

```bash
make mesh-modes

make run-mesh-mode \
  MESH_MODE=RESIDUAL_H \
  MESH_MODE_OUTPUT=mesh_mode_RESIDUAL_H.tsv

make run-all-mesh-modes
```

A direct command can override every control:

```bash
OMP_NUM_THREADS=128 ./tree_mesh_modes D.mesh \
  --mode HYBRID_HR \
  --initial-refinements 2 \
  --p 6 --P 4 --theta 0.5 \
  --steps 6 --period 2 --max-reuse 4 \
  --residual-trigger 2.0e-2 \
  --adapt-tol 1.0e-6 --refine-steps 1 \
  --move-relax 0.2 --move-iters 5 \
  --output mesh_mode_HYBRID_HR.tsv
```

The TSV records the update decision, element count, list/cache/moment/potential
timings, step time, exact-solution error, movement size, mesh quality, and the
numbers of fixed and movable vertices.  The footer separately reports common
initial setup time, sequence time, full wall time, and update counts.  This
keeps the one-time common setup distinct from the per-step cost that the six
policies are intended to compare.
