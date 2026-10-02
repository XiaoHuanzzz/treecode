# Balanced treecode parameters across mesh scales

## Objective

The purpose of this experiment is to choose a balanced combination of the
quadrature algebraic accuracy \(p\), Cartesian multipole order \(P\), and MAC
threshold \(\theta\) at each mesh scale.  Accuracy is assessed on a common set
of 786,432 level-5 target points, using the analytic potential as the reference.
The target is

$$
\|V_{\mathrm{tree}}-V_{\mathrm{ex}}\|_{L^2(\Omega)}\le 10^{-4}.
$$

All production timings use the CPU backend with 128 OpenMP threads.  The
reported values are single wall-clock measurements.

## Parameter-selection procedure

1. Screen \(p=2,4,6,8\), extending to \(p=12,16\) only when needed.
2. Screen coupled \((P,\theta)\) candidates rather than evaluating the full
   Cartesian product.
3. Use native leaf-barycenter errors only for preliminary screening.
4. Validate promising configurations for both adaptive and uniform meshes at
   the same 786,432 target coordinates.
5. Compute the final error against the analytic solution, not against the
   stored uniform potential values.
6. Increase parameters selectively if a native-grid winner fails the common
   validation target.

The screened \((P,\theta)\) pairs were

$$
(2,0.9),(2,0.7),(2,0.5),(4,0.7),(4,0.5),(4,0.3),
(6,0.7),(6,0.5),(6,0.3),(8,0.7),(8,0.5).
$$

The very strict value \(\theta=0.1\) was excluded from the staged search
because earlier tests showed a large increase in direct interactions with
negligible accuracy improvement over \(\theta=0.3\).

## Final balanced configurations

| Scale | Mesh | \(n\) | \(p\) | \(P\) | \(\theta\) | \(T_{\mathrm{list}}\) (s) | \(T_{\mathrm{moment}}\) (s) | \(T_{\mathrm{eval}}\) (s) | \(T_{\mathrm{total}}\) (s) | Common-grid \(L^2\) error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L2 | Adaptive | 1,032 | 16 | 4 | 0.7 | 0.0595 | 0.0141 | 0.0682 | 0.2972 | \(4.4614\times10^{-3}\) |
| L2 | Uniform | 1,536 | 16 | 4 | 0.7 | 0.0225 | 0.0159 | 0.0640 | 0.2968 | \(4.4599\times10^{-3}\) |
| L3 | Adaptive | 10,496 | 16 | 8 | 0.7 | 0.0769 | 0.1388 | 0.6308 | 1.3423 | \(4.4554\times10^{-4}\) |
| L3 | Uniform | 12,288 | 16 | 8 | 0.7 | 0.0789 | 0.1509 | 0.6418 | 1.4176 | \(4.4559\times10^{-4}\) |
| L4 | Adaptive | 100,432 | 8 | 8 | 0.7 | 1.0107 | 0.3958 | 2.7325 | 6.6935 | \(8.3734\times10^{-5}\) |
| L4 | Uniform | 98,304 | 16 | 8 | 0.7 | 0.8763 | 2.0391 | 10.0355 | 16.9804 | \(7.1433\times10^{-5}\) |
| L5 | Adaptive | 823,672 | 8 | 6 | 0.5 | 8.5547 | 3.0858 | 37.1372 | 67.9263 | \(9.0704\times10^{-5}\) |
| L5 | Uniform | 786,432 | 8 | 6 | 0.5 | 7.8554 | 2.4889 | 34.0349 | 65.8349 | \(5.9447\times10^{-5}\) |

## Main observations

- Neither L2 nor L3 reaches the \(10^{-4}\) target under the tested limit
  \(p\le16\).  They should therefore be described as the best tested coarse
  configurations, not as target-satisfying choices.
- Both L4 configurations meet the target.  The adaptive mesh needs only
  \(p=8\), whereas the uniform mesh needs \(p=16\).  The adaptive L4 run is
  about 2.54 times faster in total and 3.67 times faster in potential
  evaluation.
- Both L5 configurations meet the target with \((p,P,\theta)=(8,6,0.5)\).
  At this scale, the uniform run is about 3.1% faster and has the smaller
  common-grid error.  Adaptivity is therefore most advantageous near the
  intermediate 100,000-element scale in this manufactured problem.
- The balanced quadrature order decreases as the mesh is refined.  Coarse
  tetrahedra require high-order quadrature to resolve the narrow analytic
  Gaussian; fine tetrahedra resolve it geometrically and need a lower \(p\).
- Native leaf-barycenter errors can be misleading.  For example, the L3
  adaptive native error is \(8.29\times10^{-5}\), but its common-grid error is
  \(4.46\times10^{-4}\).  Common-target validation is essential for a fair
  adaptive-versus-uniform comparison.
- The initial L5 choice \((6,6,0.5)\) also looked adequate on its native target
  set, but failed common validation.  Increasing to \(p=8\) reduced the common
  error below the target.

## Reproducibility files

- `run_balanced_tuning.sh`: resumable staged-search driver.
- `balanced_parameter_screen.tsv`: all preliminary screening runs.
- `balanced_parameter_selection.tsv`: provisional native-grid choices.
- `balanced_common_validation.tsv`: common-grid validation and corrections.
- `balanced_final_production.tsv`: final production timing measurements.
- `balanced_final_results.tsv`: final joined timing and error table.
- `experiment_results/balanced_parameter_screen/`: screening logs.
- `experiment_results/balanced_common_validation/`: validation logs.
- `experiment_results/balanced_final_production/`: final timing logs.

For publication-quality timing statistics, repeat each final configuration at
least three times and report the median together with the observed variation.
