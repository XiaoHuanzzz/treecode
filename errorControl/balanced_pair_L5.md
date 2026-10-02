# Uniform versus adaptive mesh comparison

Both calculations use the same manufactured Gaussian problem, CPU backend, 128 OpenMP threads, one target at each active-leaf barycenter, and identical treecode parameters: \(p=8\), \(P=8\), and \(\theta=0.5\). Adaptive base level 0 and uniform level 5 give comparable element counts. Timings are arithmetic means of 1 runs.

| Mesh | Level | Elements | \(T_{\mathrm{list}}\) (s) | \(T_{\mathrm{moment}}\) (s) | \(T_{\mathrm{eval}}\) (s) | \(T_{\mathrm{total}}\) (s) | Resolved \(\lVert V_{\mathrm{ex}}\rVert_{L^2}\) | \(L^2\) error | Relative error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Adaptive | 0 | 823672 | 9.16832028702 | 4.1320351176 | 65.2259049499 | 97.2509549893 | 0.134326563889 | 1.23670357687e-05 | 9.20669405264e-05 |
| Uniform | 5 | 786432 | 6.65608682763 | 3.56101782806 | 60.7888064869 | 90.8709460609 | 0.13440452606 | 5.25091527332e-05 | 0.00039067994414 |

For reference, the effectively untruncated analytic norm is approximately \(0.134321\); the resolved-norm column therefore also reveals whether one barycenter per leaf resolves the localized Gaussian.

## Interpretation

- The uniform/adaptive total-time ratio is **0.934396**: the uniform run is **-6.56%** slower, so the two costs are effectively comparable at this problem size.
- The adaptive leaf-barycenter error is **4.2459 times smaller** than the uniform value.
- The adaptive barycenter rule resolves the analytic norm with a signed bias of **0.003962%**, whereas the uniform bias is **0.062%**. The uniform level-4 target rule therefore under-resolves the localized Gaussian peak.

The error ratio is a production-grid, leaf-barycenter comparison. It should not be presented as a common-grid continuum-error ratio until both solutions are checked on an independent validation rule.
