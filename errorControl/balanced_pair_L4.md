# Uniform versus adaptive mesh comparison

Both calculations use the same manufactured Gaussian problem, CPU backend, 128 OpenMP threads, one target at each active-leaf barycenter, and identical treecode parameters: \(p=6\), \(P=6\), and \(\theta=0.5\). Adaptive base level 0 and uniform level 4 give comparable element counts. Timings are arithmetic means of 1 runs.

| Mesh | Level | Elements | \(T_{\mathrm{list}}\) (s) | \(T_{\mathrm{moment}}\) (s) | \(T_{\mathrm{eval}}\) (s) | \(T_{\mathrm{total}}\) (s) | Resolved \(\lVert V_{\mathrm{ex}}\rVert_{L^2}\) | \(L^2\) error | Relative error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Adaptive | 0 | 100432 | 1.10640884377 | 0.375420853496 | 2.85267727263 | 6.14892576262 | 0.134404526057 | 0.000157979428544 | 0.00117540259379 |
| Uniform | 4 | 98304 | 0.997539805248 | 0.322652247734 | 2.66426591855 | 6.2034102371 | 0.111790155645 | 0.00601361262064 | 0.0537937583674 |

For reference, the effectively untruncated analytic norm is approximately \(0.134321\); the resolved-norm column therefore also reveals whether one barycenter per leaf resolves the localized Gaussian.

## Interpretation

- The uniform/adaptive total-time ratio is **1.00886**: the uniform run is **0.8861%** slower, so the two costs are effectively comparable at this problem size.
- The adaptive leaf-barycenter error is **38.0658 times smaller** than the uniform value.
- The adaptive barycenter rule resolves the analytic norm with a signed bias of **0.062%**, whereas the uniform bias is **-16.77%**. The uniform level-4 target rule therefore under-resolves the localized Gaussian peak.

The error ratio is a production-grid, leaf-barycenter comparison. It should not be presented as a common-grid continuum-error ratio until both solutions are checked on an independent validation rule.
