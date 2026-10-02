# Uniform versus adaptive mesh comparison

Both calculations use the same manufactured Gaussian problem, CPU backend, 128 OpenMP threads, one target at each active-leaf barycenter, and identical treecode parameters: \(p=2\), \(P=2\), and \(\theta=0.5\). Adaptive base level 0 and uniform level 2 give comparable element counts. Timings are arithmetic means of 1 runs.

| Mesh | Level | Elements | \(T_{\mathrm{list}}\) (s) | \(T_{\mathrm{moment}}\) (s) | \(T_{\mathrm{eval}}\) (s) | \(T_{\mathrm{total}}\) (s) | Resolved \(\lVert V_{\mathrm{ex}}\rVert_{L^2}\) | \(L^2\) error | Relative error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Adaptive | 0 | 1032 | 0.0565687706694 | 0.0101287616417 | 0.0093529522419 | 0.188828760758 | 2.21452069431e-06 | 2.73964609718 | 1237128.24369 |
| Uniform | 2 | 1536 | 0.0356544526294 | 0.0489861285314 | 0.0110442023724 | 0.192421467975 | 2.21452069431e-06 | 2.74171360565 | 1238061.85812 |

For reference, the effectively untruncated analytic norm is approximately \(0.134321\); the resolved-norm column therefore also reveals whether one barycenter per leaf resolves the localized Gaussian.

## Interpretation

- The uniform/adaptive total-time ratio is **1.01903**: the uniform run is **1.903%** slower, so the two costs are effectively comparable at this problem size.
- The adaptive leaf-barycenter error is **1.00075 times smaller** than the uniform value.
- The adaptive barycenter rule resolves the analytic norm with a signed bias of **-100%**, whereas the uniform bias is **-100%**. The uniform level-4 target rule therefore under-resolves the localized Gaussian peak.

The error ratio is a production-grid, leaf-barycenter comparison. It should not be presented as a common-grid continuum-error ratio until both solutions are checked on an independent validation rule.
