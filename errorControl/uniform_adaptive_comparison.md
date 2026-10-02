# Uniform versus adaptive mesh comparison

Both calculations use the same manufactured Gaussian problem, CPU backend, 128 OpenMP threads, one target at each active-leaf barycenter, and identical treecode parameters: \(p=6\), \(P=4\), and \(\theta=0.5\). Adaptive base level 2 and uniform level 4 give comparable element counts. Timings are arithmetic means of 3 runs.

| Mesh | Level | Elements | \(T_{\mathrm{list}}\) (s) | \(T_{\mathrm{moment}}\) (s) | \(T_{\mathrm{eval}}\) (s) | \(T_{\mathrm{total}}\) (s) | Resolved \(\lVert V_{\mathrm{ex}}\rVert_{L^2}\) | \(L^2\) error | Relative error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Adaptive | 2 | 100656 | 1.00807463968 | 0.142230929496 | 1.47837244595 | 4.53385357869 | 0.134404526057 | 0.0002243440338 | 0.00166917023096 |
| Uniform | 4 | 98304 | 0.91368392203 | 0.15895266893 | 1.4197713295 | 4.59847770662 | 0.111790155645 | 0.00601865077929 | 0.0538388263667 |

For reference, the effectively untruncated analytic norm is approximately \(0.134321\); the resolved-norm column therefore also reveals whether one barycenter per leaf resolves the localized Gaussian.

## Interpretation

- The uniform/adaptive total-time ratio is **1.01425**: the uniform run is **1.425%** slower, so the two costs are effectively comparable at this problem size.
- The adaptive leaf-barycenter error is **26.8278 times smaller** than the uniform value.
- The adaptive barycenter rule resolves the analytic norm with a signed bias of **0.06201%**, whereas the uniform bias is **-16.77%**. The uniform level-4 target rule therefore under-resolves the localized Gaussian peak.

The error ratio is a production-grid, leaf-barycenter comparison. It should not be presented as a common-grid continuum-error ratio until both solutions are checked on an independent validation rule.
