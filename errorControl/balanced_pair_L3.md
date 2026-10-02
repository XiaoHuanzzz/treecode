# Uniform versus adaptive mesh comparison

Both calculations use the same manufactured Gaussian problem, CPU backend, 128 OpenMP threads, one target at each active-leaf barycenter, and identical treecode parameters: \(p=4\), \(P=4\), and \(\theta=0.5\). Adaptive base level 0 and uniform level 3 give comparable element counts. Timings are arithmetic means of 1 runs.

| Mesh | Level | Elements | \(T_{\mathrm{list}}\) (s) | \(T_{\mathrm{moment}}\) (s) | \(T_{\mathrm{eval}}\) (s) | \(T_{\mathrm{total}}\) (s) | Resolved \(\lVert V_{\mathrm{ex}}\rVert_{L^2}\) | \(L^2\) error | Relative error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Adaptive | 0 | 10496 | 0.103330202401 | 0.0122186718509 | 0.0942197181284 | 0.421264374629 | 0.0196068352803 | 0.231238546505 | 11.7937720799 |
| Uniform | 3 | 12288 | 0.121993175708 | 0.0495482254773 | 0.105379216373 | 0.492282622494 | 0.0196068352803 | 0.23124191664 | 11.7939439656 |

For reference, the effectively untruncated analytic norm is approximately \(0.134321\); the resolved-norm column therefore also reveals whether one barycenter per leaf resolves the localized Gaussian.

## Interpretation

- The uniform/adaptive total-time ratio is **1.16858**: the uniform run is **16.86%** slower, so the two costs are effectively comparable at this problem size.
- The adaptive leaf-barycenter error is **1.00001 times smaller** than the uniform value.
- The adaptive barycenter rule resolves the analytic norm with a signed bias of **-85.4%**, whereas the uniform bias is **-85.4%**. The uniform level-4 target rule therefore under-resolves the localized Gaussian peak.

The error ratio is a production-grid, leaf-barycenter comparison. It should not be presented as a common-grid continuum-error ratio until both solutions are checked on an independent validation rule.
