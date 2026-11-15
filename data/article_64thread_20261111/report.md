# Article timing rerun on one 64-core socket

All timings in this report are medians of three independent runs. OpenMP used
64 threads bound to CPU cores 0--63 of one AMD EPYC 7713 socket. Dynamic teams
were disabled. Every raw log reports an actual team size of 64. The complete
commands, load averages, source and executable hashes, individual timings, and
generated-mesh hashes are stored in `runs.csv`, `manifest.json`, and `raw/`.

## Adaptive and uniform meshes

The common-target errors are unchanged because thread count does not change the
mathematical problem. Only the production timings were rerun.

| Scale | Adaptive N | Uniform N | Adaptive total (s) | Uniform total (s) |
|---|---:|---:|---:|---:|
| Coarse | 12,064 | 12,288 | 0.577 | 0.631 |
| Medium | 95,616 | 98,304 | 4.541 | 5.381 |
| Fine | 771,088 | 786,432 | 42.223 | 46.107 |

On the finest pair, adaptive refinement reduces total time by 8.4 percent and
the existing common-target error by a factor of 9.5.

## Evaluation-parameter sweeps

| Sweep | Value | N | T_list (s) | T_eval (s) | T_total (s) | Absolute error |
|---|---:|---:|---:|---:|---:|---:|
| p | 2 | 100,656 | 0.853 | 1.224 | 4.002 | 2.3569e-2 |
| p | 4 | 100,656 | 0.914 | 1.388 | 4.186 | 1.2447e-3 |
| p | 6 | 100,656 | 0.915 | 1.700 | 4.838 | 2.1886e-4 |
| p | 8 | 100,656 | 0.929 | 2.250 | 5.872 | 1.6025e-4 |
| p | 12 | 100,656 | 0.854 | 4.231 | 8.284 | 1.5177e-4 |
| P | 2 | 100,656 | 0.947 | 0.969 | 4.142 | 2.0987e-3 |
| P | 4 | 100,656 | 0.915 | 1.700 | 4.838 | 2.1886e-4 |
| P | 6 | 100,656 | 0.889 | 3.277 | 6.355 | 1.5787e-4 |
| P | 8 | 100,656 | 0.822 | 6.170 | 9.234 | 1.5671e-4 |
| theta | 0.2 | 100,656 | 2.000 | 12.477 | 16.356 | 1.5668e-4 |
| theta | 0.3 | 100,656 | 1.209 | 4.951 | 8.453 | 1.5709e-4 |
| theta | 0.5 | 100,656 | 0.915 | 1.700 | 4.838 | 2.1886e-4 |
| theta | 0.7 | 100,656 | 0.747 | 0.913 | 3.947 | 5.7681e-4 |

## Mesh refinement

| N | T_list (s) | T_eval (s) | T_total (s) | Absolute error |
|---:|---:|---:|---:|---:|
| 10,328 | 0.094 | 0.141 | 0.553 | 5.9736e-3 |
| 34,464 | 0.285 | 0.636 | 1.726 | 1.1487e-3 |
| 100,656 | 0.915 | 1.700 | 4.838 | 2.1886e-4 |
| 191,544 | 2.125 | 4.113 | 10.564 | 2.2705e-4 |
| 425,498 | 5.059 | 9.882 | 23.602 | 1.7938e-4 |

The current source snapshot generates slightly different coarse-mesh leaf
counts from the older article dataset. The report uses the counts actually
printed by the rerun. An additional 822,720-leaf case is retained in the raw
archive but omitted from the five-point article curve to preserve a more even
range of mesh sizes.
