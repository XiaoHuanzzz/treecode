# Measured HNL--MAC interaction-list reuse

The experiment uses N=100,656, p=6, P=4, theta=0.5 and 64 CPU
threads. A manufactured density sequence changes at every step while
the mesh, targets and theta remain fixed. Each of three independent runs
contains 20 updates. At every update the control path actually rebuilds
the HNL--MAC lists on the same mesh; the reuse path retains the initial
lists. Both paths share identical density, moment and potential work.

| updates K | rebuild total (s) | reuse total (s) | saved (%) | speedup | reuse time/update (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.895 | 2.987 | -3.1 | 0.970 | 2.987 |
| 2 | 5.571 | 4.878 | 12.4 | 1.142 | 2.439 |
| 5 | 13.605 | 10.571 | 22.3 | 1.287 | 2.114 |
| 10 | 27.050 | 20.195 | 26.1 | 1.352 | 2.019 |
| 20 | 54.122 | 39.378 | 27.6 | 1.381 | 1.969 |

Median component times:

- initial_list_s: 1.0447 s
- rebuild_list_s: 0.7815 s
- density_update_s: 0.0140 s
- moment_update_s: 0.1240 s
- potential_evaluation_s: 1.7642 s

Every measured rebuild reproduced the cached near/far lists exactly;
the source-partition and MAC checks also passed. The experiment isolates
interaction-list reuse on a frozen mesh. It does not include the cost or
possible accuracy benefit of adapting the mesh between density updates.
