# HNL construction cost and list reuse

## Work growth

All cases use p=6, P=4, theta=0.5 and 16 OpenMP threads. Times are medians of three interleaved runs; counts are deterministic.

| N | depth | T_list median [min,max] (s) | us/target | candidates/target | materialization/target | MAC visits/target | list entries/target |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,328 | 4 | 0.124 [0.123,0.124] | 12.00 | 463.3 | 506.6 | 1856.1 | 1679.8 |
| 34,464 | 5 | 0.678 [0.677,0.690] | 19.68 | 775.6 | 860.5 | 2800.6 | 2545.1 |
| 100,656 | 5 | 1.957 [1.952,1.958] | 19.44 | 769.9 | 856.0 | 2768.2 | 2516.2 |
| 191,544 | 5 | 5.284 [5.258,5.289] | 27.59 | 1104.0 | 1234.9 | 3549.1 | 3240.9 |
| 822,720 | 6 | 24.152 [24.081,24.179] | 29.36 | 1174.5 | 1317.4 | 3723.2 | 3402.3 |

N increases by 79.7 times. The average candidate, materialization, MAC-visit, and final-list counts increase by 2.54, 2.60, 2.01, and 2.03 times, respectively.
These measurements quantify work on this mesh family but do not establish an asymptotic complexity bound.

## Repeated density updates

The reuse study uses N=100,656, p=6, P=4, theta=0.5 and 64 OpenMP threads. Each control rebuild uses the same HNL construction; no alternative traversal is involved.

| updates | rebuild total (s) | reuse total (s) | saved (%) | speedup |
|---:|---:|---:|---:|---:|
| 1 | 2.798 | 2.950 | -0.9 | 0.991 |
| 2 | 5.570 | 4.857 | 13.3 | 1.154 |
| 5 | 13.684 | 10.612 | 22.5 | 1.290 |
| 10 | 27.245 | 20.193 | 25.9 | 1.349 |
| 20 | 53.934 | 38.988 | 27.6 | 1.381 |

At 20 updates, retaining the initial lists saves 27.6% of cumulative time and gives a 1.381-fold speedup.
Every measured rebuild reproduces the cached near/far lists exactly and passes the source-partition checks.
The sequence is manufactured and SCF-like; it is not a Kohn--Sham SCF calculation.
