# HNL--MAC work counts on the article adaptive meshes

The five runs use `p=6`, `P=4`, `theta=0.5`, a fixed 24-root coarse
mesh, and the current adaptive indicator. Counts are deterministic and
are accumulated over every target leaf; timings are not used here.

| N | L | HNL candidates avg/max | Raw candidates avg | MAC visits avg/max | Near avg | Far avg | List avg/max | C/L | V/L | I/L |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,328 | 4 | 463.3/720 | 608.5 | 1856.1/3,145 | 286.2 | 1393.7 | 1679.8/2,810 | 115.8 | 464.0 | 420.0 |
| 34,464 | 5 | 775.6/1,672 | 1074.3 | 2800.6/4,640 | 317.4 | 2227.7 | 2545.1/4,147 | 155.1 | 560.1 | 509.0 |
| 100,656 | 5 | 769.9/1,714 | 1067.2 | 2768.2/5,080 | 305.4 | 2210.8 | 2516.2/4,553 | 154.0 | 553.6 | 503.2 |
| 191,544 | 5 | 1104.0/1,819 | 1557.2 | 3549.1/5,302 | 322.1 | 2918.8 | 3240.9/4,777 | 220.8 | 709.8 | 648.2 |
| 425,498 | 6 | 1228.6/2,232 | 1762.0 | 3906.0/7,048 | 331.2 | 3237.7 | 3568.9/6,275 | 204.8 | 651.0 | 594.8 |

Across a 41.2-fold increase in N, the observed per-level ranges are:
- final HNL candidates: 115.8--220.8;
- MAC node visits: 464.0--709.8;
- final list entries: 420.0--648.2.

These bounded measured ranges support the per-level-work assumption
for this adaptive family, but they do not prove a mesh-independent bound.
All partitions pass the coverage check and have maximum accepted MAC
ratio 0.5.
