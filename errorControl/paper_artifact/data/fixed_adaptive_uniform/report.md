# Fixed-parameter adaptive versus uniform study

The comparison uses the manufactured Gaussian problem on
`[-20,20]^3`, the CPU HNL--MAC evaluator, 128 OpenMP threads, and fixed
parameters

\[
(p,P,\theta)=(6,4,0.5).
\]

Production times are medians of three interleaved runs.  They include the
one-target-per-leaf potential evaluation but exclude the separate common-target
validation.  Both mesh types use the same 343,000 weighted target points for
the reported error.  The points are a tensor product of 10-point Gauss rules
on the intervals separated by

\[
-20,-2,-1,-0.5,0.5,1,2,20.
\]

Increasing the common-target rule from order 8 to order 10 changes the
reported errors by at most 1.1%.

| Scale | Mesh | N | Absolute common-target error | T_list (s) | T_eval (s) | T_total (s) |
|---|---:|---:|---:|---:|---:|---:|
| Coarse | Adaptive | 12,064 | 5.6474447 | 0.155 | 0.143 | 0.793 |
| Coarse | Uniform | 12,288 | 5.6474547 | 0.189 | 0.120 | 0.803 |
| Medium | Adaptive | 95,616 | 1.1628460e-1 | 1.225 | 1.436 | 4.837 |
| Medium | Uniform | 98,304 | 1.8891448 | 1.246 | 1.461 | 5.283 |
| Fine | Adaptive | 771,088 | 1.2194963e-2 | 9.373 | 15.115 | 41.383 |
| Fine | Uniform | 786,432 | 1.1625231e-1 | 9.705 | 15.381 | 47.044 |

The adaptive meshes use 1.8%, 2.7%, and 2.0% fewer leaves.  The coarsest pair
has essentially equal error, while the two finer adaptive errors are 16.2 and
9.5 times smaller.  Total time is reduced by 1.2%, 8.5%, and 12.0%.

The complete timing repetitions are in `runs.csv`, and all common-target
checks are in `common_validation.csv`.  `manifest.json` records the domain,
parameters, mesh hash, and execution protocol.
