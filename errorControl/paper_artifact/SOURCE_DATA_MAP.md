# Source and data provenance

`SOURCE_DATA_MAP.tsv` maps each manuscript study to its archived output and
source version. “Exact” means that the run manifest itself records the source
and executable hashes. “Git capture” means that source and output coexist in a
preserved Git tree, but the older driver did not write a binary hash into its
run manifest; this distinction is kept explicit rather than inferred away.

The CPU strong-scaling evidence is fully exact. Git tree
`0385d7f52d240f150b9febf65fe921979952c554` contains 21 solver-produced result
records, their manifests, and the complete frozen source. Run
`article_strong_scaling_100432_git/verify_archive.py` to check all hashes and
reproduce the seven medians and speedups in the article.

The old `article_strong_scaling_100656_20261110` directory is a superseded
summary without raw repeats. It is excluded from manuscript evidence and is
retained only as clearly marked audit history in the complete package.

The 64-thread production rerun is under `article_64thread_20261111`. It replaces
the 128-thread timings in the adaptive--uniform comparison, fixed-mesh
parameter sweeps, and mesh-refinement curve. All runs are bound to cores 0--63
of one socket. The archive records exact executable and source hashes, the
actual OpenMP team size, load averages, commands, raw logs, individual results,
and three-run medians. Mesh-independent validation errors are unchanged and
their costs remain excluded from production timings.
