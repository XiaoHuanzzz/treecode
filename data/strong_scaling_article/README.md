# Recovered strong-scaling archive

This archive was recovered from Git tree `0385d7f52d240f150b9febf65fe921979952c554` on 2026-11-10.
It is the measured source of the article's 1--64-thread strong-scaling curve.

- Problem: `N=100432`, `p=6`, `P=4`, `theta=0.5`.
- Backend: CPU, cached per-target HNL--MAC interaction lists.
- Targets: one leaf barycenter per active leaf.
- Statistic: median of three independent processes.
- OpenMP: `OMP_DYNAMIC=FALSE`, `OMP_PLACES=cores`,
  `OMP_PROC_BIND=close`.
- The 2- and 4-thread runs are a later measured batch made with the same
  executable. An 8-thread bridge differed from the original median by
  0.0343% and was not substituted into the summary.

Each `raw/t*_r*/results.txt` is the solver-produced one-row result for a
single run. The adjacent `manifest.json` records its command, environment,
and SHA-256 hashes. Git did not retain the ignored `console.log` files, but
the result records and manifests needed to reproduce every median are present.

`runs_1_64.csv` preserves the run order and all extracted timings.
`summary_full_1_64.csv` is the original median/minimum/maximum calculation.
`article_summary.csv` contains only the values plotted in the manuscript.
The frozen source in `source_snapshot/` matches the hashes in every run
manifest. The executable hash recorded by those manifests is
`1558ed5dae90015d15e1fcb656d698111516b49235836e1a32c648fa75e49c78`.

The previously used `N=100656`, 162.096--2.798 s summary had no matching raw
records in Git and must not be used as measured data.
