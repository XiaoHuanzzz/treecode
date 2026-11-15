# Experimental data

The data directories retain raw logs and generated outputs whenever those files
were available.  Files ending in `.job.json` record command lines and environment
settings; `.log` files contain program output; `.csv` and `.tsv` files contain raw
or summarized measurements; `.dx` files are generated meshes or fields.

The article uses several distinct mesh sequences. Identical-looking element counts
must not be substituted across studies. `strong_scaling_article/` is the measured
`N=100432` article study and contains 21 result records, their manifests, the exact
source snapshot, and an automated verifier. `strong_scaling_superseded_summary/`
retains the unsupported `N=100656` summary only for audit history. The archived
target-sampling sequence is distinct from the HNL/reuse production mesh.
