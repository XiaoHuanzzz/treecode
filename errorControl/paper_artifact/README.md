# Adaptive-mesh treecode for Hartree potential evaluation

This directory is the curated reproducibility artifact for the manuscript
**“A Treecode Algorithm for Hartree Potential Evaluation on Locally Refined
Meshes.”**  It contains the implementation, regression tests, experiment
drivers, processed and small raw records, meshes, provenance metadata, and the
current arXiv source.

The repository intentionally excludes large generated OpenDX meshes and common
target files from ordinary Git history.  The complete raw archive is attached
to the GitHub release
[`arxiv-v1`](https://github.com/XiaoHuanzzz/treecode/releases/tag/arxiv-v1) as
`treecode_hartree_complete_20261110.zip`.  Its SHA-256 digest is

```text
877a04105847b019fa852f5653979217467f2cb3991ab9259b62b26f5fc4637e
```

The exact arXiv upload source is attached to the same release as
`arxiv_source_20261115.zip` (SHA-256
`4c83e4972db42ef0ff45828846714a981cab18c6f7c77e1bf80bc0f1f018f9c9`).

## Contents

- `code/`: CPU solver, molecular and general-domain variants, experiment
  scripts, and self-contained regression tests.
- `data/`: records used for strong scaling, 64-thread production timings,
  error validation, parameter studies, HNL work statistics, list reuse,
  general-domain tests, and molecular examples.  Large generated inputs are in
  the release archive.
- `meshes/`: small AFEPack and Gmsh inputs for the irregular-domain tests.
- `manuscript/`: the audited arXiv source.  Its top-level file is
  `article.tex`.
- `SOURCE_DATA_MAP.md` and `SOURCE_DATA_MAP.tsv`: links from manuscript studies
  to source snapshots and archived evidence.
- `SHA256SUMS.txt`: checksums of all files in this curated directory, excluding
  the checksum file itself.

## Requirements

The main solver requires a C++17 compiler, OpenMP, MPI, AFEPack, and deal.II.
The paths in `code/Makefile` reflect the Moss workstation and may need to be
changed for another installation.  The CPU targets can be built with

```bash
cd code
make exact root-exact reuse
```

The principal executable accepts

```text
D.mesh initial_refinements p P theta compare backend [stage1_tol stage2_tol ...]
```

For example,

```bash
OMP_NUM_THREADS=16 ./tree_exact D.mesh 2 6 4 0.5 1 CPU 0.01 0.001
```

The self-contained regression tests can be run without AFEPack:

```bash
cd code/regression_tests
make check
```

## Manuscript build

```bash
cd manuscript
pdflatex article.tex
pdflatex article.tex
```

The pre-generated `article.bbl` is included.  To regenerate it from
`ref.bib`, run `bibtex article` between the two PDFLaTeX passes.

## Provenance

`SOURCE_DATA_MAP.tsv` distinguishes exact run records from older studies for
which the source or executable hash was not embedded in the run manifest.  The
strong-scaling and 64-thread production archives include source/executable
hashes, commands, actual OpenMP team sizes, and repeated-run records.  Historical
limitations are retained rather than replaced by inferred provenance.

## Code and data availability

The repository is public for inspection and reproducibility.  No open-source
license has yet been selected by all authors; normal copyright restrictions
therefore remain in force.  Contact the corresponding author before reuse or
redistribution beyond verification of the reported results.

