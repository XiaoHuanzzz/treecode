# Article source

This directory contains the source of **A Treecode Algorithm for Hartree
Potential Evaluation on Locally Refined Meshes**.

For the public, versioned code, data, regression tests, and audited arXiv
source, start with [`paper_artifact/`](paper_artifact/README.md).  That directory
is the reproducibility record associated with the `arxiv-v1` Git tag; the other
files in this working directory include historical development material.

## Files

- `article.tex`: manuscript source.
- `article.pdf`: compiled manuscript.
- `ref.bib`: bibliography database.
- `article_figure_pdfs/`: the 17 figures used by the manuscript.
- `mac_geometry_figure_body.tex`: TikZ source included directly by the manuscript.
- `algorithm2e.sty` and `sn-mathphys-num.bst`: bundled LaTeX dependencies.
- `article_source_latest.zip`: portable copy of the complete article source.

## Build

```bash
pdflatex article.tex
bibtex article
pdflatex article.tex
pdflatex article.tex
```

## Recovered experiment

The earlier H2O Becke fuzzy-cell study recovered from Git history is kept in
[`recovered_fuzzy_cell/`](recovered_fuzzy_cell/README.md). It is separate from
the files used to compile the current article.
