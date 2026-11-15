# Recovered ten-H2O solver source

This directory contains the coherent historical source snapshot used for the
ten-water experiments. It was recovered on 2026-11-10 from the preserved
`treecode-review-extract-*` review-package snapshots after the working copies
had been deleted.

The entry point is `compare_water10.cpp`. It wraps the matching historical
`compare_exact.cpp`; the current top-level solver has evolved since these runs,
so this directory intentionally keeps its own supporting headers.

Build from the repository root with:

```bash
make water10
```

Run the default case with:

```bash
make run-water10
```

The molecular geometry and archived experiment outputs are in
`../water10_case/`. The model is a synthetic positive 30-center Gaussian
density with 100 electrons, not a self-consistent Kohn--Sham density.

The recovered entry-point hashes are:

```text
3857d79d8cdd3a36177e3706ccf879ae75e4740a46a7d94cfe8d5fe13a4f2836  compare_water10.cpp
2e73d7ba30de443063ba0e1c9cf3a349b60b8f36dbc3fe95b4f966069bba4377  compare_exact.cpp
c1e5ecbd23db43e6897d298b65df11d533171ee13c75487d8a2c8716fe8d45ba  water10_test_problem.hpp
```
