# Treecode workspace

Open [CATALOG.html](CATALOG.html) in a browser to search files by name and browse code, scripts, documents, figures, inputs, and results. Regenerate it after adding files:

```sh
python3 tools/catalog.py
```

## Code variants

Each directory remains a separate working area. Run its commands from inside that directory: programs and scripts use relative input and output paths.

| Directory | Contents |
| --- | --- |
| `errorControl/` | Error-control experiments, CPU/CUDA implementations, H2O problems, mesh policies, reports, and presentations; included in the workspace Git repository |
| `2d/` | Two-dimensional treecode and MAC comparison sources |
| `MACrefine/` | MAC refinement implementation |
| `gradient_based_indicator/` | Gradient-based indicator implementations |
| `reference/` | Reference calculation sources and binary results |
| `rough_tree/` | Treecode variants and comparison sources |
| `strictMAC/` | Strict MAC and singular/comparison sources |

Root-level `checkbin.cpp` and `femspace.cpp` are standalone sources. Their associated files remain in place. Meshes, binary reference data, `.dx` visualizations, and `.mp` files are preserved; do not assume they are disposable.

## Working with errorControl

```sh
cd errorControl
make                 # default CPU executable
make h2o             # H2O executable
make mesh-modes      # mesh-policy executable
make gpu             # CUDA executable
```

The makefile expects MPI C++, AFEPack, deal.II, and OpenMP at configured system paths; GPU targets also require CUDA. These commands are a guide to existing targets, not a statement that dependencies are installed.

- Core implementation: `compare_exact.cpp`, `local_quadrature.hpp`, `cartesian_multipole.hpp`.
- CUDA implementation: `compare_gpu.cu`, `compare_gpu.hpp`.
- Problems and variants: `gaussian_test_problem.hpp`, `h2o_*test_problem.hpp`, `compare_h2o.cpp`, `compare_reuse.cpp`, `compare_mesh_modes.cpp`.
- Experiments: `run_*.sh`; detailed run output lives in `experiment_results/`.
- Reports: `balanced_parameter_report.md`, `uniform_adaptive_comparison.md`, `mesh_modes.md`.
- Papers and slides: `article.tex`, `hartree_*beamer*.tex` and corresponding PDFs/build scripts.

Use `git status` from the workspace root to inspect changes across all variants. The earlier `errorControl` Git metadata, including its history and staging, is preserved locally in `.local-history/errorControl.git` and excluded from uploads. The new workspace repository starts from the current files; the older history is not part of its initial commit. To inspect that history: `git --git-dir=.local-history/errorControl.git log --oneline`.

## Keeping the workspace organized

Keep each variant's source, build files, and required runtime inputs together until its relative paths are migrated. Place new experiment runs in named subdirectories of `errorControl/experiment_results/`; use supported output arguments where available. Keep figures beside documents that reference them.

The first cleanup archives only selected LaTeX build intermediates in `errorControl/archive/cleanup-2026-10-02/`. Its `manifest.json` records original paths, archived paths, and SHA-256 hashes. No research results or source files are deleted. LaTeX may regenerate intermediates on its next build; the local ignore rules keep them out of Git status. Bibliography `.bbl` files and generated math assets are retained in place because document builds may need them.
