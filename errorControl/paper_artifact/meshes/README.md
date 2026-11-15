# AFEPack starter meshes for domain-geometry tests

These are boundary-conforming, linear tetrahedral starting meshes for testing
the treecode on different domain geometries. The `.mesh` files are AFEPack
internal mesh inputs; the `.geo` and `.msh` files preserve the Gmsh geometry and
mesh-generation source.

| File | Geometry | Active tetrahedra | Mesh volume |
|---|---|---:|---:|
| `sphere.mesh` | Radius-5 sphere centered at the origin | 5,899 | 519.325 |
| `l_prism.mesh` | `[-5,5]^3` with `[0,5] x [0,5] x [-5,5]` removed | 975 | 750 |
| `perforated_cube.mesh` | `[-5,5]^3` with the inner cube `[-1,1]^3` removed | 1,661 | 992 |

The sphere is represented by straight-sided tetrahedra, so its faceted mesh
volume is about 0.82% below the exact sphere volume `4*pi*5^3/3`. The two
polyhedral domains are represented exactly. These are starter meshes, not
matched-size experiment meshes; refine/adapt them to matched leaf counts before
comparing timing or error.

## Run with the treecode

Build the current executable from the project root with `make exact`, then run
one mesh by passing its path as argument 1. For a quick mesh-load/evaluation
check with the archived executable, for example:

```sh
OMP_NUM_THREADS=2 ./tree_exact domain_meshes/l_prism.mesh \
  0 2 2 0.5 0 CPU 1.0 1.0 RADIAL 6 0
```

The fields after the mesh path are initial global refinements, quadrature
accuracy `p`, multipole order `P`, MAC threshold, comparison flag, backend,
stage-1 and stage-2 adaptation tolerances, indicator, indicator quadrature
order, and mesh-estimator flag. The example disables comparison and adaptation;
it checks that AFEPack can read the mesh and that the HNL--MAC evaluation runs.

## Regenerate

Run `./generate_meshes.sh` from this directory. It requires Gmsh and the
AFEPack `gmsh2mesh.bin` converter. The generated `.msh` files use MSH 4.1,
which is the format expected by the installed AFEPack converter. Default
characteristic lengths are recorded in the `.geo` files.

Before using these meshes in an article experiment, audit the mesh boundary,
adapt to comparable `N`, and evaluate each treecode result against a direct
reference integrated over that same truncated mesh domain.
