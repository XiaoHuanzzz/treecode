#!/usr/bin/env python3
"""Generate three overlapping atom-centered tetrahedral meshes for H2O.

The meshes have independent vertices and connectivity.  They overlap in the
same molecular coordinate system and are collected as three fields in one
OpenDX group.  This is an experimental alternative to standard atom-centered
radial/Lebedev quadrature.
"""

from dataclasses import dataclass
from math import cos, pi, sin, sqrt
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, Delaunay


ANGSTROM_TO_BOHR = 1.8897261246257702
GOLDEN_ANGLE = pi * (3.0 - sqrt(5.0))


@dataclass(frozen=True)
class AtomMeshSpecification:
    name: str
    grid_id: int
    center: tuple
    radial_shells: int
    angular_points: int
    radius: float


SPECIFICATIONS = (
    AtomMeshSpecification("O", 0, (0.0, 0.0, 0.0), 12, 194, 3.4),
    AtomMeshSpecification(
        "H1",
        1,
        (
            0.756950327 * ANGSTROM_TO_BOHR,
            0.0,
            0.585882276 * ANGSTROM_TO_BOHR,
        ),
        10,
        110,
        2.7,
    ),
    AtomMeshSpecification(
        "H2",
        2,
        (
            -0.756950327 * ANGSTROM_TO_BOHR,
            0.0,
            0.585882276 * ANGSTROM_TO_BOHR,
        ),
        10,
        110,
        2.7,
    ),
)


def fibonacci_directions(count, shell_index):
    directions = np.empty((count, 3), dtype=np.float64)
    phase = 0.371 * shell_index
    for index in range(count):
        z = 1.0 - 2.0 * (index + 0.5) / count
        radius_xy = sqrt(max(0.0, 1.0 - z * z))
        azimuth = (index + phase) * GOLDEN_ANGLE
        directions[index] = (
            radius_xy * cos(azimuth), radius_xy * sin(azimuth), z
        )
    return directions


def radial_shells(count, extent):
    parameter = np.linspace(0.035, 1.0, count)
    return extent * parameter**1.65


def build_points(specification):
    center = np.asarray(specification.center, dtype=np.float64)
    shells = [center]
    for shell_index, radius in enumerate(
        radial_shells(specification.radial_shells, specification.radius)
    ):
        directions = fibonacci_directions(
            specification.angular_points, shell_index
        )
        shells.extend(center + radius * directions)
    return np.asarray(shells, dtype=np.float64)


def orient_and_filter_tetrahedra(points, simplices):
    valid = np.all(simplices < points.shape[0], axis=1)
    simplices = simplices[valid].copy()
    edge_matrix = np.stack(
        (
            points[simplices[:, 1]] - points[simplices[:, 0]],
            points[simplices[:, 2]] - points[simplices[:, 0]],
            points[simplices[:, 3]] - points[simplices[:, 0]],
        ),
        axis=1,
    )
    signed_six_volume = np.linalg.det(edge_matrix)
    nondegenerate = np.abs(signed_six_volume) > 1.0e-13
    simplices = simplices[nondegenerate]
    signed_six_volume = signed_six_volume[nondegenerate]
    negative = signed_six_volume < 0.0
    temporary = simplices[negative, 2].copy()
    simplices[negative, 2] = simplices[negative, 3]
    simplices[negative, 3] = temporary
    return simplices


def tetrahedron_volumes(points, tetrahedra):
    matrices = np.stack(
        (
            points[tetrahedra[:, 1]] - points[tetrahedra[:, 0]],
            points[tetrahedra[:, 2]] - points[tetrahedra[:, 0]],
            points[tetrahedra[:, 3]] - points[tetrahedra[:, 0]],
        ),
        axis=1,
    )
    return np.abs(np.linalg.det(matrices)) / 6.0


def becke_polynomial(value):
    return 1.5 * value - 0.5 * value**3


def fuzzy_weights(points):
    centers = np.asarray(
        [specification.center for specification in SPECIFICATIONS],
        dtype=np.float64,
    )
    distances = np.linalg.norm(
        points[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2
    )
    products = np.ones_like(distances)
    for atom_a in range(len(SPECIFICATIONS)):
        for atom_b in range(len(SPECIFICATIONS)):
            if atom_a == atom_b:
                continue
            nuclear_distance = np.linalg.norm(
                centers[atom_a] - centers[atom_b]
            )
            mu = (distances[:, atom_a] - distances[:, atom_b]) \
                / nuclear_distance
            mu = np.clip(mu, -1.0, 1.0)
            for _ in range(3):
                mu = becke_polynomial(mu)
            products[:, atom_a] *= 0.5 * (1.0 - mu)
    denominator = products.sum(axis=1)
    if np.any(denominator <= 0.0):
        raise RuntimeError("Non-positive fuzzy-cell normalization")
    return products / denominator[:, np.newaxis]


def generate_mesh(specification):
    points = build_points(specification)
    triangulation = Delaunay(points, qhull_options="Qbb Qc Q12 QJ")
    tetrahedra = orient_and_filter_tetrahedra(points, triangulation.simplices)
    volumes = tetrahedron_volumes(points, tetrahedra)
    hull_volume = ConvexHull(points).volume
    relative_volume_error = abs(volumes.sum() - hull_volume) / hull_volume
    weights = fuzzy_weights(points)[:, specification.grid_id]
    return points, tetrahedra, weights, volumes, hull_volume, relative_volume_error


def write_array(stream, object_id, values, type_name):
    if values.ndim == 1:
        stream.write(
            f"object {object_id} class array type {type_name} rank 0 "
            f"items {values.shape[0]} data follows\n"
        )
        if np.issubdtype(values.dtype, np.integer):
            for offset in range(0, values.shape[0], 24):
                stream.write(
                    " ".join(str(int(value)) for value in values[offset:offset + 24])
                    + "\n"
                )
        else:
            for offset in range(0, values.shape[0], 8):
                stream.write(
                    " ".join(
                        f"{float(value):.15e}"
                        for value in values[offset:offset + 8]
                    )
                    + "\n"
                )
    else:
        stream.write(
            f"object {object_id} class array type {type_name} rank 1 "
            f"shape {values.shape[1]} items {values.shape[0]} data follows\n"
        )
        for row in values:
            if np.issubdtype(values.dtype, np.integer):
                stream.write(" ".join(str(int(value)) for value in row) + "\n")
            else:
                stream.write(
                    " ".join(f"{float(value):.15e}" for value in row) + "\n"
                )


def write_opendx(output, meshes):
    with output.open("w", encoding="ascii") as stream:
        stream.write("# Three overlapping atom-centered H2O tetrahedral meshes\n")
        stream.write("# Units: bohr\n")
        stream.write("# Independent connectivity: no tetrahedron joins different grids.\n")
        stream.write("# Equal-size Becke fuzzy weights are stored as field data.\n")

        field_names = []
        for mesh_index, (specification, mesh) in enumerate(
            zip(SPECIFICATIONS, meshes)
        ):
            points, tetrahedra, weights, volumes, hull_volume, error = mesh
            base = 1 + 4 * mesh_index
            positions_id = base
            connections_id = base + 1
            weight_id = base + 2
            grid_id_object = base + 3
            field_name = f"{specification.name}_tetra_mesh"
            field_names.append(field_name)

            stream.write(
                f"# {specification.name}: center={specification.center}, "
                f"vertices={points.shape[0]}, tetrahedra={tetrahedra.shape[0]}, "
                f"hull_volume={hull_volume:.15e}, "
                f"partition_error={error:.3e}\n"
            )
            write_array(stream, positions_id, points, "double")
            write_array(stream, connections_id, tetrahedra, "int")
            stream.write('attribute "element type" string "tetrahedra"\n')
            stream.write('attribute "ref" string "positions"\n')
            write_array(stream, weight_id, weights, "double")
            stream.write('attribute "dep" string "positions"\n')
            grid_ids = np.full(
                points.shape[0], specification.grid_id, dtype=np.int64
            )
            write_array(stream, grid_id_object, grid_ids, "int")
            stream.write('attribute "dep" string "positions"\n')
            stream.write(f'object "{field_name}" class field\n')
            stream.write(f'component "positions" value {positions_id}\n')
            stream.write(f'component "connections" value {connections_id}\n')
            stream.write(f'component "data" value {weight_id}\n')
            stream.write(f'component "grid_id" value {grid_id_object}\n')

        stream.write('object "H2O_three_tetrahedral_meshes" class group\n')
        for specification, field_name in zip(SPECIFICATIONS, field_names):
            stream.write(
                f'member "{specification.name}" value "{field_name}"\n'
            )
        stream.write("end\n")


def write_single_mesh_opendx(output, specification, mesh):
    points, tetrahedra, weights, volumes, hull_volume, error = mesh
    with output.open("w", encoding="ascii") as stream:
        stream.write(
            f"# H2O {specification.name}-centered tetrahedral mesh\n"
        )
        stream.write("# Units: bohr\n")
        stream.write(
            f"# center={specification.center}, vertices={points.shape[0]}, "
            f"tetrahedra={tetrahedra.shape[0]}, "
            f"volume={volumes.sum():.15e}, "
            f"hull_volume={hull_volume:.15e}, "
            f"partition_error={error:.3e}\n"
        )
        stream.write("# Field data contains the equal-size Becke fuzzy weight.\n")
        write_array(stream, 1, points, "double")
        write_array(stream, 2, tetrahedra, "int")
        stream.write('attribute "element type" string "tetrahedra"\n')
        stream.write('attribute "ref" string "positions"\n')
        write_array(stream, 3, weights, "double")
        stream.write('attribute "dep" string "positions"\n')
        grid_ids = np.full(
            points.shape[0], specification.grid_id, dtype=np.int64
        )
        write_array(stream, 4, grid_ids, "int")
        stream.write('attribute "dep" string "positions"\n')
        stream.write(f'object "{specification.name}_tetra_mesh" class field\n')
        stream.write('component "positions" value 1\n')
        stream.write('component "connections" value 2\n')
        stream.write('component "data" value 3\n')
        stream.write('component "grid_id" value 4\n')
        stream.write("end\n")


def main():
    meshes = []
    for specification in SPECIFICATIONS:
        mesh = generate_mesh(specification)
        meshes.append(mesh)
        points, tetrahedra, weights, volumes, hull_volume, error = mesh
        print(
            f"{specification.name}: vertices={points.shape[0]}, "
            f"tetrahedra={tetrahedra.shape[0]}, "
            f"volume={volumes.sum():.12g}, "
            f"hull-volume relative error={error:.3e}, "
            f"fuzzy weight range=[{weights.min():.6g}, {weights.max():.6g}]"
        )
        if error > 1.0e-10:
            raise RuntimeError(
                f"{specification.name} tetrahedra do not cover their convex hull"
            )

    output = Path(__file__).resolve().with_name(
        "h2o_three_tetrahedral_meshes.dx"
    )
    write_opendx(output, meshes)
    print(f"Wrote {output}")
    for specification, mesh in zip(SPECIFICATIONS, meshes):
        individual_output = Path(__file__).resolve().with_name(
            f"h2o_{specification.name}_tetra_mesh.dx"
        )
        write_single_mesh_opendx(individual_output, specification, mesh)
        print(f"Wrote {individual_output}")


if __name__ == "__main__":
    main()
