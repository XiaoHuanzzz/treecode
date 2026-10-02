#!/usr/bin/env python3
"""Generate an OpenDX point field for the proposed three H2O source grids."""

from math import cos, pi, sin, sqrt
from pathlib import Path


ANGSTROM_TO_BOHR = 1.8897261246257702
GOLDEN_ANGLE = pi * (3.0 - sqrt(5.0))

# name, grid id, center in bohr, radial shells, angular points, display radius
GRIDS = (
    ("O", 0, (0.0, 0.0, 0.0), 9, 110, 3.4),
    (
        "H1",
        1,
        (
            0.756950327 * ANGSTROM_TO_BOHR,
            0.0,
            0.585882276 * ANGSTROM_TO_BOHR,
        ),
        7,
        74,
        2.7,
    ),
    (
        "H2",
        2,
        (
            -0.756950327 * ANGSTROM_TO_BOHR,
            0.0,
            0.585882276 * ANGSTROM_TO_BOHR,
        ),
        7,
        74,
        2.7,
    ),
)


def fibonacci_directions(count):
    """Approximately uniform directions on the unit sphere."""
    for index in range(count):
        z = 1.0 - 2.0 * (index + 0.5) / count
        radius_xy = sqrt(max(0.0, 1.0 - z * z))
        azimuth = index * GOLDEN_ANGLE
        yield radius_xy * cos(azimuth), radius_xy * sin(azimuth), z


def radial_shells(count, extent):
    """Nonuniform illustrative shells concentrated near the nucleus."""
    for index in range(count):
        transformed = 0.08 + 0.80 * index / max(1, count - 1)
        yield extent * transformed**1.65


def generate_points():
    points = []
    grid_ids = []
    counts = {}
    for name, grid_id, center, shell_count, angular_count, extent in GRIDS:
        directions = tuple(fibonacci_directions(angular_count))
        start = len(points)
        for radius in radial_shells(shell_count, extent):
            for direction in directions:
                points.append(
                    tuple(
                        center[coordinate] + radius * direction[coordinate]
                        for coordinate in range(3)
                    )
                )
                grid_ids.append(grid_id)
        counts[name] = len(points) - start
    return points, grid_ids, counts


def write_opendx(output, points, grid_ids, counts):
    with output.open("w", encoding="ascii") as stream:
        stream.write("# Proposed H2O three-grid fuzzy-cell geometry\n")
        stream.write("# Units: bohr\n")
        stream.write("# data/grid_id: 0=O, 1=H1, 2=H2\n")
        stream.write(
            "# Illustrative radial-shell/Fibonacci-sphere points; "
            "not final Lebedev quadrature data.\n"
        )
        for name, _, center, _, _, _ in GRIDS:
            stream.write(
                f"# {name} center: {center[0]:.15e} "
                f"{center[1]:.15e} {center[2]:.15e}; "
                f"points={counts[name]}\n"
            )

        stream.write(
            f"object 1 class array type double rank 1 shape 3 "
            f"items {len(points)} data follows\n"
        )
        for x, y, z in points:
            stream.write(f"{x:.15e} {y:.15e} {z:.15e}\n")

        stream.write(
            f"object 2 class array type int rank 0 "
            f"items {len(grid_ids)} data follows\n"
        )
        for offset in range(0, len(grid_ids), 24):
            stream.write(
                " ".join(str(value) for value in grid_ids[offset:offset + 24])
                + "\n"
            )
        stream.write('attribute "dep" string "positions"\n')

        stream.write('object "H2O_three_grids" class field\n')
        stream.write('component "positions" value 1\n')
        stream.write('component "data" value 2\n')
        stream.write('attribute "name" string "H2O atom-centered grid id"\n')
        stream.write("end\n")


def main():
    points, grid_ids, counts = generate_points()
    output = Path(__file__).resolve().with_name("h2o_three_grids.dx")
    write_opendx(output, points, grid_ids, counts)
    print(f"Wrote {output}")
    print(
        "Points: "
        + ", ".join(f"{name}={counts[name]}" for name, *_ in GRIDS)
        + f", total={len(points)}"
    )


if __name__ == "__main__":
    main()
