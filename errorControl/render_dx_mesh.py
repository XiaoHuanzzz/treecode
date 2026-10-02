#!/usr/bin/env python3
"""Render an OpenDX tetrahedral mesh as a surface and planar cut."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.collections import PolyCollection
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


OBJECT_RE = re.compile(
    r"object\s+\d+\s+class\s+array.*?shape\s+(\d+)\s+item\s+(\d+)\s+data\s+follows"
)


def read_dx_tetra_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read positions and tetrahedral connectivity from an ASCII OpenDX file."""
    with path.open("r", encoding="utf-8") as stream:
        point_header = stream.readline()
        point_match = OBJECT_RE.search(point_header)
        if point_match is None or int(point_match.group(1)) != 3:
            raise ValueError("The first OpenDX object is not a three-dimensional point array.")
        point_count = int(point_match.group(2))
        points = np.asarray(
            [np.fromstring(stream.readline(), sep=" ") for _ in range(point_count)],
            dtype=float,
        )

        cell_header = stream.readline()
        while cell_header and not cell_header.strip():
            cell_header = stream.readline()
        cell_match = OBJECT_RE.search(cell_header)
        if cell_match is None or int(cell_match.group(1)) != 4:
            raise ValueError("The second OpenDX object is not tetrahedral connectivity.")
        cell_count = int(cell_match.group(2))
        tetrahedra = np.asarray(
            [np.fromstring(stream.readline(), sep=" ", dtype=np.int64) for _ in range(cell_count)],
            dtype=np.int64,
        )

    if points.shape != (point_count, 3):
        raise ValueError(f"Expected {(point_count, 3)} points, found {points.shape}.")
    if tetrahedra.shape != (cell_count, 4):
        raise ValueError(f"Expected {(cell_count, 4)} cells, found {tetrahedra.shape}.")
    if tetrahedra.min() < 0 or tetrahedra.max() >= point_count:
        raise ValueError("Tetrahedral connectivity contains an invalid vertex index.")
    return points, tetrahedra


def boundary_faces(tetrahedra: np.ndarray) -> np.ndarray:
    """Return the triangular faces belonging to exactly one tetrahedron."""
    local_faces = np.array(
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64
    )
    faces = tetrahedra[:, local_faces].reshape(-1, 3)
    canonical_faces = np.sort(faces, axis=1)
    _, first, counts = np.unique(
        canonical_faces, axis=0, return_index=True, return_counts=True
    )
    return faces[first[counts == 1]]


def slice_polygons(
    points: np.ndarray, tetrahedra: np.ndarray, plane_y: float
) -> tuple[list[np.ndarray], np.ndarray]:
    """Intersect tetrahedra with y=plane_y and return polygons in the x-z plane."""
    tetra_points = points[tetrahedra]
    y_values = tetra_points[:, :, 1]
    intersects = (y_values.min(axis=1) < plane_y) & (y_values.max(axis=1) > plane_y)
    candidates = tetra_points[intersects]
    edge_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))

    polygons: list[np.ndarray] = []
    sizes: list[float] = []
    for tetra in candidates:
        intersections = []
        for first, second in edge_pairs:
            y_first = tetra[first, 1]
            y_second = tetra[second, 1]
            if (y_first - plane_y) * (y_second - plane_y) < 0.0:
                fraction = (plane_y - y_first) / (y_second - y_first)
                point = tetra[first] + fraction * (tetra[second] - tetra[first])
                intersections.append(point[[0, 2]])

        if len(intersections) not in (3, 4):
            continue
        polygon = np.asarray(intersections)
        center = polygon.mean(axis=0)
        angles = np.arctan2(polygon[:, 1] - center[1], polygon[:, 0] - center[0])
        polygon = polygon[np.argsort(angles)]
        tetra_edge_lengths = [
            np.linalg.norm(tetra[second] - tetra[first])
            for first, second in edge_pairs
        ]
        polygons.append(polygon)
        sizes.append(float(max(tetra_edge_lengths)))

    return polygons, np.asarray(sizes)


def positive_log_norm(values: np.ndarray) -> colors.LogNorm:
    positive = values[values > 0.0]
    if positive.size == 0:
        raise ValueError("Cannot color the mesh because all element sizes are zero.")
    lower, upper = np.percentile(positive, [2.0, 98.0])
    if lower == upper:
        lower = positive.min()
        upper = positive.max()
    if lower == upper:
        upper = lower * 1.01
    return colors.LogNorm(vmin=lower, vmax=upper)


def render_mesh(
    points: np.ndarray,
    tetrahedra: np.ndarray,
    output: Path,
    plane_y: float,
    dpi: int,
    mesh_label: str,
    line_only: bool,
) -> tuple[int, int]:
    faces = boundary_faces(tetrahedra)
    surface_triangles = points[faces]
    surface_areas = 0.5 * np.linalg.norm(
        np.cross(
            surface_triangles[:, 1] - surface_triangles[:, 0],
            surface_triangles[:, 2] - surface_triangles[:, 0],
        ),
        axis=1,
    )
    surface_norm = positive_log_norm(np.sqrt(surface_areas))

    polygons, slice_sizes = slice_polygons(points, tetrahedra, plane_y)
    slice_norm = positive_log_norm(slice_sizes)
    cmap = plt.get_cmap("viridis_r")

    figure = plt.figure(figsize=(15.5, 7.2), constrained_layout=True)
    grid = figure.add_gridspec(1, 2, width_ratios=(1.03, 1.0))

    axis_3d = figure.add_subplot(grid[0, 0], projection="3d")
    if line_only:
        surface_collection = Poly3DCollection(
            surface_triangles,
            facecolors=(1.0, 1.0, 1.0, 0.0),
            edgecolors=(0.0, 0.0, 0.0, 0.62),
            linewidths=0.18,
        )
    else:
        surface_collection = Poly3DCollection(
            surface_triangles,
            facecolors=cmap(surface_norm(np.sqrt(surface_areas))),
            edgecolors=(0.08, 0.15, 0.22, 0.45),
            linewidths=0.12,
            alpha=0.88,
        )
    axis_3d.add_collection3d(surface_collection)
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    axis_3d.set_xlim(lower[0], upper[0])
    axis_3d.set_ylim(lower[1], upper[1])
    axis_3d.set_zlim(lower[2], upper[2])
    axis_3d.set_box_aspect(upper - lower)
    axis_3d.view_init(elev=24, azim=-52)
    axis_3d.set_xlabel(r"$x$")
    axis_3d.set_ylabel(r"$y$")
    axis_3d.set_zlabel(r"$z$")
    axis_3d.set_title("(a) Boundary triangulation", pad=14, fontsize=13)
    axis_3d.grid(False)
    axis_3d.xaxis.pane.set_alpha(0.0)
    axis_3d.yaxis.pane.set_alpha(0.0)
    axis_3d.zaxis.pane.set_alpha(0.0)

    axis_slice = figure.add_subplot(grid[0, 1])
    if line_only:
        slice_collection = PolyCollection(
            polygons,
            facecolors="none",
            edgecolors=(0.0, 0.0, 0.0, 0.78),
            linewidths=0.24,
        )
    else:
        slice_collection = PolyCollection(
            polygons,
            array=slice_sizes,
            cmap=cmap,
            norm=slice_norm,
            edgecolors=(0.04, 0.09, 0.14, 0.72),
            linewidths=0.20,
        )
    axis_slice.add_collection(slice_collection)
    axis_slice.set_xlim(lower[0], upper[0])
    axis_slice.set_ylim(lower[2], upper[2])
    axis_slice.set_aspect("equal", adjustable="box")
    axis_slice.set_xlabel(r"$x$")
    axis_slice.set_ylabel(r"$z$")
    if abs(plane_y) < 1.0e-4:
        slice_title = r"(b) Central tetrahedral cut ($y\approx0$)"
    else:
        slice_title = rf"(b) Tetrahedral cut at $y={plane_y:g}$"
    axis_slice.set_title(slice_title, pad=10, fontsize=13)
    axis_slice.grid(False)
    if not line_only:
        colorbar = figure.colorbar(
            ScalarMappable(norm=slice_norm, cmap=cmap),
            ax=axis_slice,
            fraction=0.046,
            pad=0.04,
        )
        colorbar.set_label("Local element size on the cut")

    figure.suptitle(
        f"Adaptive tetrahedral mesh: {mesh_label}\n"
        f"{points.shape[0]:,} vertices, {tetrahedra.shape[0]:,} tetrahedra",
        fontsize=16,
    )
    figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return faces.shape[0], len(polygons)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="D.dx", type=Path)
    parser.add_argument("-o", "--output", default="D_mesh.png", type=Path)
    parser.add_argument(
        "--plane-y",
        default=1.0e-6,
        type=float,
        help="y-coordinate of the cut; a small offset avoids vertex degeneracy",
    )
    parser.add_argument("--dpi", default=240, type=int)
    parser.add_argument(
        "--line-only",
        action="store_true",
        help="draw mesh edges only, without face colors or a color bar",
    )
    arguments = parser.parse_args()

    points, tetrahedra = read_dx_tetra_mesh(arguments.input)
    boundary_count, cut_count = render_mesh(
        points,
        tetrahedra,
        arguments.output,
        arguments.plane_y,
        arguments.dpi,
        arguments.input.name,
        arguments.line_only,
    )
    print(
        f"Wrote {arguments.output}: {points.shape[0]} vertices, "
        f"{tetrahedra.shape[0]} tetrahedra, {boundary_count} boundary faces, "
        f"{cut_count} cut polygons."
    )


if __name__ == "__main__":
    main()
