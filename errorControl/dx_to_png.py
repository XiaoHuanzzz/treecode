#!/usr/bin/env python3
"""Render an ASCII OpenDX mesh or point field as a PNG image.

Examples:
    python3 dx_to_png.py h2o_three_grids.dx
    python3 dx_to_png.py D.dx D_mesh_new.png --view xz
    python3 dx_to_png.py D.dx D_mesh_3d.png --view 3d --max-cells 30000
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np


ARRAY_HEADER = re.compile(
    r"^object\s+(\S+)\s+class\s+array\s+type\s+(\w+)\s+"
    r"rank\s+(\d+)(?:\s+shape\s+(\d+))?\s+"
    r"items?\s+(\d+)\s+data\s+follows\s*$"
)


def read_ascii_dx(path):
    """Read numeric array objects from the ASCII OpenDX subset used here."""
    lines = path.read_text(encoding="ascii").splitlines()
    arrays = {}
    index = 0
    while index < len(lines):
        match = ARRAY_HEADER.match(lines[index].strip())
        if match is None:
            index += 1
            continue

        object_id, type_name, rank_text, shape_text, item_text = match.groups()
        rank = int(rank_text)
        shape = int(shape_text) if shape_text is not None else 1
        item_count = int(item_text)
        value_count = item_count * (shape if rank > 0 else 1)
        values = []
        index += 1
        while index < len(lines) and len(values) < value_count:
            stripped = lines[index].strip()
            if stripped and not stripped.startswith("#"):
                values.extend(stripped.split())
            index += 1
        if len(values) != value_count:
            raise ValueError(
                f"OpenDX object {object_id} expected {value_count} values "
                f"but contained {len(values)}"
            )

        integer_type = type_name.lower() in {
            "byte", "short", "int", "integer", "long", "ubyte", "ushort",
            "uint", "ulong",
        }
        dtype = np.int64 if integer_type else np.float64
        array = np.asarray(values, dtype=dtype)
        if rank > 0:
            array = array.reshape(item_count, shape)
        else:
            array = array.reshape(item_count)
        arrays[object_id] = {
            "values": array,
            "type": type_name,
            "rank": rank,
            "shape": shape,
        }
    return arrays


def identify_geometry(arrays):
    positions = None
    connections = None
    point_data = None
    for entry in arrays.values():
        values = entry["values"]
        if entry["rank"] == 1 and entry["shape"] == 3 \
                and np.issubdtype(values.dtype, np.floating):
            if positions is None:
                positions = values
        elif entry["rank"] == 1 and entry["shape"] in (4, 8) \
                and np.issubdtype(values.dtype, np.integer):
            if connections is None:
                connections = values

    if positions is None:
        raise ValueError("No floating-point rank-1 shape-3 position array found")

    for entry in arrays.values():
        values = entry["values"]
        if entry["rank"] == 0 and values.size == positions.shape[0]:
            point_data = values
            break
    return positions, connections, point_data


def selected_cells(connections, maximum):
    if connections is None or connections.shape[0] <= maximum:
        return connections
    indices = np.linspace(
        0, connections.shape[0] - 1, maximum, dtype=np.int64
    )
    return connections[indices]


def unique_edges(connections):
    if connections is None or connections.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    vertex_count = connections.shape[1]
    pairs = [
        (first, second)
        for first in range(vertex_count)
        for second in range(first + 1, vertex_count)
    ]
    edges = np.concatenate(
        [connections[:, pair] for pair in pairs], axis=0
    )
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def equalize_3d_axes(axis, positions):
    minimum = positions.min(axis=0)
    maximum = positions.max(axis=0)
    center = 0.5 * (minimum + maximum)
    half_extent = 0.5 * max(float(np.max(maximum - minimum)), 1.0e-12)
    axis.set_xlim(center[0] - half_extent, center[0] + half_extent)
    axis.set_ylim(center[1] - half_extent, center[1] + half_extent)
    axis.set_zlim(center[2] - half_extent, center[2] + half_extent)
    axis.set_box_aspect((1.0, 1.0, 1.0))


def add_grid_legend(axis):
    colors = ("#d62728", "#1f77b4", "#2ca02c")
    labels = ("O grid (0)", "H1 grid (1)", "H2 grid (2)")
    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", color=color,
            markersize=6, label=label
        )
        for color, label in zip(colors, labels)
    ]
    axis.legend(handles=handles, loc="best", framealpha=0.92)


def render_3d(positions, connections, point_data, arguments):
    figure = plt.figure(figsize=(9.2, 8.0))
    axis = figure.add_subplot(111, projection="3d")
    cells = selected_cells(connections, arguments.max_cells)
    edges = unique_edges(cells)
    if edges.size:
        segments = positions[edges]
        axis.add_collection3d(
            Line3DCollection(
                segments, colors="#2463a5", linewidths=0.20, alpha=0.24
            )
        )
    else:
        if point_data is None:
            axis.scatter(
                positions[:, 0], positions[:, 1], positions[:, 2],
                s=arguments.point_size, color="#2463a5", alpha=0.62,
                depthshade=False,
            )
        else:
            color_map = ListedColormap(("#d62728", "#1f77b4", "#2ca02c"))
            axis.scatter(
                positions[:, 0], positions[:, 1], positions[:, 2],
                c=point_data, cmap=color_map, vmin=-0.5, vmax=2.5,
                s=arguments.point_size, alpha=0.58, depthshade=False,
            )
            if set(np.unique(point_data)).issubset({0, 1, 2}):
                add_grid_legend(axis)

    equalize_3d_axes(axis, positions)
    axis.set_xlabel("x (bohr)")
    axis.set_ylabel("y (bohr)")
    axis.set_zlabel("z (bohr)")
    axis.view_init(elev=arguments.elevation, azim=arguments.azimuth)
    return figure, axis


def render_projection(positions, connections, point_data, arguments):
    coordinate_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    first, second = coordinate_map[arguments.view]
    labels = "xyz"
    figure, axis = plt.subplots(figsize=(8.4, 7.4))
    cells = selected_cells(connections, arguments.max_cells)
    edges = unique_edges(cells)
    projected = positions[:, (first, second)]
    if edges.size:
        axis.add_collection(
            LineCollection(
                projected[edges], colors="#2463a5", linewidths=0.22,
                alpha=0.28
            )
        )
    else:
        if point_data is None:
            axis.scatter(
                projected[:, 0], projected[:, 1],
                s=arguments.point_size, color="#2463a5", alpha=0.58,
                linewidths=0,
            )
        else:
            color_map = ListedColormap(("#d62728", "#1f77b4", "#2ca02c"))
            axis.scatter(
                projected[:, 0], projected[:, 1], c=point_data,
                cmap=color_map, vmin=-0.5, vmax=2.5,
                s=arguments.point_size, alpha=0.58, linewidths=0,
            )
            if set(np.unique(point_data)).issubset({0, 1, 2}):
                add_grid_legend(axis)
    axis.autoscale()
    axis.margins(0.025)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(f"{labels[first]} (bohr)")
    axis.set_ylabel(f"{labels[second]} (bohr)")
    axis.grid(color="#dddddd", linewidth=0.45, alpha=0.65)
    return figure, axis


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Render an ASCII OpenDX tetrahedral mesh or point field"
    )
    parser.add_argument("input", type=Path, help="input ASCII .dx file")
    parser.add_argument(
        "output", type=Path, nargs="?",
        help="output PNG (default: INPUT_STEM_render.png)",
    )
    parser.add_argument(
        "--view", choices=("3d", "xy", "xz", "yz"), default="3d",
        help="rendering view (default: 3d)",
    )
    parser.add_argument(
        "--max-cells", type=int, default=30000,
        help="maximum cells rendered from a large connected mesh",
    )
    parser.add_argument("--point-size", type=float, default=5.0)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--elevation", type=float, default=22.0)
    parser.add_argument("--azimuth", type=float, default=-58.0)
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    if arguments.max_cells <= 0:
        raise ValueError("--max-cells must be positive")
    if not arguments.input.is_file():
        raise FileNotFoundError(arguments.input)
    if arguments.output is None:
        arguments.output = arguments.input.with_name(
            arguments.input.stem + "_render.png"
        )

    arrays = read_ascii_dx(arguments.input)
    positions, connections, point_data = identify_geometry(arrays)
    if connections is not None:
        if connections.min() < 0 or connections.max() >= positions.shape[0]:
            raise ValueError("OpenDX connection index is outside positions")

    if arguments.view == "3d":
        figure, axis = render_3d(
            positions, connections, point_data, arguments
        )
    else:
        figure, axis = render_projection(
            positions, connections, point_data, arguments
        )
    geometry_name = "tetrahedral mesh" if connections is not None else "point grids"
    axis.set_title(
        f"{arguments.input.name}: {geometry_name} ({arguments.view} view)",
        pad=14,
    )
    figure.tight_layout()
    figure.savefig(arguments.output, dpi=arguments.dpi, bbox_inches="tight")
    plt.close(figure)
    connection_count = 0 if connections is None else connections.shape[0]
    print(
        f"Read {positions.shape[0]} positions and {connection_count} cells; "
        f"wrote {arguments.output}"
    )


if __name__ == "__main__":
    main()
