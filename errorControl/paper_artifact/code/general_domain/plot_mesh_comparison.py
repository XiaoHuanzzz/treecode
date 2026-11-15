#!/usr/bin/env python3
"""Plot actual plane sections of adaptive and uniform-reference tetra meshes.

The AFEPack OpenDX export already decomposes twin/four tetrahedron elements
into ordinary tetrahedra. N in each panel comes from the matching run JSON
and remains the active AFEPack element count, not the exported tet count.

Example, from the project root:
    python3 general_domain/plot_mesh_comparison.py
"""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import islice
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np


HERE = Path(__file__).resolve().parent
CASES = (
    ("sphere", "Sphere"),
    ("l_prism", "L-shaped prism"),
    ("perforated_cube", "Perforated cube"),
)
EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
ARRAY_HEADER = re.compile(r"\bshape\s+(\d+)\s+items?\s+(\d+)\s+data follows")


def read_dx(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read AFEPack's ASCII positions and tetrahedron connection arrays."""
    points = None
    tetrahedra = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("object ") or "class array" not in line:
                continue
            match = ARRAY_HEADER.search(line)
            if match is None:
                continue
            shape, count = map(int, match.groups())
            if shape not in (3, 4):
                continue
            dtype = np.int64 if "type int" in line else np.float64
            data = np.loadtxt(islice(stream, count), dtype=dtype, ndmin=2)
            if data.shape != (count, shape):
                raise ValueError(f"Invalid array shape in {path}: {data.shape}")
            if shape == 3 and points is None:
                points = data
            elif shape == 4 and tetrahedra is None:
                tetrahedra = data
    if points is None or tetrahedra is None:
        raise ValueError(f"Missing positions or tetrahedron connections: {path}")
    if tetrahedra.min() < 0 or tetrahedra.max() >= len(points):
        raise ValueError(f"Connection index outside positions array: {path}")
    return points, tetrahedra


def section_segments(
    points: np.ndarray, tetrahedra: np.ndarray, plane: float
) -> tuple[np.ndarray, int]:
    """Intersect each tetrahedron with z=plane; do not project its 3D edges."""
    z = points[tetrahedra, 2] - plane
    tolerance = 2.0e-9 * max(1.0, np.ptp(points, axis=0).max())
    crossing = np.flatnonzero(
        (z.min(axis=1) <= tolerance) & (z.max(axis=1) >= -tolerance)
    )
    segments: Counter = Counter()
    polygon_count = 0
    for index in crossing:
        tet = points[tetrahedra[index]]
        dz = tet[:, 2] - plane
        intersections = []
        for vertex in range(4):
            if abs(dz[vertex]) <= tolerance:
                intersections.append(tet[vertex, :2])
        for a, b in EDGES:
            if (dz[a] < -tolerance and dz[b] > tolerance) or (
                dz[b] < -tolerance and dz[a] > tolerance
            ):
                fraction = -dz[a] / (dz[b] - dz[a])
                intersections.append(tet[a, :2] + fraction * (tet[b, :2] - tet[a, :2]))
        if len(intersections) < 3:
            continue
        polygon = np.unique(np.round(intersections, decimals=9), axis=0)
        if len(polygon) < 3:
            continue
        center = polygon.mean(axis=0)
        angle = np.arctan2(polygon[:, 1] - center[1], polygon[:, 0] - center[0])
        polygon = polygon[np.argsort(angle)]
        area2 = abs(
            np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
            - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))
        )
        if area2 <= tolerance * tolerance:
            continue
        polygon_count += 1
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            endpoints = tuple(sorted((tuple(a), tuple(b))))
            if endpoints[0] != endpoints[1]:
                segments[endpoints] += 1
    if not segments:
        raise ValueError(f"No nondegenerate tetrahedron sections at z={plane}")
    return np.asarray(list(segments), dtype=float), polygon_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=HERE / "runs" / "pilot_20261108")
    parser.add_argument("--adaptive", default="adaptive_large")
    parser.add_argument("--reference", default="uniform_L2_p12")
    parser.add_argument("--plane", type=float, default=0.0)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    output = args.output_prefix or args.run_dir / "mesh_comparison"

    paths = []
    missing = []
    for shape, label in CASES:
        row = []
        for stem in (args.adaptive, args.reference):
            dx = args.run_dir / shape / f"{stem}.dx"
            metadata = args.run_dir / shape / f"{stem}.json"
            row.append((dx, metadata))
            missing.extend(str(path) for path in (dx, metadata) if not path.is_file())
        paths.append(row)
    if missing:
        parser.error("Run outputs are not complete yet:\n" + "\n".join(missing))

    plt.rcParams.update({
        "font.family": "serif", "font.size": 10.5,
        "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42,
        "axes.linewidth": 0.7, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "xtick.major.size": 3, "ytick.major.size": 3,
    })
    fig, axes = plt.subplots(3, 2, figsize=(10.8, 15.0))
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.065, top=0.92,
                        hspace=0.33, wspace=0.19)
    fig.suptitle(
        f"General-domain meshes: actual $z={args.plane:g}$ sections",
        fontsize=16, y=0.982,
    )
    source_style = dict(marker="o", markersize=5.5, markerfacecolor="#b2182b",
                        markeredgecolor="white", markeredgewidth=0.8, linestyle="none")
    source_handle = Line2D([], [], **source_style)
    fig.legend([source_handle], ["Density center (-2, -2, 0)"],
               frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.963), fontsize=10)

    for row, (_, label) in enumerate(CASES):
        for column, kind in enumerate(("adaptive", "uniform reference")):
            dx, metadata = paths[row][column]
            stats = json.loads(metadata.read_text())
            points, tetrahedra = read_dx(dx)
            segments, intersections = section_segments(points, tetrahedra, args.plane)
            axis = axes[row, column]
            axis.add_collection(LineCollection(segments, colors="#4d555d",
                                               linewidths=0.19, rasterized=False))
            if abs(args.plane) <= 1e-12:
                axis.plot([-2.0], [-2.0], zorder=5, **source_style)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlim(-5.15, 5.15)
            axis.set_ylim(-5.15, 5.15)
            axis.set_xticks([-5, 0, 5])
            axis.set_yticks([-5, 0, 5])
            axis.set_xlabel("$x$", labelpad=2)
            axis.set_ylabel("$y$", labelpad=2)
            axis.set_title(
                f"{label}: {kind}\n$N={int(stats['N']):,}$",
                fontsize=11, pad=8,
            )
            axis.grid(False)
            print(f"{dx}: N={int(stats['N'])}, exported tetrahedra={len(tetrahedra)}, "
                  f"section polygons={intersections}, unique lines={len(segments)}")

    fig.text(0.5, 0.025,
             "$N$ counts active AFEPack elements. Lines are tetrahedron–plane intersections, "
             "not projected volume edges.\n"
             "Both mesh families preserve the same starter-domain boundary, including the faceted sphere.",
             ha="center", va="center", fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output.with_suffix(".png"), dpi=190, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Wrote {output.with_suffix('.pdf')} and {output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
