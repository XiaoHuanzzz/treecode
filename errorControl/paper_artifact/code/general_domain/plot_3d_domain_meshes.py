#!/usr/bin/env python3
"""Render article-ready 3D boundary meshes for the irregular-domain study."""

import os
from pathlib import Path
import sys
import tempfile
from collections import defaultdict

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "treecode_general_3d_mpl")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE / "runs" / "irregular_scaling_p2P4_20261109"
OUT = ROOT / "general_domain_meshes_3d"

sys.path.insert(0, str(HERE))
from plot_mesh_comparison import read_dx


CASES = (
    ("sphere", "Sphere"),
    ("l_prism", "L-shaped prism"),
    ("perforated_cube", "Perforated cube"),
)


def boundary_faces(cells):
    """Return triangular faces incident to exactly one tetrahedron."""
    face_count = defaultdict(int)
    face_vertices = {}
    local_faces = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    for cell in np.asarray(cells, dtype=int):
        for ids in local_faces:
            face = tuple(int(cell[i]) for i in ids)
            key = tuple(sorted(face))
            face_count[key] += 1
            face_vertices[key] = face
    return np.asarray(
        [face_vertices[key] for key, count in face_count.items() if count == 1],
        dtype=int,
    )


def set_equal_box(ax, points, padding=0.03):
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = 0.5 * (lower + upper)
    radius = 0.5 * np.max(upper - lower) * (1.0 + padding)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def add_collection(ax, triangles, facecolor, edgecolor, alpha, linewidth):
    collection = Poly3DCollection(
        triangles,
        facecolors=facecolor,
        edgecolors=edgecolor,
        linewidths=linewidth,
        alpha=alpha,
    )
    collection.set_rasterized(False)
    ax.add_collection3d(collection)


def draw_case(ax, case, label):
    points, cells = read_dx(ROOT / case / "L0.dx")
    faces = boundary_faces(cells)
    triangles = points[faces]
    centroids = triangles.mean(axis=1)

    if case == "perforated_cube":
        # The cavity is [-1,1]^3.  Draw its boundary more strongly while
        # retaining the exterior surface as a translucent wireframe.
        cavity = np.max(np.abs(centroids), axis=1) <= 1.001
        add_collection(
            ax,
            triangles[~cavity],
            facecolor="#d9e4ec",
            edgecolor="#4f5d68",
            alpha=0.075,
            linewidth=0.23,
        )
        add_collection(
            ax,
            triangles[cavity],
            facecolor="#e9a06a",
            edgecolor="#9d3f27",
            alpha=0.72,
            linewidth=0.45,
        )
    else:
        add_collection(
            ax,
            triangles,
            facecolor="#d9e4ec",
            edgecolor="#4f5d68",
            alpha=0.22,
            linewidth=0.25,
        )

    ax.scatter(
        [-2.0], [-2.0], [0.0],
        s=23,
        c="#b2182b",
        edgecolors="white",
        linewidths=0.6,
        depthshade=False,
        zorder=10,
    )
    set_equal_box(ax, points)
    ax.view_init(elev=23, azim=-52)
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    ax.text2D(
        0.5,
        -0.035,
        f"{label}  ($N={len(cells):,}$)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.5,
    )


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "dejavuserif",
            "font.size": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(11.4, 4.05))
    axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
    for ax, (case, label) in zip(axes, CASES):
        draw_case(ax, case, label)

    fig.subplots_adjust(left=0.015, right=0.995, bottom=0.09, top=0.995, wspace=0.015)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT.with_suffix(".png"), dpi=280, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote {OUT.with_suffix('.pdf')}")
    print(f"Wrote {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
