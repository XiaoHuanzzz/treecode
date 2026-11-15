#!/usr/bin/env python3
"""Plot actual z=0 sections of the scaling-study starter meshes."""
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "treecode_scaling_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE / "runs" / "irregular_scaling_20261108"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "radial_shell_meshes"))
from plot_mesh_comparison import read_dx, section_segments
from plot_shell_meshes import shell_section_segments

CASES = (("sphere", "Sphere"), ("l_prism", "L-shaped prism"),
         ("perforated_cube", "Perforated cube"))

plt.rcParams.update({"font.family": "DejaVu Serif", "mathtext.fontset": "dejavuserif",
                     "font.size": 10, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.75))
for ax, (case, title) in zip(axes, CASES):
    points, cells = read_dx(ROOT / case / "L0.dx")
    segments, _ = section_segments(points, cells, 0.0)
    shell = shell_section_segments(points, cells, 1.15)
    ax.add_collection(LineCollection(segments, colors="#52606d", linewidths=0.28))
    ax.add_collection(LineCollection(shell, colors="#b2182b", linewidths=1.5, zorder=3))
    ax.plot([-2], [-2], "o", color="#b2182b", markersize=4.5,
            markeredgecolor="white", markeredgewidth=0.6, zorder=4)
    ax.set_xlim(-5.15, 5.15); ax.set_ylim(-5.15, 5.15)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([-5, 0, 5]); ax.set_yticks([-5, 0, 5])
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    n = sum(1 for _ in cells)
    ax.set_title(f"{title}\n$N_0={n:,}$", pad=5)
fig.suptitle("Starter meshes for the four-level irregular-domain study", y=0.99, fontsize=13)
fig.text(0.5, 0.01,
         "Red: conforming interface $|\\mathbf{x}-(-2,-2,0)|=1.15$; gray: tetrahedron–plane intersections.",
         ha="center", fontsize=9)
fig.tight_layout(rect=(0, 0.055, 1, 0.94))
fig.savefig(ROOT / "scaling_meshes.pdf", bbox_inches="tight")
fig.savefig(ROOT / "scaling_meshes.png", dpi=220, bbox_inches="tight")
