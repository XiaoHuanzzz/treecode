#!/usr/bin/env python3
"""Show actual AFEPack mesh sections, without changing the pilot figures."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
import numpy as np

from plot_mesh_comparison import read_dx, section_segments

HERE = Path(__file__).resolve().parent
CASES = (("sphere", "Sphere"), ("l_prism", "L-shaped prism"),
         ("perforated_cube", "Perforated cube"))
CENTER = np.array([-2., -2., 0.])


def load_case(prefix):
    points, cells = read_dx(prefix.with_suffix(".dx"))
    metadata = json.loads(prefix.with_suffix(".json").read_text())
    tetrahedra = points[cells]
    vectors = tetrahedra[:, 1:] - tetrahedra[:, :1]
    volumes = np.abs(np.linalg.det(vectors)) / 6.
    if not np.all(np.isfinite(volumes)) or np.any(volumes <= 0):
        raise ValueError(f"Degenerate/nonfinite tetrahedra in {prefix}")
    radius = np.linalg.norm(tetrahedra.mean(axis=1) - CENTER, axis=1)
    near, far = radius < 1., radius > 3.
    size = np.cbrt(volumes)
    core_size = float(np.median(size[near]))
    far_size = float(np.median(size[far]))
    segments, count = section_segments(points, cells, 0.)
    stats = dict(N=int(metadata["N"]), exported_tetrahedra=len(cells),
                 volume=float(volumes.sum()), section_polygons=count,
                 core_median_size=core_size, far_median_size=far_size,
                 far_core_ratio=far_size/core_size,
                 core_tetrahedra=int(near.sum()), far_tetrahedra=int(far.sum()))
    return segments, stats


def draw(axis, segments, stats, title):
    axis.add_collection(LineCollection(segments, colors="#46515a",
                                      linewidths=.25))
    axis.plot([-2], [-2], "o", color="#ba2034", markeredgecolor="white",
              markeredgewidth=.7, markersize=5., zorder=5)
    axis.set(xlim=(-5.15, 5.15), ylim=(-5.15, 5.15), aspect="equal",
             xlabel="$x$", ylabel="$y$", xticks=[-5, 0, 5], yticks=[-5, 0, 5])
    axis.set_title(f"{title}\n$N={stats['N']:,}$", fontsize=13, pad=10)
    axis.tick_params(labelsize=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path,
                        default=HERE / "runs" / "localized_meshes_20261108")
    parser.add_argument("--baseline", type=Path,
                        default=HERE / "runs" / "pilot_20261108")
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Serif", "mathtext.fontset": "dejavuserif",
                         "axes.labelsize": 12, "pdf.fonttype": 42})
    meshes, stats = {}, {}
    for case, label in CASES:
        old = load_case(args.baseline / case / "adaptive_large")
        new = load_case(args.directory / case / "localized")
        rel_volume = abs(new[1]["volume"]-old[1]["volume"])/old[1]["volume"]
        if rel_volume > 1e-7:
            raise ValueError(f"Domain volume changed for {case}: {rel_volume}")
        meshes[case] = (old, new)
        stats[case] = dict(previous=old[1], localized=new[1],
                           relative_volume_difference=rel_volume)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.7))
    fig.subplots_adjust(left=.045, right=.99, bottom=.17, top=.82, wspace=.24)
    for axis, (case, label) in zip(axes, CASES):
        segments, data = meshes[case][1]
        draw(axis, segments, data, label)
    fig.suptitle("Concentrated adaptive meshes: actual $z=0$ sections", y=.98, fontsize=17)
    fig.text(.5, .07, "Red dot: density center $(-2,-2,0)$. Same starter-domain boundaries as the pilot.\n"
             "Mesh-only study: stronger refinement contrast does not establish smaller potential error.",
             ha="center", va="center", fontsize=10)
    fig.savefig(args.directory / "localized_meshes.png", dpi=180)
    fig.savefig(args.directory / "localized_meshes.pdf")
    plt.close(fig)

    with PdfPages(args.directory / "mesh_before_after.pdf") as pdf:
        for case, label in CASES:
            old, new = meshes[case]
            fig, axes = plt.subplots(1, 2, figsize=(11.7, 8.3))
            fig.subplots_adjust(left=.07, right=.97, bottom=.23, top=.82, wspace=.23)
            draw(axes[0], *old, "Previous radial monitor")
            draw(axes[1], *new, "More localized monitor")
            fig.suptitle(f"{label}: local refinement around the source", y=.96, fontsize=18)
            fig.text(.5, .14,
                     "Median far/core cell-size ratio: "
                     f"{old[1]['far_core_ratio']:.2f} (previous) to {new[1]['far_core_ratio']:.2f} (localized).\n"
                     r"Size = $|K|^{1/3}$ on exported tetrahedra; core $r<1$, far region $r>3$." "\n"
                     "The pictures use the same coordinates and line width. Element counts are not matched.\n"
                     "These are mesh illustrations, not a new accuracy or performance comparison.",
                     ha="center", va="center", fontsize=10)
            pdf.savefig(fig)
            fig.savefig(args.directory / f"{case}_before_after.png", dpi=160)
            plt.close(fig)
    (args.directory / "mesh_contrast.json").write_text(json.dumps(stats, indent=2)+"\n")
    for case, data in stats.items():
        print(case, json.dumps(data))
    print("Wrote localized_meshes.png/.pdf and mesh_before_after.pdf")


if __name__ == "__main__":
    main()
