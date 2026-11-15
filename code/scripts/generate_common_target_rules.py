#!/usr/bin/env python3
"""Generate nested-box tensor Gauss rules on [-20,20]^3."""

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiment_results" / "fixed_adaptive_uniform_20261109"
BREAKS = np.array([-20.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 20.0])


def one_dimensional_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    all_nodes = []
    all_weights = []
    for left, right in zip(BREAKS[:-1], BREAKS[1:]):
        all_nodes.extend(0.5 * (right - left) * nodes + 0.5 * (right + left))
        all_weights.extend(0.5 * (right - left) * weights)
    return np.asarray(all_nodes), np.asarray(all_weights)


def write_rule(order: int) -> None:
    nodes, weights = one_dimensional_rule(order)
    path = OUT / f"common_targets_q{order}.txt"
    with path.open("w") as stream:
        for i, x in enumerate(nodes):
            for j, y in enumerate(nodes):
                for k, z in enumerate(nodes):
                    stream.write(
                        f"{x:.17g} {y:.17g} {z:.17g} "
                        f"{weights[i] * weights[j] * weights[k]:.17g}\n"
                    )
    print(path, len(nodes) ** 3, weights.sum() ** 3)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for quadrature_order in (4, 6, 8, 10):
        write_rule(quadrature_order)
