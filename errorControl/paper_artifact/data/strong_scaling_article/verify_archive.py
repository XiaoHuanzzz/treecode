#!/usr/bin/env python3
"""Verify the recovered strong-scaling records without rerunning the solver."""

import csv
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED_THREADS = (1, 2, 4, 8, 16, 32, 64)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_result(path: Path) -> dict[str, str]:
    lines = [line.split() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) != 2 or len(lines[0]) != len(lines[1]):
        raise AssertionError(f"{path}: expected one complete result row")
    return dict(zip(lines[0], lines[1]))


def main() -> None:
    provenance = json.loads((ROOT / "provenance.json").read_text())
    expected_hashes = provenance["source_snapshot_sha256"]
    for name, expected in expected_hashes.items():
        if name == "tree_exact":
            continue
        actual = sha256(ROOT / "source_snapshot" / name)
        assert actual == expected, f"source hash mismatch: {name}"

    measured: dict[int, list[float]] = {thread: [] for thread in EXPECTED_THREADS}
    for thread in EXPECTED_THREADS:
        for repeat in (1, 2, 3):
            run = ROOT / "raw" / f"t{thread}_r{repeat}"
            manifest = json.loads((run / "manifest.json").read_text())
            result = read_result(run / "results.txt")
            key = manifest["key"]
            assert key["environment"]["OMP_NUM_THREADS"] == str(thread)
            assert key["hashes"]["tree_exact"] == expected_hashes["tree_exact"]
            for name, expected in expected_hashes.items():
                assert key["hashes"][name] == expected
            assert int(result["n_elements"]) == provenance["N"]
            assert int(result["p"]) == provenance["p"]
            assert int(result["P"]) == provenance["P"]
            assert float(result["theta"]) == provenance["theta"]
            assert int(result["omp_threads_used"]) == thread
            measured[thread].append(float(result["potential_evaluation_time_s"]))

    with (ROOT / "article_summary.csv").open(newline="") as stream:
        summary = {int(row["threads"]): row for row in csv.DictReader(stream)}
    baseline = statistics.median(measured[1])
    for thread in EXPECTED_THREADS:
        median = statistics.median(measured[thread])
        reported = float(summary[thread]["T_eval_s"])
        assert abs(median - reported) <= 5e-11 * max(1.0, abs(median))
        speedup = baseline / median
        assert abs(speedup - float(summary[thread]["speedup"])) <= 5e-10

    print("PASS: 21 raw result records and manifests verified")
    print("PASS: all source and executable hashes agree across runs")
    print("PASS: all seven article medians and speedups were reproduced")


if __name__ == "__main__":
    main()
