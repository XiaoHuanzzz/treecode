#!/usr/bin/env python3
"""Build the consolidated general-domain report from verified study outputs."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "treecode_gd_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


HERE = Path(__file__).resolve().parent
CASES = ("sphere", "l_prism", "perforated_cube")
LABELS = {"sphere": "Sphere", "l_prism": "L-shaped prism",
          "perforated_cube": "Perforated cube"}


def read_rows(path: Path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 12:
        raise ValueError(f"Expected 12 accuracy rows, found {len(rows)}")
    return rows


def select(rows, case, run, group):
    matches = [row for row in rows if row["case"] == case and
               row["run"] == run and row["target_group"] == group]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {case}/{run}/{group}")
    return matches[0]


def check_inputs(rows, accuracy_dir):
    for case in CASES:
        for run in ("radial_baseline", "localized"):
            job = json.loads((accuracy_dir / case / f"{run}.job.json").read_text())
            if job.get("returncode") != 0 or not job.get("mesh_hash_match"):
                raise ValueError(f"Unverified mesh/result: {case}/{run}")
            if not job.get("element_count_match") or not job.get("thread_count_match"):
                raise ValueError(f"Count/thread check failed: {case}/{run}")
        for group in ("volume", "core"):
            for run in ("radial_baseline", "localized"):
                row = select(rows, case, run, group)
                if row["reference_guard"] != "CHECK_PASSED":
                    raise ValueError(f"Reference guard failed: {case}/{run}/{group}")


def cover_pdf(path, rows):
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(.06, .92, "General-domain Hartree treecode study",
                 fontsize=25, weight="bold")
        fig.text(.06, .865,
                 "Adaptive meshes, direct references, localized grading, and radial controls",
                 fontsize=14, color="#36454f")
        setup = (
            r"Source: $\rho=\exp[-(x+2)^2-2(y+2)^2-3z^2]$, restricted to each domain.  "
            r"Treecode settings: $p=6$, $P=4$, $\theta=0.5$; 16 OpenMP threads."
            "\n"
            "Errors use the same 128 volume-uniform targets and 32 source-core targets. "
            "They are sampled relative RMS diagnostics against the tightened uniform direct reference, "
            "not certified continuum $L^2$ errors."
        )
        fig.text(.06, .78, setup, fontsize=11.2, linespacing=1.4, va="top", wrap=True)

        table_rows = []
        for case in CASES:
            radial_v = select(rows, case, "radial_baseline", "volume")
            local_v = select(rows, case, "localized", "volume")
            local_c = select(rows, case, "localized", "core")
            table_rows.append([
                LABELS[case], f"{int(radial_v['N']):,}", f"{int(local_v['N']):,}",
                f"{float(local_v['Eref_relative_rms']):.3e}",
                f"{float(local_c['Eref_relative_rms']):.3e}",
                f"{float(local_v['localized_over_radial_Ttotal']):.3f}",
            ])
        axis = fig.add_axes([.06, .48, .88, .21])
        axis.axis("off")
        table = axis.table(
            cellText=table_rows,
            colLabels=["Domain", r"$N$ radial", r"$N$ localized",
                       "Volume RMS", "Core RMS",
                       r"$T_{\rm total}$ ratio"],
            cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10.5)
        for (row, col), cell in table.get_celld().items():
            cell.set_linewidth(.55)
            cell.set_edgecolor("#aab7c4")
            if row == 0:
                cell.set_facecolor("#dfe8f1")
                cell.set_text_props(weight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#f5f7f9")

        findings = (
            "Main observations\n"
            "• HNL–MAC source partitions were complete and disjoint on the faceted sphere, "
            "nonconvex L-shaped prism, and perforated cube.\n"
            "• The localized meshes retained only 10.2–23.8% of the radial leaf counts. "
            "This reduced the observed total times, but the production workloads also became smaller.\n"
            "• Accuracy was not uniformly improved: the sphere worsened in both target groups; "
            "the L-shaped case changed little; the perforated cube improved in the core but worsened "
            "over the volume targets.\n"
            "• The results support applicability to non-cuboidal locally refined meshes. They do not "
            "establish an optimal indicator, equal-cost superiority, a convergence rate, or a controlled speedup."
        )
        fig.text(.06, .395, findings, fontsize=11.2, va="top", linespacing=1.45, wrap=True)
        fig.text(.06, .075,
                 "Reference qualification: the final p/depth sensitivity guards passed for all cases. "
                 "Direct and treecode evaluations share quadrature and singular-integration kernels; "
                 "the sphere boundary remains faceted. Timings are fresh single runs on a shared server.",
                 fontsize=9.5, color="#4b5563", va="bottom", wrap=True)
        fig.text(.94, .035, "Updated 2026-11-08", ha="right", fontsize=9, color="#555")
        pdf.savefig(fig)
        plt.close(fig)


def markdown_summary(path, rows, output_pdf):
    lines = [
        "# General-domain Hartree treecode study", "",
        "This consolidated report combines the original general-domain pilot, actual mesh sections, "
        "the concentrated adaptive meshes and their potential-accuracy evaluation, and a separate "
        "study using a radial Gaussian source with a radial target-size indicator.", "",
        "All treecode runs use `p=6`, `P=4`, `theta=0.5`, and 16 OpenMP threads. "
        "The errors below are separate sampled relative RMS diagnostics at 128 volume targets and "
        "32 source-core targets; they are not certified continuum L2 errors.", "",
        "| Domain | Mesh | N | Volume error | Core error | Ttotal (s) |", "|---|---|---:|---:|---:|---:|"
    ]
    for case in CASES:
        for run, label in (("radial_baseline", "Radial"), ("localized", "Localized")):
            volume = select(rows, case, run, "volume")
            core = select(rows, case, run, "core")
            lines.append(
                f"| {LABELS[case]} | {label} | {int(volume['N']):,} | "
                f"{float(volume['Eref_relative_rms']):.4e} | "
                f"{float(core['Eref_relative_rms']):.4e} | "
                f"{float(volume['Ttotal_s']):.4f} |")
    lines += ["", "## Interpretation", "",
        "The HNL–MAC method operated with validated source coverage on all three domains. "
        "The localized meshes reduce the element count substantially, but they do not improve every error measure. "
        "They support general-domain applicability rather than an optimal-mesh or equal-cost superiority claim.", "",
        "The numerical reference is direct integration on the same starter geometry after two global refinements, "
        "using p=16 and near-depth 4. The final p/depth tightening guards passed, but direct and treecode "
        "calculations share quadrature and singular-integration kernels. The sphere is a fixed faceted domain.", "",
        "## Report components", "",
        f"- Consolidated PDF: [{output_pdf.name}]({output_pdf.name})",
        "- Original pilot details: `pilot_20261108/report.md`",
        "- Localized-mesh accuracy: `localized_accuracy_20261108/report.md`",
        "- Full localized results: `localized_accuracy_20261108/summary.csv`",
        "- Radial source and radial indicator: `radial_domain_20261108/report.md`",
        "- Full radial results: `radial_domain_20261108/summary.csv`",
        "- Reproduction instructions: `../../README.md`", ""]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=HERE / "runs")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runs = args.runs.resolve()
    output = (args.output.resolve() if args.output else
              runs / "general_domain_report_20261108.pdf")
    pilot = runs / "pilot_20261108"
    meshes = runs / "localized_meshes_20261108"
    accuracy = runs / "localized_accuracy_20261108"
    radial = runs / "radial_domain_20261108"
    inputs = [pilot / "report.pdf", pilot / "mesh_comparison.pdf",
              meshes / "mesh_before_after.pdf", accuracy / "report.pdf",
              radial / "mesh_comparison.pdf", radial / "report.pdf"]
    for path in inputs:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    rows = read_rows(accuracy / "summary.csv")
    check_inputs(rows, accuracy)
    for case in CASES:
        job = json.loads((radial / case / "adaptive_radial_graded.job.json").read_text())
        if (job.get("returncode") != 0 or not job.get("source_verified") or
                not job.get("monitor_verified")):
            raise ValueError(f"Unverified radial study: {case}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="general-domain-report-", dir="/tmp") as temporary:
        cover = Path(temporary) / "cover.pdf"
        merged = Path(temporary) / "report.pdf"
        cover_pdf(cover, rows)
        pdfunite = shutil.which("pdfunite")
        if not pdfunite:
            raise RuntimeError("pdfunite is required to assemble the report")
        subprocess.run([pdfunite, str(cover), *(str(path) for path in inputs), str(merged)], check=True)
        shutil.copyfile(merged, output)
    markdown_summary(runs / "general_domain_report_20261108.md", rows, output)
    print(f"Wrote {output}")
    print(f"Wrote {runs / 'general_domain_report_20261108.md'}")


if __name__ == "__main__":
    main()
