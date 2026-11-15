#!/usr/bin/env python3
"""Summarize the completed radial-source, radial-indicator study."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import tempfile
import textwrap

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "treecode_gd_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from plot_mesh_comparison import read_dx
from summarize_study import read_values, metrics

HERE = Path(__file__).resolve().parent
CASES = ("sphere", "l_prism", "perforated_cube")
LABELS = {"sphere": "Sphere", "l_prism": "L-shaped prism",
          "perforated_cube": "Perforated cube"}
GROUPS = ("volume", "core")


def mesh_contrast(path):
    points, cells = read_dx(path)
    tets = points[cells]
    edge = tets[:, 1:] - tets[:, :1]
    size = np.cbrt(np.abs(np.linalg.det(edge))/6.0)
    radius = np.linalg.norm(tets.mean(axis=1)-np.array([-2., -2., 0.]), axis=1)
    near, far = size[radius < 1.0], size[radius > 3.0]
    if not len(near) or not len(far):
        raise ValueError(f"Insufficient core/far cells in {path}")
    return float(np.median(near)), float(np.median(far)), int(len(cells))


def load(directory):
    manifest = json.loads((directory/"manifest.json").read_text())
    rows = []
    domains = []
    for case in CASES:
        folder = directory/case
        stats = json.loads((folder/"adaptive_radial_graded.json").read_text())
        reference_stats = json.loads((folder/"uniform_L2_p16_d4.json").read_text())
        values = read_values(folder/"adaptive_radial_graded.values.csv")
        direct = read_values(folder/"adaptive_radial_graded.direct.csv")
        reference = read_values(folder/"uniform_L2_p16_d4.values.csv")
        p12 = read_values(folder/"uniform_L2_p12_d3.values.csv")
        job = json.loads((folder/"adaptive_radial_graded.job.json").read_text())
        log = (folder/"adaptive_radial_graded.log").read_text(errors="replace")
        if (job.get("returncode") != 0 or not job.get("source_verified") or
                not job.get("monitor_verified") or "Validated " not in log):
            raise ValueError(f"Run validation missing for {case}")
        core_size, far_size, exported = mesh_contrast(folder/"adaptive_radial_graded.dx")
        domain = dict(case=case, stats=stats, reference=reference_stats,
                      core_size=core_size, far_size=far_size,
                      size_ratio=far_size/core_size, exported_tetrahedra=exported,
                      groups={})
        for group in GROUPS:
            error = metrics(values, reference, group)
            acceleration = metrics(values, direct, group)
            sensitivity = metrics(p12, reference, group)
            row = dict(case=case, target_group=group,
                       sample_count=error["sample_count"], N=int(stats["N"]),
                       root_count=int(stats["root_count"]), p=int(stats["p"]),
                       P=int(stats["P"]), theta=stats["theta"],
                       threads=int(stats["threads"]), charge=stats["charge"],
                       reference_N=int(reference_stats["N"]),
                       reference_charge=reference_stats["charge"],
                       Eref_relative_rms=error["relative_rms"],
                       Eref_absolute_rms=error["absolute_rms"],
                       Eref_max_abs=error["max_abs"],
                       Esame_direct_relative_rms=acceleration["relative_rms"],
                       Esame_direct_max_abs=acceleration["max_abs"],
                       reference_change_relative_rms=sensitivity["relative_rms"],
                       reference_change_max_abs=sensitivity["max_abs"],
                       reference_change_over_error_rms=sensitivity["relative_rms"]/error["relative_rms"],
                       reference_change_over_error_max=sensitivity["max_abs"]/error["max_abs"],
                       reference_guard=(sensitivity["relative_rms"] <= .1*error["relative_rms"] and
                                        sensitivity["max_abs"] <= .1*error["max_abs"]),
                       core_median_size=core_size, far_median_size=far_size,
                       far_core_size_ratio=far_size/core_size,
                       exported_tetrahedra=exported,
                       Tlist_s=stats["list_s"], Tprep_s=stats["preparation_s"],
                       Teval_s=stats["evaluation_s"], Ttotal_s=stats["total_solver_s"])
            rows.append(row)
            domain["groups"][group] = row
        domains.append(domain)
    return manifest, domains, rows


def write_csv(path, rows):
    fields = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def write_markdown(path, manifest, domains):
    lines = ["# Radial source and radial indicator on general domains", "",
        r"The source is $\rho(\mathbf x)=\exp[-4|\mathbf x-(-2,-2,0)|^2]$, restricted to each discrete domain.", "",
        r"The radial target-size profile is $h_{\rm rel}(r)=1+(\gamma-1)r^2/(r^2+R^2)$ and the element indicator is $\eta_K=\int_K h_{\rm rel}(r)^{-3}\,d\mathbf x$, with $R=1.5$, $\gamma=4$, and tolerance $2\times10^{-4}$. It is a prescribed monitor, not an a posteriori error estimator.", "",
        "All treecode solves use p=6, P=4, theta=0.5, four adaptation passes, and 16 OpenMP threads. The numerical reference uses two global refinements, p=16, and near-depth 4. The same 128 volume targets and 32 source-core targets are reported separately.", "",
        "| Domain | N | Far/core size | Volume rel. RMS | Core rel. RMS | Tlist (s) | Teval (s) | Ttotal (s) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for d in domains:
        v, c = d["groups"]["volume"], d["groups"]["core"]
        lines.append(f"| {LABELS[d['case']]} | {v['N']:,} | {d['size_ratio']:.2f} | "
                     f"{v['Eref_relative_rms']:.4e} | {c['Eref_relative_rms']:.4e} | "
                     f"{v['Tlist_s']:.4f} | {v['Teval_s']:.4f} | {v['Ttotal_s']:.4f} |")
    lines += ["", "The far/core size ratio uses the medians of $|K|^{1/3}$ for exported tetrahedra whose centroids satisfy $r>3$ and $r<1$. It describes radial grading; it is not a mesh-quality metric.", "",
              "## Validation", "",
              "| Domain | Group | Max error | Same-mesh direct RMS | Reference-change/error (RMS) | Reference-change/error (max) |", "|---|---|---:|---:|---:|---:|"]
    for d in domains:
        for group in GROUPS:
            r=d["groups"][group]
            lines.append(f"| {LABELS[d['case']]} | {group} | {r['Eref_max_abs']:.4e} | "
                         f"{r['Esame_direct_relative_rms']:.4e} | "
                         f"{100*r['reference_change_over_error_rms']:.2f}% | "
                         f"{100*r['reference_change_over_error_max']:.2f}% |")
    guards = all(r["reference_guard"] for d in domains for r in d["groups"].values())
    lines += ["", f"All p12/depth3 to p16/depth4 sensitivity guards: **{'PASS' if guards else 'NOT PASSED'}**.", "",
              "Errors are sampled diagnostics, not certified continuum L2 errors. Direct and treecode calculations share source quadrature and singular-integration kernels. The sphere remains faceted, and timings are single runs on a shared server.", ""]
    path.write_text("\n".join(lines))


def add_table(axis, cell_text, labels, widths=None, fontsize=9.5):
    axis.axis("off")
    table=axis.table(cellText=cell_text,colLabels=labels,cellLoc="center",loc="center",bbox=[0,0,1,1],colWidths=widths)
    table.auto_set_font_size(False); table.set_fontsize(fontsize)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor("#aab7c4"); cell.set_linewidth(.55)
        if row==0: cell.set_facecolor("#dfe8f1"); cell.set_text_props(weight="bold")
        elif row%2==0: cell.set_facecolor("#f5f7f9")


def write_pdf(path, manifest, domains):
    plt.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42})
    with PdfPages(path) as pdf:
        fig=plt.figure(figsize=(11.7,8.3)); fig.text(.06,.93,"Radial general-domain study",fontsize=24,weight="bold")
        fig.text(.06,.87,r"$\rho=e^{-4r^2}$ and a smooth radial target-size indicator",fontsize=15,color="#36454f")
        setup=(r"Center $(-2,-2,0)$; $R=1.5$, $\gamma=4$, tolerance $2\times10^{-4}$; "
               r"$p=6$, $P=4$, $\theta=0.5$; 16 OpenMP threads. The monitor is prescribed, not an error estimator.")
        fig.text(.06,.80,textwrap.fill(setup,125),fontsize=11.5,va="top")
        rows=[]
        for d in domains:
            v,c=d["groups"]["volume"],d["groups"]["core"]
            rows.append([LABELS[d["case"]],f"{v['N']:,}",f"{d['size_ratio']:.2f}",
                         f"{v['Eref_relative_rms']:.3e}",f"{c['Eref_relative_rms']:.3e}",f"{v['Ttotal_s']:.3f}"])
        axis=fig.add_axes([.06,.48,.88,.24]); add_table(axis,rows,["Domain","N","Far/core size","Volume RMS","Core RMS","Ttotal (s)"],fontsize=10.5)
        note=("The radial grading is evident in the median cell-size ratios. HNL–MAC coverage validation completed for every target. "
              "Accuracy is measured against the tightened uniform direct reference on common targets. These are sampled diagnostics; "
              "the direct reference is numerical and shares integration kernels with production.")
        fig.text(.06,.39,textwrap.fill(note,135),fontsize=11.2,va="top",linespacing=1.4)
        fig.text(.06,.16,"Interpretation",fontsize=13,weight="bold")
        fig.text(.06,.12,"The test isolates a radially symmetric source and radial mesh-size field while retaining non-cuboidal boundaries.\nIt demonstrates general-domain operation; it does not establish an optimal indicator or domain-independent convergence rate.",fontsize=11.2,va="top",linespacing=1.45)
        pdf.savefig(fig); plt.close(fig)

        fig=plt.figure(figsize=(11.7,8.3)); fig.text(.06,.93,"Reference and acceleration checks",fontsize=22,weight="bold")
        rows=[]
        for d in domains:
            for group in GROUPS:
                r=d["groups"][group]
                rows.append([LABELS[d["case"]],group,f"{r['Eref_max_abs']:.3e}",
                             f"{r['Esame_direct_relative_rms']:.3e}",
                             f"{100*r['reference_change_over_error_rms']:.2f}%",
                             f"{100*r['reference_change_over_error_max']:.2f}%",
                             "PASS" if r["reference_guard"] else "FAIL"])
        axis=fig.add_axes([.05,.40,.90,.42]); add_table(axis,rows,["Domain","Targets","Max error","Tree/direct RMS","Ref./error RMS","Ref./error max","Guard"],fontsize=9.5)
        text=("Tree/direct RMS compares the accelerated result with direct evaluation on the same adaptive mesh and production quadrature. "
              "Ref./error compares the p12/depth3-to-p16/depth4 reference change with the reported error; both RMS and maximum must be below 10%. "
              "PASS is a sensitivity check, not a rigorous error bound. The 128 volume and 32 core targets remain separate.")
        fig.text(.06,.31,textwrap.fill(text,140),fontsize=11,va="top",linespacing=1.4)
        fig.text(.06,.10,"Caveats: fixed faceted sphere geometry; shared reference kernels; no continuous-norm certification; single-run shared-server timings.",fontsize=10.5)
        pdf.savefig(fig); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,default=HERE/"runs"/"radial_domain_20261108")
    args=parser.parse_args(); directory=args.directory.resolve()
    manifest,domains,rows=load(directory)
    write_csv(directory/"summary.csv",rows)
    write_markdown(directory/"report.md",manifest,domains)
    write_pdf(directory/"report.pdf",manifest,domains)
    compact=["# Radial source + radial indicator; sampled relative RMS errors.",
             "domain N far_core_size E_volume E_core Tlist_s Teval_s Ttotal_s"]
    for d in domains:
        v,c=d["groups"]["volume"],d["groups"]["core"]
        compact.append(f"{d['case']} {v['N']} {d['size_ratio']:.8g} {v['Eref_relative_rms']:.8g} "
                       f"{c['Eref_relative_rms']:.8g} {v['Tlist_s']:.8g} {v['Teval_s']:.8g} {v['Ttotal_s']:.8g}")
    (directory/"results_radial_domain.txt").write_text("\n".join(compact)+"\n")
    print("Wrote report.md, report.pdf, summary.csv, and results_radial_domain.txt")


if __name__=="__main__": main()
