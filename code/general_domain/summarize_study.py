#!/usr/bin/env python3
"""Summarize the general-domain pilot without changing any raw results.

Usage: python3 summarize_study.py --directory runs/pilot_20261108
Reference differences are diagnostics, not rigorous error bounds.
"""
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


HERE = Path(__file__).resolve().parent
CASES = ("sphere", "l_prism", "perforated_cube")
LABELS = {"sphere": "Sphere", "l_prism": "L prism", "perforated_cube": "Perforated cube"}
GROUPS = ("volume", "core")
CHECKS = (("mesh_L1_to_L2", "uniform_L1_p12", "uniform_L2_p12"),
          ("quadrature_p8_to_p12", "uniform_L2_p8", "uniform_L2_p12"))
TIGHT_REFERENCE = "uniform_L2_p16_d4"


def primary_runs(domain):
    selected = [r for r in domain["runs"] if r["name"] in
                ("adaptive_fine", "adaptive_medium", "adaptive_large")]
    return selected if selected else domain["runs"]


def read_json(path):
    return json.loads(path.read_text())


def read_values(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for row in rows:
        key = int(row["id"])
        if key in result:
            raise ValueError(f"Duplicate target id {key}: {path}")
        point = tuple(float(row[k]) for k in ("x", "y", "z"))
        value = float(row["potential"])
        if not all(math.isfinite(v) for v in (*point, value)):
            raise ValueError(f"Nonfinite value: {path}, id {key}")
        result[key] = (row["group"], point, value)
    if not result:
        raise ValueError(f"Empty values file: {path}")
    return result


def metrics(candidate, reference, group):
    if set(candidate) != set(reference):
        raise ValueError("Target IDs differ; comparison would be invalid")
    errors, refvalues = [], []
    for key, (refgroup, refpoint, refvalue) in reference.items():
        testgroup, testpoint, value = candidate[key]
        if testgroup != refgroup or any(abs(a-b) > 1e-12 for a, b in zip(testpoint, refpoint)):
            raise ValueError(f"Target identity/coordinates differ at id {key}")
        if refgroup == group:
            errors.append(value-refvalue)
            refvalues.append(refvalue)
    if not errors:
        return {"sample_count": 0, "relative_rms": math.nan,
                "absolute_rms": math.nan, "max_abs": math.nan}
    error_sq = math.fsum(v*v for v in errors)
    ref_sq = math.fsum(v*v for v in refvalues)
    return {"sample_count": len(errors),
            "relative_rms": math.sqrt(error_sq/ref_sq) if ref_sq > 0 else math.nan,
            "absolute_rms": math.sqrt(error_sq/len(errors)),
            "max_abs": max(map(abs, errors))}


def relative_difference(value, reference):
    return abs(value-reference)/abs(reference) if reference else math.nan


def number(value, digits=3):
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"{value:.{digits}e}"


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_study(directory, nominal):
    summary, checks, domains, warnings = [], [], [], []
    for case in CASES:
        folder = directory/case
        if not folder.exists():
            warnings.append(f"{case}: domain directory not present")
            continue
        selected_reference = (TIGHT_REFERENCE if (folder/(TIGHT_REFERENCE+".json")).exists()
                              else "uniform_L2_p12") if nominal == "auto" else nominal
        selected_checks = list(CHECKS)
        if selected_reference != "uniform_L2_p12":
            selected_checks.append(("final_reference_tightening", "uniform_L2_p12", selected_reference))
        final_tightened = selected_reference != "uniform_L2_p12"
        nominal_name = selected_reference
        refpath = folder/(nominal_name+".json")
        refvaluespath = folder/(nominal_name+".values.csv")
        if not refpath.exists() or not refvaluespath.exists():
            warnings.append(f"{case}: nominal reference {nominal_name} is missing")
            reference, reference_values = None, None
        else:
            reference, reference_values = read_json(refpath), read_values(refvaluespath)
        domain = {"case": case, "reference": reference, "reference_name": nominal_name,
                  "status": {g: "INCOMPLETE" for g in GROUPS}, "runs": []}
        for path in sorted(folder.glob("adaptive_*.json")):
            if path.name.endswith(".job.json"):
                continue
            stats = read_json(path)
            if stats.get("mode") != "solve":
                continue
            name = path.stem
            values = read_values(folder/(name+".values.csv"))
            directpath = folder/(name+".direct.csv")
            direct = read_values(directpath) if directpath.exists() else None
            if direct is None:
                warnings.append(f"{case}/{name}: same-mesh direct comparison is missing")
            jobpath = folder/(name+".job.json")
            job = read_json(jobpath) if jobpath.exists() else {}
            run = {"name": name, "stats": stats, "groups": {}}
            for group in GROUPS:
                row = {"case": case, "run": name, "target_group": group,
                       "N": int(stats["N"]), "root_count": int(stats["root_count"]),
                       "p": int(stats["p"]), "P": int(stats["P"]), "theta": stats["theta"],
                       "threads": int(stats["threads"]), "Q": stats["charge"],
                       "mesh_volume": stats["volume"], "tolerance": stats["tolerance"],
                       "passes": int(stats["passes"]), "quadrature_points": int(stats["quadrature_points"]),
                       "Tlist_s": stats["list_s"], "Tlist_kernel_s": stats["list_kernel_s"],
                       "Tprep_s": stats["preparation_s"], "Teval_s": stats["evaluation_s"],
                       "Ttotal_s": stats["total_solver_s"], "mesh_s": stats["mesh_s"],
                       "validation_target_evaluation_s": stats["validation_target_evaluation_s"],
                       "same_mesh_direct_s": stats.get("same_mesh_direct_s", math.nan),
                       "job_wall_s": job.get("wall_s", math.nan),
                       "nominal_reference": nominal_name, "reference_status": "INCOMPLETE"}
                for prefix, comparison in (("Eref", reference_values), ("Esame_direct", direct)):
                    m = metrics(values, comparison, group) if comparison else {
                        "sample_count": sum(v[0] == group for v in values.values()),
                        "relative_rms": math.nan, "absolute_rms": math.nan, "max_abs": math.nan}
                    row["sample_count"] = m.pop("sample_count")
                    row.update({prefix+"_"+key: value for key, value in m.items()})
                row["Q_reference"] = reference["charge"] if reference else math.nan
                row["Q_relative_difference"] = relative_difference(
                    stats["charge"], reference["charge"]) if reference else math.nan
                summary.append(row)
                run["groups"][group] = row
            domain["runs"].append(run)
        domain["runs"].sort(key=lambda r: r["stats"]["N"])
        if reference_values:
            for group in GROUPS:
                rows = [r for r in summary if r["case"] == case and r["target_group"] == group]
                min_rms = min((r["Eref_relative_rms"] for r in rows), default=math.nan)
                min_max = min((r["Eref_max_abs"] for r in rows), default=math.nan)
                statuses = []
                for check, other, check_base in selected_checks:
                    statpath, valuepath = folder/(other+".json"), folder/(other+".values.csv")
                    base_statpath, base_valuepath = folder/(check_base+".json"), folder/(check_base+".values.csv")
                    active_check = (not final_tightened or check == "final_reference_tightening")
                    row = {"case": case, "check": check, "target_group": group,
                           "candidate": other, "check_reference": check_base,
                           "nominal_reference": nominal_name, "used_for_final_status": active_check,
                           "minimum_adaptive_relative_rms": min_rms,
                           "minimum_adaptive_max_abs": min_max,
                           "rms_threshold_10pct": 0.1*min_rms,
                           "maxabs_threshold_10pct": 0.1*min_max}
                    if not all(p.exists() for p in (statpath, valuepath, base_statpath, base_valuepath)):
                        row["status"] = "INCOMPLETE"
                        warnings.append(f"{case}/{other}: reference check unavailable ({group})")
                    else:
                        stat = read_json(statpath)
                        base_stat, base_values = read_json(base_statpath), read_values(base_valuepath)
                        row.update(metrics(read_values(valuepath), base_values, group))
                        row.update(candidate_N=int(stat["N"]), check_reference_N=int(base_stat["N"]),
                                   candidate_Q=stat["charge"], check_reference_Q=base_stat["charge"],
                                   Q_relative_difference=relative_difference(stat["charge"], base_stat["charge"]))
                        if not math.isfinite(min_rms) or not math.isfinite(min_max):
                            row["status"] = "INCOMPLETE"
                        elif (row["relative_rms"] > 0.1*min_rms
                              or row["max_abs"] > 0.1*min_max):
                            row["status"] = "UNRESOLVED"
                        else:
                            row["status"] = "CHECK_PASSED"
                    if active_check:
                        statuses.append(row["status"])
                    checks.append(row)
                status = ("UNRESOLVED" if "UNRESOLVED" in statuses else
                          "INCOMPLETE" if "INCOMPLETE" in statuses else "CHECK_PASSED")
                domain["status"][group] = status
                for row in rows:
                    row["reference_status"] = status
        domains.append(domain)
    summary.sort(key=lambda row: (CASES.index(row["case"]), row["N"],
                                  GROUPS.index(row["target_group"])))
    return summary, checks, domains, list(dict.fromkeys(warnings))


def markdown_report(directory, domains, summary, checks, manifest, warnings):
    load = manifest.get("load_average_at_start", [])
    lines = ["# General-domain Hartree pilot", "",
        "**The direct reference is numerical, not exact. Reference checks below must be inspected before interpreting adaptive errors.**", "",
        "## Setup", "",
        "The positive model density is rho(x,y,z) = exp(-(x+2)^2 - 2(y+2)^2 - 3z^2), restricted to each mesh domain. It is not an SCF-generated molecular density.", "",
        "The prescribed radial monitor is eta_K = integral_K [0.02 + 1/sqrt(|x-(-2,-2,0)|^2 + 0.25)] dx, evaluated at quadrature accuracy 6 and recomputed on each adaptation pass. This is a mesh monitor, not a proven a posteriori estimator.", "",
        "Uniform references globally refine the same starting tetrahedral mesh. The nominal reference is the tightened L2,p=16,near-depth=4 run when available, otherwise L2,p=12,near-depth=3. L1,p=12 to L2,p=12 checks mesh sensitivity; L2,p=8 to L2,p=12 checks quadrature sensitivity. With the tightened reference available, the final status uses only the last p/depth tightening; earlier changes remain visible as historical checks, not error estimates for the improved reference. Direct evaluation uses no HNL, MAC, or multipoles, but shares the source/quadrature/singular-integration kernels with production. All mesh levels retain the same faceted boundary.", "",
        "Two target groups remain separate: volume-uniform samples across the domain and additional samples in the radius-0.9 source-core ball. Eref is the sampled error against the nominal uniform reference; Esame_direct is the sampled error against direct integration on the adaptive mesh with production quadrature. Relative RMS = sqrt(sum(error^2)/sum(reference^2)); max absolute error is reported separately. These are sampled diagnostics, not certified continuum L2 errors.", "",
        "**UNRESOLVED** means that the applicable reference change exceeds 10% of the smallest adaptive error in that group, measured in relative RMS or maximum absolute error. CHECK_PASSED means only that this pilot sensitivity check passed; it is not a rigorous reference-error bound or a separate proof of uniform-mesh convergence at the tightened order.", "",
        "## Runtime scope", "",
        "Tlist includes hierarchy/list preprocessing and its validation. Tprep is quadrature/source preparation plus moments. Teval evaluates one potential per active leaf. Ttotal = mesh processing + Tlist + Tprep + Teval. Same-mesh direct comparisons, common-target validation, reference construction, and file export are excluded. Reference runs evaluate only the sampled targets; their times cannot be used to claim a speedup for all-leaf production evaluation.", "",
        f"Recorded OpenMP threads: {manifest.get('threads', 'see rows')}. Startup system load averages: {load}. "
        + manifest.get("timing_note", "Timing provenance unavailable; treat timings as provisional."), "",
        "## Domains and reference status", "",
        "| Domain | Roots | Reference N | Volume | Reference Q | Volume status | Core status |",
        "|---|---:|---:|---:|---:|---|---|"]
    sphere_note = None
    for domain in domains:
        ref = domain["reference"]
        if ref:
            lines.append(f"| {LABELS[domain['case']]} | {int(ref['root_count'])} | {int(ref['N'])} | {ref['volume']:.6f} | {ref['charge']:.10g} | {domain['status']['volume']} | {domain['status']['core']} |")
            if domain["case"] == "sphere":
                deficit = 1-ref["volume"]/(4*math.pi*5**3/3)
                sphere_note = f"Sphere faceting volume deficit relative to the ideal radius-5 ball: {deficit:.3%}. Refinement here does not remove this geometry discrepancy."
    if sphere_note:
        lines.extend(["", sphere_note, ""])
    primary_names = {(d["case"], r["name"]) for d in domains for r in primary_runs(d)}
    lines.extend(["", "## Adaptive results", "",
        "Main rows show adaptive_fine, adaptive_medium, and adaptive_large when present. The initial adaptive_coarse pilot is retained in summary.csv but omitted here to avoid nearly duplicated sphere mesh sizes.", "",
        "| Domain/run | Group | N | p/P/theta | Threads | Q | Eref rel. RMS | Eref max abs. | Esame_direct rel. RMS | Charge rel. difference |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|"])
    for row in summary:
        if (row["case"], row["run"]) not in primary_names:
            continue
        lines.append(f"| {row['case']}/{row['run']} | {row['target_group']} | {row['N']} | {row['p']}/{row['P']}/{row['theta']:g} | {row['threads']} | {row['Q']:.10g} | {number(row['Eref_relative_rms'])} | {number(row['Eref_max_abs'])} | {number(row['Esame_direct_relative_rms'])} | {number(row['Q_relative_difference'])} |")
    lines.extend(["", "| Domain/run | N | Tlist (s) | Tprep (s) | Teval (s) | Ttotal (s) |",
                  "|---|---:|---:|---:|---:|---:|"])
    for row in summary:
        if row["target_group"] == "volume" and (row["case"], row["run"]) in primary_names:
            lines.append(f"| {row['case']}/{row['run']} | {row['N']} | {row['Tlist_s']:.5f} | {row['Tprep_s']:.5f} | {row['Teval_s']:.5f} | {row['Ttotal_s']:.5f} |")
    lines.extend(["", "## Reference checks", "",
        "| Domain | Check | Group | Relative RMS change | Max absolute change | Charge rel. change | Status | Used for final status |",
        "|---|---|---|---:|---:|---:|---|---|"])
    for row in checks:
        lines.append(f"| {row['case']} | {row['candidate']} -> {row['check_reference']} | {row['target_group']} | {number(row.get('relative_rms'))} | {number(row.get('max_abs'))} | {number(row.get('Q_relative_difference'))} | {row['status']} | {row['used_for_final_status']} |")
    lines.extend(["", "## Interpretation", "",
        "The same-mesh direct comparison checks acceleration error for the production discretization. The uniform-reference comparison additionally includes integration/mesh differences. Differences between domain cases should not be interpreted as isolated shape effects: root counts, volumes, adaptive leaf counts, source truncation, and shared-server load differ.", "",
        "At fixed p=6, P=4, theta=0.5, the source-core relative errors decrease across the three selected meshes in each domain. Volume-sampled errors are not strictly monotone in the sphere and L-prism cases; no monotone convergence rate is claimed. The same-mesh direct discrepancies are much smaller than the source-core discrepancies against the refined reference, indicating that source integration is the dominant remaining contribution in those core samples.", "",
        "Full precision results and reference stability thresholds are in summary.csv and reference_checks.csv. Raw commands, logs, values, source snapshots, and hashes are retained beside this report.", "",
        "Mesh comparison: [PDF](mesh_comparison.pdf), [PNG](mesh_comparison.png). The images show actual z=0 cuts of the adaptive meshes and the globally refined source-reference meshes."])
    if warnings:
        lines.extend(["", "## Missing data / warnings", ""] + ["- "+w for w in warnings])
    (directory/"report.md").write_text("\n".join(lines)+"\n")


def table(fig, bounds, headers, rows, size=9):
    ax = fig.add_axes(bounds)
    ax.set_axis_off()
    if not rows:
        ax.text(0.5, 0.5, "No completed data", ha="center", va="center")
        return
    tab = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center", bbox=[0, 0, 1, 1])
    tab.auto_set_font_size(False)
    tab.set_fontsize(size)
    for (row, _), cell in tab.get_celld().items():
        cell.set_edgecolor("#bac3ce")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor("#e8eef5")
            cell.set_text_props(weight="semibold")


def paragraph(fig, x, y, text, width=125, size=10):
    fig.text(x, y, "\n".join(textwrap.wrap(text, width)), va="top", fontsize=size, linespacing=1.4)


def new_page(title, page):
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.text(.05, .95, title, fontsize=18, weight="semibold", va="top")
    fig.text(.05, .025, "General-domain Hartree pilot | numerical references and shared-server timings", fontsize=8, color="#555")
    fig.text(.95, .025, str(page), fontsize=8, ha="right", color="#555")
    return fig


def pdf_report(directory, domains, checks, manifest, warnings):
    allruns = [(d, r) for d in domains for r in primary_runs(d)]
    with PdfPages(directory/"report.pdf") as pdf:
        fig = new_page("General-domain Hartree pilot: setup and reliability", 1)
        fig.text(.05, .89, "The direct reference is numerical, not exact.", fontsize=12, color="#a02020", weight="bold")
        paragraph(fig, .05, .85, "Three starting tetrahedral domains: radius-5 sphere (faceted), L-shaped prism, and cube with a centered cubic cavity. Each density is restricted to its mesh domain. No analytical potential is used.")
        fig.text(.05, .75, r"$\rho(x,y,z)=\exp[-(x+2)^2-2(y+2)^2-3z^2]$", fontsize=13)
        fig.text(.05, .69, r"$\eta_K=\int_K[0.02+(|\mathbf{x}-(-2,-2,0)|^2+0.25)^{-1/2}]\,d\mathbf{x}$", fontsize=12)
        paragraph(fig, .05, .65, "The prescribed radial monitor is recomputed with order-6 quadrature at each adaptation pass. Direct references use uniform global refinements of the same initial mesh, with local near-singular correction. They bypass HNL/MAC and multipoles but share quadrature and singular-integration kernels.")
        rows = []
        for d in domains:
            ref = d["reference"]
            rows.append([LABELS[d["case"]], str(int(ref["root_count"])) if ref else "n/a",
                         str(int(ref["N"])) if ref else "n/a", f"{ref['volume']:.3f}" if ref else "n/a",
                         d["status"]["volume"], d["status"]["core"]])
        table(fig, [.05, .39, .90, .15], ["Domain", "Roots", "Reference N", "Volume", "Volume check", "Core check"], rows, 9)
        paragraph(fig, .05, .35, "References: use L2,p=16,near-depth=4 if available, otherwise L2,p=12,depth=3. Final status uses the last available tightening. UNRESOLVED if its change exceeds 10% of the smallest adaptive error in that group (relative RMS or max absolute). Earlier refinement checks remain visible, not as error estimates for the improved reference. CHECK_PASSED is not a rigorous bound.")
        load = manifest.get("load_average_at_start", [])
        paragraph(fig, .05, .22, f"Timing provenance: {manifest.get('threads', 'unrecorded')} OpenMP threads; recorded initial load averages {load}. Runs were made on a shared server, not an isolated performance benchmark. Reference timings use far fewer targets and are not used to claim production speedups.")
        paragraph(fig, .05, .10, f"Data warnings: {len(warnings)}. See report.md for details. The sphere remains the same faceted domain at every refinement level.", size=9)
        pdf.savefig(fig); plt.close(fig)

        fig = new_page("Adaptive results: errors, source charge, and solver time", 2)
        paragraph(fig, .05, .88, "Eref: relative sampled RMS versus the nominal uniform reference. Edirect: relative sampled RMS versus direct evaluation on the same adaptive mesh and quadrature. Volume and source-core samples are never pooled.")
        rows = []
        for d, run in allruns:
            s, v, c = run["stats"], run["groups"]["volume"], run["groups"]["core"]
            rows.append([LABELS[d["case"]]+"/"+run["name"].replace("adaptive_", ""),
                         int(s["N"]), f"{s['charge']:.7f}", number(v["Eref_relative_rms"]),
                         number(c["Eref_relative_rms"]), number(v["Esame_direct_relative_rms"]),
                         number(c["Esame_direct_relative_rms"])])
        table(fig, [.04, .50, .92, .25], ["Domain/run", "N", "Charge Q", "Eref volume", "Eref core", "Edirect volume", "Edirect core"], rows, 8.5)
        rows = []
        for d, run in allruns:
            s = run["stats"]
            rows.append([LABELS[d["case"]]+"/"+run["name"].replace("adaptive_", ""),
                         f"{int(s['p'])}/{int(s['P'])}/{s['theta']:g}", int(s["threads"]),
                         *[f"{s[key]:.4f}" for key in ("list_s", "preparation_s", "evaluation_s", "total_solver_s")]])
        table(fig, [.04, .20, .92, .24], ["Domain/run", "p/P/theta", "Threads", "Tlist (s)", "Tprep (s)", "Teval (s)", "Ttotal (s)"], rows, 8.5)
        paragraph(fig, .05, .14, "Tlist: hierarchy and HNL/MAC preparation including validation. Tprep: quadrature plus moments. Teval: one potential per active leaf. Ttotal also includes mesh processing; it excludes same-mesh direct checks, sampled-target validation, references, and exports. Full-precision errors, max errors, and charge differences are in the CSV files.", size=9)
        pdf.savefig(fig); plt.close(fig)

        fig = new_page("Error trends and reference checks", 3)
        ax1, ax2 = fig.add_axes([.08,.55,.37,.28]), fig.add_axes([.57,.55,.37,.28])
        colors = ("#2369ad", "#df7924", "#23936a")
        for d, color in zip(domains, colors):
            runs = primary_runs(d)
            if not runs:
                continue
            ns = [r["stats"]["N"] for r in runs]
            for group, style in (("volume", "-o"), ("core", "--s")):
                es = [r["groups"][group]["Eref_relative_rms"] for r in runs]
                ax1.plot(ns, es, style, color=color, label=LABELS[d["case"]]+" "+group, markersize=4)
            ax2.plot(ns, [r["stats"]["total_solver_s"] for r in runs], "-o", color=color, label=LABELS[d["case"]], markersize=4)
        ax1.set_yscale("log")
        ax1.set_xlabel("Adaptive active elements N"); ax1.set_ylabel("Relative sampled RMS error")
        ax2.set_xlabel("Adaptive active elements N"); ax2.set_ylabel("Total solver time (s)")
        for ax in (ax1, ax2):
            ax.grid(alpha=.2); ax.tick_params(labelsize=8)
            if ax.lines:
                if ax is ax1:
                    ax.legend(fontsize=7, frameon=False, loc="lower center",
                              bbox_to_anchor=(.5,1.02), ncol=2)
                else:
                    ax.legend(fontsize=7, frameon=False)
        names = {"mesh_L1_to_L2": "L1 to L2", "quadrature_p8_to_p12": "p8 to p12",
                 "final_reference_tightening": "p12 to p16/d4"}
        rows = [[LABELS[r["case"]], names[r["check"]]+("*" if r["used_for_final_status"] else ""), r["target_group"],
                 number(r.get("relative_rms")), number(r.get("max_abs")), number(r.get("Q_relative_difference")), r["status"]]
                for r in checks]
        table(fig, [.04,.10,.92,.36], ["Domain", "Check (*final)", "Group", "Rel. RMS change", "Max abs. change", "Charge rel. change", "Status"], rows, 7.2 if len(rows)>12 else 7.8)
        paragraph(fig, .05, .075, "* Checks used in final status. Lines connect pilot measurements only, not a proven rate or complexity. Geometry, root count, and source truncation differ between cases.", size=8, width=145)
        pdf.savefig(fig); plt.close(fig)


def compact_results(directory, domains):
    lines = [
        "# CPU pilot; rho=exp(-(x+2)^2-2*(y+2)^2-3*z^2).",
        "# E_volume and E_core are separate sampled relative RMS errors against the uniform p16/depth4 direct reference.",
        "# Timings in seconds; shared-server pilot, not an isolated benchmark.",
        "# T_total includes mesh processing; reference/validation/I-O are excluded.",
        f"{'domain':18s} {'N':>8s} {'p':>3s} {'P':>3s} {'theta':>6s} {'threads':>7s} {'N_ref':>8s} {'E_volume':>12s} {'E_core':>12s} {'T_list':>10s} {'T_prep':>10s} {'T_eval':>10s} {'T_total':>10s}",
    ]
    for d in domains:
        for run in primary_runs(d):
            s=run["stats"]
            lines.append(
                f"{d['case']:18s} {int(s['N']):8d} {int(s['p']):3d} {int(s['P']):3d} {s['theta']:6.2f} {int(s['threads']):7d} "
                f"{int(d['reference']['N']):8d} {run['groups']['volume']['Eref_relative_rms']:12.5e} {run['groups']['core']['Eref_relative_rms']:12.5e} "
                f"{s['list_s']:10.5f} {s['preparation_s']:10.5f} {s['evaluation_s']:10.5f} {s['total_solver_s']:10.5f}")
    (directory/"results_general_domain.txt").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=HERE/"runs"/"pilot_20261108")
    parser.add_argument("--reference", default="auto", help="auto prefers uniform_L2_p16_d4 when available")
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not directory.is_dir():
        parser.error(f"Study directory does not exist: {directory}")
    manifestpath = directory/"manifest.json"
    manifest = read_json(manifestpath) if manifestpath.exists() else {}
    summary, checks, domains, warnings = load_study(directory, args.reference)
    write_csv(directory/"summary.csv", summary)
    write_csv(directory/"reference_checks.csv", checks)
    markdown_report(directory, domains, summary, checks, manifest, warnings)
    pdf_report(directory, domains, checks, manifest, warnings)
    compact_results(directory, domains)
    print(f"Wrote summary.csv, reference_checks.csv, report.md and report.pdf in {directory}")
    for domain in domains:
        print(domain["case"]+": "+", ".join(g+"="+domain["status"][g] for g in GROUPS))
    for warning in warnings:
        print("WARNING: "+warning)


if __name__ == "__main__":
    main()
