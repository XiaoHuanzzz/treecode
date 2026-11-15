#!/usr/bin/env python3
"""Report measured localized-mesh accuracy; never generate or substitute data.

Run after run_localized_accuracy.py. All three domains and both fresh solves
must be complete. The saved pilot supplies the historical radial solve and
the numerical uniform-mesh references, not a replacement for fresh timings.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import textwrap

from summarize_study import (CASES, GROUPS, HERE, LABELS, metrics, number,
                             plt, PdfPages, read_values, table, write_csv)


REFERENCE = "uniform_L2_p16_d4"
RUNS = ("radial_baseline", "localized")
RUN_LABELS = {"radial_baseline": "Radial", "localized": "Localized"}
TIMES = ("list_s", "preparation_s", "evaluation_s", "total_solver_s")


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ratio(numerator, denominator):
    if denominator == 0:
        return 1.0 if numerator == 0 else math.inf
    return numerator / denominator


def factor(value):
    if not math.isfinite(value):
        return "inf"
    return f"{value:.6f}" if abs(value - 1) < .001 else f"{value:.3f}"


def command_option(job, flag):
    command = job.get("command", [])
    return command[command.index(flag) + 1] if flag in command else None


def validate_targets(values, context):
    counts = {group: sum(v[0] == group for v in values.values()) for group in GROUPS}
    if counts != {"volume": 128, "core": 32} or len(values) != 160:
        raise ValueError(f"{context}: expected exactly 128 volume and 32 core targets, got {counts}")


def geometry_check(folder, reference, reference_job, runs):
    """Verify input bytes, not merely equal volumes or similar pictures."""
    base_hash = digest(folder / "base.mesh")
    if reference_job.get("mesh_sha256") != base_hash:
        raise ValueError(f"{folder.name}: reference hash does not match saved base.mesh")
    for run in runs.values():
        stats, job = run["stats"], run["job"]
        mesh_path = command_option(job, "--mesh")
        if mesh_path is None:
            raise ValueError(f"{folder.name}/{run['name']}: input mesh absent from command")
        mesh_path = Path(mesh_path)
        if not mesh_path.is_absolute():
            mesh_path = HERE / mesh_path
        if digest(mesh_path) != base_hash:
            raise ValueError(f"{folder.name}/{run['name']}: starting geometry differs from reference")
        if job.get("mesh_sha256", base_hash) != base_hash:
            raise ValueError(f"{folder.name}/{run['name']}: recorded mesh hash differs")
        if int(stats["root_count"]) != int(reference["root_count"]):
            raise ValueError(f"{folder.name}/{run['name']}: root count differs from reference")
        if not math.isclose(stats["volume"], reference["volume"], rel_tol=1e-9, abs_tol=1e-10):
            raise ValueError(f"{folder.name}/{run['name']}: domain volume differs from reference")
        if stats["source"] != reference["source"]:
            raise ValueError(f"{folder.name}/{run['name']}: source definition differs from reference")
    return base_hash


def load_study(directory, pilot):
    manifest = read_json(directory / "manifest.json")
    domains, summary = [], []
    for case in CASES:
        folder, oldfolder = directory / case, pilot / case
        reference = read_json(oldfolder / f"{REFERENCE}.json")
        reference_job = read_json(oldfolder / f"{REFERENCE}.job.json")
        if (reference.get("mode") != "reference" or int(reference["level"]) != 2
                or int(reference["p"]) != 16 or int(reference["near_depth"]) != 4):
            raise ValueError(f"{case}: unexpected uniform reference parameters")
        refvalues = read_values(oldfolder / f"{REFERENCE}.values.csv")
        reference_hash = digest(oldfolder / f"{REFERENCE}.values.csv")
        targets_hash = digest(oldfolder / "targets.csv")
        if reference_job.get("target_sha256") != targets_hash:
            raise ValueError(f"{case}: reference target-file hash differs from saved targets")
        p12values = read_values(oldfolder / "uniform_L2_p12.values.csv")
        oldstats = read_json(oldfolder / "adaptive_large.json")
        oldvalues = read_values(oldfolder / "adaptive_large.values.csv")
        validate_targets(refvalues, f"{case}/reference")
        runs = {}
        for name in RUNS:
            stats = read_json(folder / f"{name}.json")
            job = read_json(folder / f"{name}.job.json")
            if job.get("returncode") != 0 or job.get("mesh_hash_match") is not True:
                raise ValueError(f"{case}/{name}: successful job and mesh_hash_match=true required")
            if job.get("reference_sha256") != reference_hash or job.get("targets_sha256") != targets_hash:
                raise ValueError(f"{case}/{name}: reference/target bytes differ from recorded provenance")
            if digest(folder / f"{name}.dx") != job.get("expected_mesh_sha256"):
                raise ValueError(f"{case}/{name}: actual DX mesh does not match recorded expected mesh")
            if (stats.get("mode") != "solve" or int(stats["p"]) != 6 or int(stats["P"]) != 4
                    or stats["theta"] != .5 or int(stats["threads"]) != 16):
                raise ValueError(f"{case}/{name}: expected actual p=6, P=4, theta=0.5, threads=16 solve")
            values = read_values(folder / f"{name}.values.csv")
            direct = read_values(folder / f"{name}.direct.csv")
            validate_targets(values, f"{case}/{name}")
            runs[name] = {"name": name, "stats": stats, "job": job, "values": values,
                          "direct": direct, "groups": {}}
        mesh_hash = geometry_check(oldfolder, reference, reference_job, runs)
        checks, repeat = {}, {}
        for group in GROUPS:
            sensitivity = metrics(p12values, refvalues, group)
            historical = metrics(oldvalues, refvalues, group)
            reproducibility = metrics(runs["radial_baseline"]["values"], oldvalues, group)
            repeat[group] = reproducibility
            checks[group] = {"sensitivity": sensitivity, "historical": historical}
            for name, run in runs.items():
                s = run["stats"]
                err = metrics(run["values"], refvalues, group)
                direct = metrics(run["values"], run["direct"], group)
                row = dict(case=case, run=name, target_group=group,
                           sample_count=err["sample_count"], N=int(s["N"]),
                           root_count=int(s["root_count"]), p=int(s["p"]), P=int(s["P"]),
                           theta=s["theta"], threads=int(s["threads"]), Q=s["charge"],
                           mesh_volume=s["volume"], tolerance=s["tolerance"], passes=int(s["passes"]),
                           monitor=s.get("monitor"), indicator=s.get("indicator"),
                           monitor_width=s.get("monitor_width"), monitor_floor=s.get("monitor_floor"),
                           Tlist_s=s["list_s"], Tprep_s=s["preparation_s"],
                           Teval_s=s["evaluation_s"], Ttotal_s=s["total_solver_s"],
                           mesh_s=s["mesh_s"], same_mesh_direct_s=s.get("same_mesh_direct_s"),
                           job_wall_s=run["job"].get("wall_s"),
                           reference=REFERENCE, reference_N=int(reference["N"]),
                           reference_Q=reference["charge"],
                           Q_relative_difference=abs(s["charge"] - reference["charge"]) / abs(reference["charge"]),
                           starting_mesh_sha256=mesh_hash, mesh_hash_match=True,
                           reference_values_sha256=reference_hash, targets_sha256=targets_hash,
                           actual_mesh_sha256=job["expected_mesh_sha256"],
                           historical_radial_N=int(oldstats["N"]),
                           historical_radial_Ttotal_s=oldstats["total_solver_s"])
                for prefix, measurement in (("Eref", err), ("Esame_direct", direct),
                                             ("reference_p12_to_p16", sensitivity),
                                             ("historical_radial_Eref", historical),
                                             ("fresh_radial_vs_pilot", reproducibility)):
                    row.update({f"{prefix}_{key}": value for key, value in measurement.items()
                                if key != "sample_count"})
                row["reference_change_over_error_rms"] = ratio(sensitivity["relative_rms"], err["relative_rms"])
                row["reference_change_over_error_maxabs"] = ratio(sensitivity["max_abs"], err["max_abs"])
                row["reference_guard"] = ("CHECK_PASSED" if
                    row["reference_change_over_error_rms"] <= .1 and
                    row["reference_change_over_error_maxabs"] <= .1 else "UNRESOLVED")
                run["groups"][group] = row
                summary.append(row)
            radial, localized = (runs[n]["groups"][group] for n in RUNS)
            for row in (radial, localized):
                row.update(localized_over_radial_rms=ratio(localized["Eref_relative_rms"], radial["Eref_relative_rms"]),
                           localized_over_radial_maxabs=ratio(localized["Eref_max_abs"], radial["Eref_max_abs"]),
                           localized_over_radial_N=ratio(localized["N"], radial["N"]),
                           localized_over_radial_Ttotal=ratio(localized["Ttotal_s"], radial["Ttotal_s"]),
                           localized_over_historical_rms=ratio(localized["Eref_relative_rms"], historical["relative_rms"]),
                           localized_over_historical_maxabs=ratio(localized["Eref_max_abs"], historical["max_abs"]),
                           localized_over_historical_Ttotal=ratio(localized["Ttotal_s"], oldstats["total_solver_s"]))
        domains.append(dict(case=case, reference=reference, runs=runs, checks=checks,
                            repeat=repeat, historical=oldstats, mesh_hash=mesh_hash))
    return manifest, domains, summary


def conclusion(domain):
    parts = []
    for group in GROUPS:
        row = domain["runs"]["localized"]["groups"][group]
        rms, maximum = row["localized_over_radial_rms"], row["localized_over_radial_maxabs"]
        if abs(maximum - 1) < .001:
            verdict = ("lower RMS, nearly unchanged maximum" if rms <= 1
                       else "higher RMS, nearly unchanged maximum")
        elif rms > 1 and maximum > 1:
            verdict = "higher errors in both measures"
        elif rms <= 1 and maximum <= 1:
            verdict = "no increase in either measure"
        else:
            verdict = "a mixed change in the two measures"
        parts.append(f"{group}: {verdict} (RMS {factor(rms)}x, max {factor(maximum)}x)")
    unresolved = any(domain["runs"][name]["groups"][group]["reference_guard"] != "CHECK_PASSED"
                     for name in RUNS for group in GROUPS)
    tail = " Reference sensitivity fails the 10% guard; interpretation remains tentative." if unresolved else ""
    return LABELS[domain["case"]] + " — " + "; ".join(parts) + "." + tail


def repeat_note(domains):
    if all(d["repeat"][g]["max_abs"] == 0 for d in domains for g in GROUPS):
        return "All 160 stored fresh radial potentials per domain agree exactly with the pilot values (zero sampled discrepancy)."
    return "Nonzero floating-point differences are retained explicitly in the reproduction table."


def accuracy_rows(domains):
    rows = []
    for d in domains:
        for group in GROUPS:
            old, new = (d["runs"][name]["groups"][group] for name in RUNS)
            rows.append([LABELS[d["case"]], group, number(old["Eref_relative_rms"]),
                         number(new["Eref_relative_rms"]), factor(new["localized_over_radial_rms"]),
                         number(old["Eref_max_abs"]), number(new["Eref_max_abs"]),
                         factor(new["localized_over_radial_maxabs"])])
    return rows


def direct_rows(domains):
    rows = []
    for d in domains:
        for group in GROUPS:
            old, new = (d["runs"][name]["groups"][group] for name in RUNS)
            rows.append([LABELS[d["case"]], group, number(old["Esame_direct_relative_rms"]),
                         number(new["Esame_direct_relative_rms"]),
                         number(old["Esame_direct_max_abs"]), number(new["Esame_direct_max_abs"])])
    return rows


def check_rows(domains):
    rows = []
    for d in domains:
        for group in GROUPS:
            old, new = (d["runs"][name]["groups"][group] for name in RUNS)
            sensitivity = d["checks"][group]["sensitivity"]
            rows.append([LABELS[d["case"]], group, number(sensitivity["relative_rms"]),
                         number(sensitivity["max_abs"]),
                         f"{old['reference_change_over_error_rms']:.2%}",
                         f"{new['reference_change_over_error_rms']:.2%}",
                         f"{old['reference_change_over_error_maxabs']:.2%}",
                         f"{new['reference_change_over_error_maxabs']:.2%}",
                         "PASS" if all(r["reference_guard"] == "CHECK_PASSED" for r in (old, new)) else "UNRESOLVED"])
    return rows


def md_table(headers, rows):
    return ["| " + " | ".join(map(str, headers)) + " |",
            "|" + "|".join("---" for _ in headers) + "|"] + [
                "| " + " | ".join(map(str, row)) + " |" for row in rows]


SETUP = (
    "The positive model density is rho=exp(-(x+2)^2-2(y+2)^2-3z^2), restricted to each mesh domain. "
    "Fresh radial and localized solves use the actual meshes, not interpolated or synthetic potentials. "
    "The localized monitor concentrates refinement near (-2,-2,0) and leaves the outer region coarser. "
    "It is a prescribed mesh monitor, not a certified error estimator. Active element counts N are NOT matched.")
SAMPLES = (
    "The same 128 volume-uniform targets and 32 source-core targets (radius 0.9) are compared separately. "
    "All CSV comparisons require matching IDs, groups and coordinates. Relative RMS is "
    "sqrt(sum((u-u_ref)^2)/sum(u_ref^2)); max abs is max|u-u_ref|. These sampled diagnostics are not certified continuum L2 errors.")
REFERENCE_NOTE = (
    "The numerical reference is direct evaluation on the same starting geometry, globally refined twice, "
    "with p=16 and near-depth=4. The p12-to-p16/depth4 change measures sensitivity, not a rigorous error bound "
    "or a proof of mesh convergence. The 10% guard must pass in BOTH relative RMS and max abs, relative to "
    "BOTH fresh radial and localized errors. Direct evaluation bypasses HNL/MAC/multipoles but shares "
    "source, quadrature and singular-integration kernels; common-mode errors can remain.")
TIMING_NOTE = (
    "Fresh timings are single runs on a shared server, not a controlled performance benchmark. "
    "Ratios are observations, not confident speedup claims. Tlist includes hierarchy/list preparation "
    "and validation; Tprep includes quadrature and moments; Teval evaluates one potential per active leaf. "
    "Ttotal includes mesh processing plus these stages, but excludes sampled-target evaluation, "
    "same-mesh direct checks, uniform references and file export. Different N means different production workloads.")


def mesh_rows(domains):
    rows = []
    for d in domains:
        for name in RUNS:
            s = d["runs"][name]["stats"]
            rows.append([LABELS[d["case"]], RUN_LABELS[name], int(s["N"]), int(s["root_count"]),
                         f"{int(s['p'])}/{int(s['P'])}/{s['theta']:g}", int(s["threads"]), f"{s['charge']:.10g}"])
    return rows


def timing_rows(domains):
    rows = []
    for d in domains:
        for name in RUNS:
            s = d["runs"][name]["stats"]
            rows.append([LABELS[d["case"]], RUN_LABELS[name], *[f"{s[k]:.5f}" for k in TIMES]])
    return rows


def markdown_report(directory, pilot, domains, manifest):
    lines = ["# Localized-mesh potential accuracy", "", SETUP, "", SAMPLES, "",
             "## Actual meshes and fresh solver timings", ""]
    lines += md_table(["Domain", "Mesh", "N", "Roots", "p/P/theta", "Threads", "Q"], mesh_rows(domains))
    lines += [""] + md_table(["Domain", "Mesh", "Tlist (s)", "Tprep (s)", "Teval (s)", "Ttotal (s)"], timing_rows(domains))
    lines += ["", TIMING_NOTE, "", f"Fresh startup load averages: {manifest.get('load_average_at_start', 'not recorded')}.", "",
              "## Accuracy against the uniform direct reference", "",
              "Old = the fresh radial baseline; new = localized. Ratios are new/old, so a value above 1 means a larger error.", ""]
    lines += md_table(["Domain", "Group", "Old rel RMS", "New rel RMS", "RMS ratio", "Old max abs", "New max abs", "Max ratio"], accuracy_rows(domains))
    lines += ["", "## Scientific interpretation", ""]
    lines += ["- " + conclusion(d) for d in domains]
    lines += ["", "These observations retain unfavorable outcomes. They compare the chosen coarser-outer localized mesh with the radial mesh; unmatched N and different refinement distributions prevent an equal-cost comparison or an isolated causal attribution to outer coarsening.", "",
              "## Same-mesh direct discrepancies", "",
              "Each accelerated result is checked against direct evaluation on its own source mesh with production quadrature; this isolates acceleration discrepancies within that discretization.", ""]
    lines += md_table(["Domain", "Group", "Radial rel RMS", "Localized rel RMS", "Radial max abs", "Localized max abs"], direct_rows(domains))
    lines += ["", "## Reference sensitivity and geometry", "", REFERENCE_NOTE, ""]
    lines += md_table(["Domain", "Group", "Change rel RMS", "Change max abs", "RMS / old error", "RMS / new error", "Max / old error", "Max / new error", "10% guard"], check_rows(domains))
    lines += ["", "PASS means only CHECK_PASSED for this sensitivity test. UNRESOLVED means at least one of the four ratios exceeds 10%.", ""]
    refrows = [[LABELS[d["case"]], int(d["reference"]["N"]), int(d["reference"]["root_count"]),
                f"{d['reference']['volume']:.10g}", f"{d['reference']['charge']:.10g}"] for d in domains]
    lines += md_table(["Domain", "Reference N", "Roots", "Domain volume", "Reference Q"], refrows)
    lines += ["", "Starting mesh SHA256 values are checked against saved pilot base.mesh and recorded reference hashes; root counts, volumes and source definitions also agree. Both fresh jobs must report mesh_hash_match=true, tying the solver mesh to the saved comparison mesh. The sphere remains faceted; refinement does not eliminate its boundary-geometry discrepancy.", "",
              "## Historical-baseline reproduction and ratios", "",
              f"Historical data: `{pilot}`. The historical adaptive_large values are compared directly with the fresh radial values below. " + repeat_note(domains), ""]
    repeatrows = []
    ratio_rows = []
    for d in domains:
        for group in GROUPS:
            delta = d["repeat"][group]
            repeatrows.append([LABELS[d["case"]], group, number(delta["relative_rms"], 5), number(delta["max_abs"], 5)])
            new = d["runs"]["localized"]["groups"][group]
            ratio_rows.append([LABELS[d["case"]], group, factor(new["localized_over_radial_N"]),
                               factor(new["localized_over_radial_Ttotal"]), factor(new["localized_over_historical_Ttotal"]),
                               factor(new["localized_over_historical_rms"]), factor(new["localized_over_historical_maxabs"])])
    lines += md_table(["Domain", "Group", "Fresh radial vs pilot rel RMS", "Fresh radial vs pilot max abs"], repeatrows)
    lines += [""] + md_table(["Domain", "Group", "New/fresh radial N", "New/fresh radial time", "New/historical time", "New/historical RMS", "New/historical max"], ratio_rows)
    lines += ["", "Historical time ratios involve runs from different sessions and are especially load-sensitive. Full-precision values, separate 10% guard statuses and reference/charge diagnostics are in summary.csv. Raw inputs are never modified by this summary script.", ""]
    (directory / "report.md").write_text("\n".join(lines))


def paragraph(fig, x, y, text, width=139, size=9.5):
    fig.text(x, y, "\n".join(textwrap.wrap(text, width)), va="top", fontsize=size, linespacing=1.35)


def new_page(title, page):
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.text(.045, .95, title, fontsize=17, weight="semibold", va="top")
    fig.text(.045, .025, "Localized-mesh accuracy | sampled numerical diagnostics, not certified L2 errors", fontsize=8, color="#555")
    fig.text(.955, .025, str(page), fontsize=8, ha="right", color="#555")
    return fig


def pdf_report(directory, domains, manifest):
    with PdfPages(directory / "report.pdf") as pdf:
        fig = new_page("Localized-mesh potential accuracy: actual meshes and timings", 1)
        paragraph(fig, .045, .89, SETUP)
        table(fig, [.045, .58, .91, .20], ["Domain", "Mesh", "Actual N", "Roots", "p/P/theta", "Threads", "Charge Q"], mesh_rows(domains), 9)
        table(fig, [.045, .32, .91, .20], ["Domain", "Mesh", "Tlist (s)", "Tprep (s)", "Teval (s)", "Ttotal (s)"], timing_rows(domains), 9)
        paragraph(fig, .045, .27, TIMING_NOTE, size=9)
        timeratios = "; ".join(LABELS[d["case"]] + " " + factor(d["runs"]["localized"]["groups"]["volume"]["localized_over_radial_Ttotal"]) + "x" for d in domains)
        paragraph(fig, .045, .12, "Observed localized / fresh radial total-time ratios: " + timeratios + ". Startup loads: " + str(manifest.get("load_average_at_start", "not recorded")) + ".", size=9)
        pdf.savefig(fig)
        plt.close(fig)

        fig = new_page("Sampled accuracy: retain both improvements and regressions", 2)
        paragraph(fig, .045, .89, SAMPLES, size=9)
        table(fig, [.035, .60, .93, .19], ["Domain", "Group", "Radial RMS", "Local RMS", "New/old", "Radial max", "Local max", "New/old"], accuracy_rows(domains), 8.5)
        paragraph(fig, .045, .565, "RMS columns are relative; max columns are absolute. New/old means localized/fresh radial: ratios above 1 are worse. Both methods use the same tightened uniform direct reference.", size=9)
        table(fig, [.035, .30, .93, .18], ["Same-mesh direct", "Group", "Radial rel RMS", "Local rel RMS", "Radial max abs", "Local max abs"], direct_rows(domains), 8.5)
        for index, domain in enumerate(domains):
            paragraph(fig, .045, .265 - index * .06, conclusion(domain), width=148, size=8.6)
        paragraph(fig, .045, .07, "Unmatched N and different refinement distributions prevent an equal-cost comparison or isolated attribution to outer coarsening. These are sample-specific observations, not continuum guarantees.", width=155, size=8)
        pdf.savefig(fig)
        plt.close(fig)

        fig = new_page("Reference sensitivity, geometry checks and reproducibility", 3)
        paragraph(fig, .045, .89, REFERENCE_NOTE, size=9)
        table(fig, [.025, .55, .95, .20], ["Domain", "Group", "Change RMS", "Change max", "RMS/old", "RMS/new", "Max/old", "Max/new", "10% guard"], check_rows(domains), 8)
        paragraph(fig, .045, .52, "Fractions are the reference change divided by the relevant observed error; PASS requires all four to be <=10%. PASS is not a rigorous bound. Old = fresh radial; new = localized.", size=8.5)
        rows = []
        for d in domains:
            volume, core = (d["repeat"][g] for g in GROUPS)
            rows.append([LABELS[d["case"]], int(d["reference"]["N"]), f"{d['reference']['charge']:.9g}",
                         number(volume["relative_rms"]), number(volume["max_abs"]),
                         number(core["relative_rms"]), number(core["max_abs"])])
        table(fig, [.035, .32, .93, .12], ["Domain", "Reference N", "Reference Q", "Repeat vol RMS", "Repeat vol max", "Repeat core RMS", "Repeat core max"], rows, 8)
        paragraph(fig, .045, .29, "Repeat = fresh radial minus saved pilot adaptive_large, with strict target alignment. " + repeat_note(domains) + " Full historical ratios appear in report.md and summary.csv.", size=9)
        paragraph(fig, .045, .20, "Geometry verified: current starting-mesh bytes equal saved pilot base.mesh and the reference job's mesh SHA256. Root counts, domain volumes and source definitions agree. Each fresh job records mesh_hash_match=true for the solver/illustration comparison. The sphere keeps the same faceted boundary at every mesh level.", size=9)
        paragraph(fig, .045, .09, "Same-mesh direct discrepancies test acceleration within the production discretization. Shared quadrature/singular kernels do not provide an independent check of all integration errors. Raw results remain unchanged; summary.csv contains full-precision metrics and provenance.", size=8.5)
        pdf.savefig(fig)
        plt.close(fig)


def compact_results(directory, domains):
    lines = ["# Actual localized versus radial meshes; measured potentials, no synthetic data.",
             "# Separate 128 volume and 32 core targets; sampled errors against uniform L2,p16,depth4 direct reference.",
             "# Not certified continuum L2 errors. Different N; single-run shared-server times in seconds.",
             "# Old=FRESH radial baseline; new=localized. Ratios >1 indicate larger errors/times.",
             "# Direct reference shares quadrature/singular kernels. Guard is sensitivity only, not a rigorous bound.",
             "domain mesh N roots p P theta threads Q E_volume E_core Max_volume Max_core Tlist Tprep Teval Ttotal"]
    for d in domains:
        for name in RUNS:
            run = d["runs"][name]
            s, v, c = run["stats"], run["groups"]["volume"], run["groups"]["core"]
            lines.append(f"{d['case']} {name} {int(s['N'])} {int(s['root_count'])} {int(s['p'])} {int(s['P'])} "
                         f"{s['theta']:g} {int(s['threads'])} {s['charge']:.12g} "
                         f"{v['Eref_relative_rms']:.8e} {c['Eref_relative_rms']:.8e} "
                         f"{v['Eref_max_abs']:.8e} {c['Eref_max_abs']:.8e} " + " ".join(f"{s[k]:.8f}" for k in TIMES))
        for group in GROUPS:
            row = d["runs"]["localized"]["groups"][group]
            lines.append(f"# {d['case']} {group}: new/old RMS={row['localized_over_radial_rms']:.8g} "
                         f"max={row['localized_over_radial_maxabs']:.8g} time={row['localized_over_radial_Ttotal']:.8g}; "
                         f"reference guard radial={d['runs']['radial_baseline']['groups'][group]['reference_guard']} "
                         f"localized={row['reference_guard']}")
        lines.append("# " + conclusion(d))
    (directory / "results_localized_accuracy.txt").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=HERE / "runs" / "localized_accuracy_20261108")
    parser.add_argument("--pilot", "--reference-directory", dest="pilot", type=Path,
                        default=HERE / "runs" / "pilot_20261108")
    args = parser.parse_args()
    directory, pilot = args.directory.resolve(), args.pilot.resolve()
    if not directory.is_dir() or not pilot.is_dir():
        parser.error("Both the completed-run directory and pilot directory must exist")
    manifest, domains, summary = load_study(directory, pilot)
    write_csv(directory / "summary.csv", summary)
    markdown_report(directory, pilot, domains, manifest)
    pdf_report(directory, domains, manifest)
    compact_results(directory, domains)
    print(f"Wrote report.md, report.pdf (3 pages), results_localized_accuracy.txt and summary.csv in {directory}")
    for domain in domains:
        print(conclusion(domain))


if __name__ == "__main__":
    main()
