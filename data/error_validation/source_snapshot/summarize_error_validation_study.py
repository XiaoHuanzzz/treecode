#!/usr/bin/env python3
"""Verify paired error vectors, summarize target refinement, and plot evidence."""
import csv
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "article_error_validation"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 12,
                     "axes.labelsize": 12, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "legend.fontsize": 10})


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write(path, data):
    with path.open("w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(data[0]))
        out.writeheader()
        out.writerows(data)


def values(row):
    data = np.genfromtxt(ROOT / row["target_file"], delimiter=",", names=True)
    if not np.all(np.isfinite(data["tree"])) or not np.all(data["weight"] > 0):
        raise RuntimeError("Invalid target data")
    return data


def save(fig, name):
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{suffix}", dpi=190)
    plt.close(fig)


def main():
    raw = read(OUT / "components_raw.csv")
    sampling = read(OUT / "sampling.csv")
    molecular = read(OUT / "molecular.csv")
    molecular += read(OUT / "molecular_fine.csv")
    selected = read(OUT / "selected.csv")
    if (len(raw), len(sampling), len(molecular)) != (12, 18, 20):
        raise RuntimeError("Study incomplete; do not publish partial evidence")
    if len(selected) != 4:
        raise RuntimeError("Selected-candidate validation incomplete")
    direct_refs = {(r["p"],r["seed"]): values(r) for r in raw
                   if r["direct_enabled"] == "1"}
    components = []
    for row in raw:
        data = values(row)
        reference = direct_refs[(row["p"],row["seed"])]
        for field in ("id","x","y","z","weight","exact"):
            if not np.array_equal(data[field], reference[field]):
                raise RuntimeError("Cannot pair different validation targets")
        w = data["weight"]
        a = data["tree"] - reference["direct"]
        b = reference["direct"] - data["exact"]
        total = data["tree"] - data["exact"]
        norm = lambda v: math.sqrt(math.fsum(w*v*v))
        cross = math.fsum(w*a*b)
        err, fast, integration = norm(total), norm(a), norm(b)
        residual = abs(err*err-fast*fast-integration*integration-2*cross)
        if err > fast+integration+1e-12 or residual > 1e-12*(err*err+fast*fast+integration*integration):
            raise RuntimeError("Error decomposition failed")
        components.append(dict(
            N=int(row["N"]), p=int(row["p"]), P=int(row["P"]),
            theta=float(row["theta"]), seed=int(row["seed"]),
            targets=len(data), E_total=err, E_fast=fast,
            E_integration=integration, reference_norm=norm(data["exact"]),
            error_cosine=cross/(fast*integration),
            squared_identity_residual=residual))
    write(OUT / "components.csv", components)

    reference_norm = (math.pi/(8*math.pi))**.25 * \
                     (math.pi/(16*math.pi))**.25 * \
                     (math.pi/(24*math.pi))**.25
    grouped = []
    for scale in ("N10k","N100k","N800k"):
        for m in (16,32,64):
            group = [r for r in sampling if r["scale"]==scale and int(r["inner"])==m]
            if len(group) != 2:
                raise RuntimeError("Missing sampling replicate")
            errors = [float(r["E_total"]) for r in group]
            norms = [float(r["reference_norm"]) for r in group]
            pooled = math.sqrt(sum(x*x for x in errors)/2)
            pooled_norm = math.sqrt(sum(x*x for x in norms)/2)
            grouped.append(dict(
                scale=scale, N=int(group[0]["N"]), inner=m,
                targets_per_seed=int(group[0]["targets"]), seeds=2,
                E_pooled=pooled, E_min=min(errors), E_max=max(errors),
                reference_norm_pooled=pooled_norm,
                reference_norm_relative_bias=abs(pooled_norm/reference_norm-1),
                seed_spread=(max(errors)-min(errors))/pooled))
    write(OUT / "sampling_summary.csv", grouped)

    fig, axes = plt.subplots(1,3,figsize=(10.5,3.5))
    predicates = [
        ("p",lambda r:r["P"]==4 and r["theta"]==.5,r"$P=4,\ \theta=0.5$"),
        ("P",lambda r:r["p"]==6 and r["theta"]==.5,r"$p=6,\ \theta=0.5$"),
        ("theta",lambda r:r["p"]==6 and r["P"]==4,r"$p=6,\ P=4$")]
    for ax,(parameter,test,title) in zip(axes,predicates):
        group = sorted((r for r in components if r["seed"]==7351 and test(r)),
                       key=lambda r:r[parameter])
        for field,label,style in [
                ("E_total","Tree - analytic","o-"),
                ("E_integration","Direct - analytic","s--"),
                ("E_fast","Tree - direct","^:")]:
            ax.semilogy([r[parameter] for r in group],[r[field] for r in group],
                        style,label=label,markersize=4)
        ax.set(xlabel=r"$\theta$" if parameter=="theta" else f"${parameter}$",
               title=title)
        ax.set_xticks([r[parameter] for r in group])
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Weighted absolute error")
    axes[0].legend(fontsize=10)
    save(fig,"error_components")

    fig, axes = plt.subplots(1,2,figsize=(9.5,3.6))
    for scale in ("N10k","N100k","N800k"):
        group = [r for r in grouped if r["scale"]==scale]
        x = [r["targets_per_seed"] for r in group]
        y = [r["E_pooled"] for r in group]
        axes[0].errorbar(x,y,yerr=([r["E_pooled"]-r["E_min"] for r in group],
                                  [r["E_max"]-r["E_pooled"] for r in group]),
                         marker="o",capsize=3,label=f"N={group[0]['N']:,}")
    axes[0].set(xscale="log",yscale="log",xlabel="Validation targets per seed",
                ylabel="Pooled absolute error")
    axes[0].legend(fontsize=10)
    for seed in (7351,19477):
        group = [r for r in sampling if r["scale"]=="N100k" and int(r["seed"])==seed]
        axes[1].plot([int(r["targets"]) for r in group],
                     [100*abs(float(r["reference_norm"])/reference_norm-1) for r in group],
                     "o-",label=f"Seed {seed}")
    axes[1].set(xscale="log",yscale="log",xlabel="Validation targets per seed",
                ylabel="Reference-norm discrepancy (%)")
    axes[1].legend(fontsize=10)
    for ax in axes: ax.grid(alpha=.2)
    save(fig,"sampling_refinement")

    production = read(ROOT / "article_hnl_study" / "summary.csv")
    production = sorted((r for r in production if r["p"]=="6" and r["P"]=="4"
                         and float(r["theta"])==.5),key=lambda r:int(r["N"]))
    old_validation = read(ROOT / "article_hnl_study" / "validation.csv")
    old_validation = sorted(old_validation,key=lambda r:int(r["N"]))
    refined = [r for r in grouped if r["inner"]==64]
    fig, axes = plt.subplots(1,2,figsize=(9.5,3.6))
    n = [int(r["N"]) for r in production]
    axes[0].loglog(n,[float(r["E_abs"]) for r in production],"s--",
                   label="Leaf barycenters")
    axes[0].loglog(n,[float(r["E_common_abs"]) for r in old_validation],"^:",
                   label="Earlier uniform targets")
    axes[0].loglog(n,[r["E_pooled"] for r in refined],"o-",
                   label="Refined focused targets")
    axes[0].set(xlabel="Active leaf elements N",ylabel="Absolute sampled error")
    for key,label in [("T_list",r"$T_{\mathrm{list}}$"),
                      ("T_eval",r"$T_{\mathrm{eval}}$"),
                      ("T_total",r"$T_{\mathrm{total}}$")]:
        axes[1].loglog(n,[float(r[key]) for r in production],"o-",label=label)
    axes[1].set(xlabel="Active leaf elements N",ylabel="Production time (s)")
    for ax in axes: ax.grid(alpha=.2); ax.legend(fontsize=10)
    save(fig,"refined_mesh_scaling")

    for model,path,grids in [
            ("single_h2o","single_h2o_hnl_refinement/summary.csv",(64,96,128)),
            ("ten_h2o","water10_case/joint_refinement/summary.csv",(48,64))]:
        production = read(ROOT / path)
        fig, axes = plt.subplots(1,2,figsize=(9.5,3.6))
        for m in grids:
            group = [r for r in molecular if r["model"]==model and int(r["grid"])==m]
            axes[0].loglog([int(r["N"]) for r in group],
                           [float(r["E_relative"]) for r in group],"o-",label=f"m={m}")
        for index,r in enumerate(production):
            val = next(v for v in molecular if v["model"]==model
                       and v["level"]==r["level"] and int(v["grid"])==grids[-1])
            axes[0].annotate(f"({r['p']},{r['P']})",(int(r["N"]),float(val["E_relative"])),
                             xytext=(-5 if index==len(production)-1 else 4,7),
                             ha="right" if index==len(production)-1 else "left",
                             textcoords="offset points",fontsize=10)
        axes[0].set(xlabel="Active leaf elements N",ylabel="Relative fixed-target error")
        for key,label in [("T_list",r"$T_{\mathrm{list}}$"),
                          ("T_eval",r"$T_{\mathrm{eval}}$"),
                          ("T_total",r"$T_{\mathrm{total}}$")]:
            axes[1].loglog([int(r["N"]) for r in production],
                           [float(r[key]) for r in production],"o-",label=label)
        axes[1].set(xlabel="Active leaf elements N",ylabel="Production time (s)",
                    title="Production: 4 CPU threads")
        for ax in axes:
            if model=="single_h2o":
                ax.set_xscale("linear")
                ax.xaxis.set_major_locator(FixedLocator([12000,16000,20000,24000]))
            else:
                ax.xaxis.set_major_locator(FixedLocator([2000,5000,10000,20000]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f"{x/1000:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel(r"Active leaf elements $N$ ($10^3$)")
            ax.margins(y=.13)
            ax.grid(alpha=.2)
            ax.legend(fontsize=10)
        save(fig,model+"_refined_validation")

    # Uniform Coulomb tail bound: split |x-y|<1 and >=1 outside the source box.
    # |rho| <= [2 sum(a_i)+4 sum(a_i^2 y_i^2)] exp(-sum a_i y_i^2)/(4*pi).
    a = [4*math.pi,8*math.pi,12*math.pi]
    full0 = [math.sqrt(math.pi/x) for x in a]
    full2 = [math.sqrt(math.pi)/(2*x**1.5) for x in a]
    tail0 = [math.sqrt(math.pi/x)*math.erfc(5*math.sqrt(x)) for x in a]
    tail2 = [5*math.exp(-25*x)/x + math.sqrt(math.pi)/(2*x**1.5)*math.erfc(5*math.sqrt(x))
             for x in a]
    mass_bound = 0
    for i in range(3):
        z0 = [tail0[j] if j==i else full0[j] for j in range(3)]
        z2 = [tail2[j] if j==i else full2[j] for j in range(3)]
        mass_bound += 2*sum(a)*math.prod(z0)
        for j in range(3):
            mass_bound += 4*a[j]**2*z2[j]*math.prod(z0[k] for k in range(3) if k!=j)
    mass_bound /= 4*math.pi
    maximum_bound = max(
        (2*sum(a)+sum(4*a[j]/math.e for j in range(3) if j!=i)+100*a[i]**2)
        *math.exp(-25*a[i])/(4*math.pi) for i in range(3))
    potential_tail_bound = 2*math.pi*maximum_bound+mass_bound
    lines = ["Independent error validation", "============================",
             "All computations: CPU C++ HNL -> MAC cached production kernels.",
             "All new times are validation-only; production timing tables are unchanged.",
             f"Analytical single-center norm: {reference_norm:.16g}.",
             f"Single-center uniform potential-tail upper bound: {potential_tail_bound:.6e}.",
             "",
             "Same-mesh direct comparison",
             "---------------------------",
             "Each diagnostic vector uses 4096 identical weighted targets across parameters.",
             "D_p: all-source direct quadrature with the same target-element singular rule.",
             "e_total=(T-D_p)+(D_p-V_ex); norms are not added as an equality.",
             "The acceleration difference is not the exact-moment theoretical truncation term.",
             "The integration/reference difference includes domain tails.",
             "These are finite-target diagnostics, not certified continuous norms.",
             "The diagnostic reference norm differs from its analytic value by",
             "approximately 20% (seed 7351): do not use this sparse set as a",
             "continuous-L2 estimate. The dense target refinement below is separate.",
             "",
             "p P theta seed     E_total       E_fast         E_integration"]
    lines += [f"{r['p']:2} {r['P']:2} {r['theta']:.1f} {r['seed']:5}  "
              f"{r['E_total']:.6e}  {r['E_fast']:.6e}  {r['E_integration']:.6e}"
              for r in components]
    lines += ["", "Target-set refinement", "---------------------",
              "Two predetermined seeds; ranges are sensitivity checks, NOT confidence intervals.",
              "N       inner targets/seed E_pooled       E_min          E_max          seed_spread"]
    lines += [f"{r['N']:7} {r['inner']:5} {r['targets_per_seed']:12} "
              f"{r['E_pooled']:.6e} {r['E_min']:.6e} {r['E_max']:.6e} {r['seed_spread']:.3%}"
              for r in grouped]
    lines += ["", "Molecular target refinement", "---------------------------"]
    for model,grids in (("single_h2o",(64,96,128)),("ten_h2o",(48,64))):
        for level in range(1,5):
            group=[r for r in molecular if r["model"]==model and int(r["level"])==level]
            ordered=sorted(group,key=lambda r:int(r["grid"]))
            lo,hi=ordered[-2:]
            change=abs(float(hi["E_relative"])-float(lo["E_relative"]))/float(hi["E_relative"])
            lines.append(f"{model} level {level}: " + ", ".join(
                f"E_{r['grid']}={float(r['E_relative']):.8e}" for r in ordered)
                + f", last-refinement difference={change:.3%}")
    lines += ["", "Previously selected cluster configurations", "-----------------------------------------",
              "p=12, P=6, theta=0.7. Selection preceded this finer validation."]
    for n in (7164,22298):
        group=sorted((r for r in selected if int(r["N"])==n),key=lambda r:int(r["grid"]))
        errors=[float(r["E_relative"]) for r in group]
        lines.append(f"N={n}: E_48={errors[0]:.8e}, E_64={errors[1]:.8e}, "
                     f"max={max(errors):.8e}")
    lines += ["", "Historical logs, meshes, tables and sparse validation results are preserved.",
              "All refined meshes match production mesh hashes.",
              "Convergence and component dominance must be interpreted using these measured data;",
              "a small reference-norm discrepancy does not certify the error norm."]
    (OUT / "report.txt").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
