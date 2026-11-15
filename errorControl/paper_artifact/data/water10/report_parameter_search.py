import csv,json
from pathlib import Path
out=Path(__file__).resolve().parent/'parameter_search'
data=json.loads((out/'selection.json').read_text())
screen=list(csv.DictReader((out/'screening.csv').open()))
lines=['Water-decamer staged parameter search','====================================',
 'CPU: 4 actual OpenMP threads; padding=4 bohr; same 30-center Gaussian model.',
 'All physical calculations performed by C++ tree_water10.',
 'Search: p=6,8,10,12 on tol2=0.03; retained p=10,12 for P=2,4,6 and theta=0.5,0.7,0.9.',
 'Promising settings also tested with tol2=0.05 and 0.01. tol1=(10/3)*tol2.',
 'Mesh indicator order=6; initial refinement=1; CACHED HNL lists; no sharing/family packing.',
 f'Screened configurations: {len(screen)} (including reused baseline results).',
 '', 'Selected settings (median of 3 solver timings; validation excluded):']
for r in data['winners']:
    if 'N' not in r:
        lines.append(f"Target {r['threshold']:.1e}: {r['status']}")
        continue
    lines.append(f"Target {r['threshold']:.1e}: N={r['N']}, p={r['p']}, P={r['P']}, theta={r['theta']}, tol2={r['tol2']}")
    lines.append(f"  Total={r['T_total_median']:.4f} s; evaluation={r['T_eval_median']:.4f} s; "
                 f"common errors (20^3 / 32^3)={r['error20']:.5e} / {r['error32']:.5e}")
lines+=['','Independent validation:',
 'Fixed stratified target sets spanning the same domain: 20^3=8000 and 32^3=32768 points.',
 'The larger of the TWO sampled relative L2 errors is used for threshold screening.',
 'These are not certified continuous-L2 upper bounds; further sampling can change the estimate.',
 'The cached MAC is conservative over each entire leaf, so its lists also cover off-center targets.',
 'Validation uses the original C++ singular/regular/multipole evaluation at these targets.',
 'Validation runtime is printed separately, after total solver timing.',
 '', 'Important finding:',
 'Leaf-barycenter errors are often substantially smaller than errors at common off-center points.',
 'Do not use a barycenter error threshold as a claim of the same whole-domain accuracy.',
 'The previous experiment tables remain valid for their stated barycenter-discrete metric.',
 'Possible causes of the discrepancy require a separate near-interface quadrature investigation;',
 'this search does not diagnose or change that algorithm.',
 '', 'All screened configurations (single screening timing, not necessarily repeated):',
 f"{'N':>7} {'tol2':>7} {'p':>3} {'P':>3} {'theta':>6} {'Ttotal':>10} {'Teval':>10} {'bary_error':>13}"]
for r in screen:
    lines.append(f"{r['N']:>7} {r['tol2']:>7} {r['p']:>3} {r['P']:>3} {r['theta']:>6} "
                 f"{float(r['T_total']):10.4f} {float(r['T_eval']):10.4f} {float(r['error']):13.5e}")
lines+=['','Common-target validation of finalists:']
for r in data['validation']:lines.append(json.dumps(r,sort_keys=True))
lines+=['','Scope: best among tested and validated contenders, NOT a globally optimal parameter set.',
 'Thresholds are numerical benchmark targets, not established DFT energy/force tolerances.',
 'Model is manufactured density, not SCF water density; no Poisson advantage is claimed.',
 'Total timing retains the original solver definition: excludes wrapper setup, includes native error accumulation.',
 'Default makefile parameters were not changed. Raw logs and all_runs.csv retain the evidence.']
(out/'report.txt').write_text('\n'.join(lines)+'\n')
print(out/'report.txt')
