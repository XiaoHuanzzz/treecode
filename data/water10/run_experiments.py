#!/usr/bin/env python3
"""Run the C++ water10 solver; preserve logs and incremental CSV results."""
import csv
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'water10_case' / 'experiment_batch'
OUT.mkdir(exist_ok=True)
env = os.environ.copy()
env.update(OMP_NUM_THREADS='4', WATER10_PADDING='4',
           TREECODE_CPU_INTERACTIONS='CACHED', TREECODE_HNL_MAC_FAMILIES='0',
           TREECODE_HNL_SHARED_TARGETS='0')
env['AFEPACK_TEMPLATE_PATH'] = ':'.join('/usr/local/include/AFEPack/template/' + x + '/'
    for x in ['triangle', 'twin_triangle', 'tetrahedron', 'twin_tetrahedron', 'four_tetrahedron'])
cases = [(f'mesh_{tol}', tol, 6, 4, .5) for tol in [.1, .03, .01, .003]]
# Single-parameter interventions, followed by their combined intervention.
for tol in [.03, .003]:
    for label,p,P,theta in [('p',10,4,.5),('P',6,8,.5),('theta',6,4,.3),('combined',10,8,.3)]:
        cases.append((f'{label}_{tol}',tol,p,P,theta))

def extract(text, label):
    match = re.search(re.escape(label) + r'\s*([\d.eE+\-]+)', text)
    if not match:
        raise ValueError('Missing output: '+label)
    return float(match[1])

rows=[]
for name,tol,p,P,theta in cases:
    for repeat in range(1,4):
        log=OUT/f'{name}_r{repeat}.log'
        command=[str(ROOT/'tree_water10'),'D.mesh','1',str(p),str(P),str(theta),'1','CPU',
                 format(tol*10/3,'.12g'),str(tol),'RADIAL','6','0']
        if not log.exists() or 'Total execution time:' not in log.read_text():
            print('START',name,repeat,flush=True)
            started=time.monotonic()
            with log.open('w') as stream:
                try:
                    run=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
                except subprocess.TimeoutExpired:
                    raise SystemExit('Timed out; preserved '+str(log))
            if run.returncode:
                raise SystemExit('Failed; preserved '+str(log))
            print('DONE',name,repeat,round(time.monotonic()-started,3),'s',flush=True)
        text=log.read_text()
        row=dict(case=name,repeat=repeat,tol1=tol*10/3,tol2=tol,p=p,P=P,theta=theta,
            N=int(extract(text,'Exact-comparison leaf barycenters:')),
            threads=int(extract(text,'OpenMP threads actually used:')),
            T_list=extract(text,'Filling neighbor_elements uses:'),
            T_moment=extract(text,'Updating multipoleInfo uses'),
            T_eval=extract(text,'Potential evaluation time:'),
            T_total=extract(text,'Total execution time:'),
            E_L2=extract(text,'L2 error against exact solution:'),
            E_relative=extract(text,'Relative L2 error against exact solution:'),
            charge=extract(text,'Integrated source charge:'),log=str(log.relative_to(ROOT)))
        rows.append(row)
        with (OUT/'runs.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerows(rows)

summary=[]
for name,*_ in cases:
    group=[r for r in rows if r['case']==name]
    r=group[0]
    s={k:r[k] for k in ['case','tol1','tol2','p','P','theta','N','threads','E_L2','E_relative','charge']}
    for key in ['T_list','T_moment','T_eval','T_total']:
        values=[r[key] for r in group]
        s[key+'_median']=statistics.median(values)
        s[key+'_min']=min(values);s[key+'_max']=max(values)
    s['error_repeat_spread']=max(r['E_L2'] for r in group)-min(r['E_L2'] for r in group)
    summary.append(s)
with (OUT/'summary.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
print('FINISHED:',len(rows),'runs;',OUT/'summary.csv',flush=True)
