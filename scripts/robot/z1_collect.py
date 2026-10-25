"""B200-only parallel bowl expert probe/collection; preserves failure evidence."""
import _bootstrap
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
from r2p.sim.bowl_expert import Expert
from r2p.tasks import load_task
from r2p.traces import write_trace

def episode(job):
    task, profile, seed, output, save = job
    expert = Expert(task, deployment=profile)
    expert.arm.range[4] = profile['roll_window_rad']
    result = expert.run(seed=seed)
    ticks = result['ticks']
    targets = np.asarray([x['q_target'] for x in ticks])
    bounds = np.asarray(profile['model_limits'])
    row = {'seed': seed, 'success': result['meta']['success'],
           'reason': result['meta']['fail_reason'], 'ticks': len(ticks),
           'limit_violation': float(max(0, (bounds[:,0]-targets).max(), (targets-bounds[:,1]).max())),
           'final': result['final']}
    if save and (row['success'] or seed < 6):
        folder = Path(output)/('success' if row['success'] else 'failure')
        path = folder/f'expert_{seed:08d}.jsonl'
        temporary = path.with_suffix('.jsonl.tmp')
        write_trace(temporary, result['meta'], ticks)
        temporary.replace(path)
    return row

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task',required=True); p.add_argument('--profile',required=True)
    p.add_argument('--out',required=True); p.add_argument('--n',type=int,default=192)
    p.add_argument('--workers',type=int,default=190); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--save',action='store_true')
    a=p.parse_args(); task=load_task(a.task)
    profile=json.loads(Path(a.profile).read_text())
    output=Path(a.out); output.mkdir(parents=True,exist_ok=True)
    start=time.time(); rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        jobs=[pool.submit(episode,(task,profile,s,a.out,a.save)) for s in range(a.seed,a.seed+a.n)]
        for f in as_completed(jobs):
            rows.append(f.result())
            if len(rows)%32==0: print(len(rows),sum(r['success'] for r in rows),flush=True)
    report={'task_spec':task,'profile':profile,'n':a.n,'successes':sum(r['success'] for r in rows),
            'failures':dict(Counter(r['reason'] for r in rows if not r['success'])),
            'seconds':time.time()-start,'episodes':sorted(rows,key=lambda r:r['seed'])}
    (output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('episodes','task_spec','profile')}),flush=True)

if __name__=='__main__': main()
