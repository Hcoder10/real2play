"""Parallel held-out twin eval and optional calibrated-camera policy recordings."""
import _bootstrap
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from r2p.policy import load
from r2p.policy.eval import rollout, NOISE
from r2p.tasks import sample_scene

def episode(job):
    export,seed,noise,video=job
    policy=load(export); scene=sample_scene(policy.config['task_spec'],np.random.default_rng(seed))
    rec=None
    if video:
        from z1_video import Recorder
        rec=Recorder(scene,policy.config['physics'],video,f'Trained policy | {noise} | seed {seed}')
    def record(row):
        if rec: rec.frame(row['tick'],row['q'],row['state']['objects'],row['command'])
    try: result=rollout(policy,scene,seed=seed,noise=noise,on_step=record if rec else None)
    finally:
        if rec: rec.close()
    result.update(seed=seed,noise=noise)
    if video: Path(video).with_suffix('.json').write_text(json.dumps(result,indent=2))
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--export',required=True); p.add_argument('--n',type=int,default=30)
    p.add_argument('--workers',type=int,default=20); p.add_argument('--video-dir'); p.add_argument('--video-seeds',type=int,nargs='*')
    p.add_argument('--noise',choices=NOISE,default='clean'); p.add_argument('--seed',type=int,default=10000)
    p.add_argument('--no-record',action='store_true',help='Write report only for concurrent noise evaluations')
    a=p.parse_args(); directory=Path(a.export); started=time.time()
    seeds=a.video_seeds if a.video_seeds else range(a.seed,a.seed+a.n)
    jobs=[(a.export,s,a.noise,str(Path(a.video_dir)/f'policy_{a.noise}_{s}.mp4') if a.video_dir else None) for s in seeds]
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(episode,j) for j in jobs]):
            rows.append(f.result()); print(json.dumps(rows[-1]),flush=True)
    cfg=json.loads((directory/'config.json').read_text())
    report={'n':len(rows),'noise':a.noise,'seed':a.seed,'scene_seeds':list(seeds),
            'success':sum(r['success'] for r in rows)/len(rows),'episodes':sorted(rows,key=lambda r:r['seed']),
            'failures':dict(Counter(r['reason'] for r in rows if not r['success'])),
            'seconds':time.time()-started,'weights_sha256':hashlib.sha256((directory/'weights.npz').read_bytes()).hexdigest(),
            'joint_step_rad':(cfg.get('deployment') or {}).get('joint_step_rad',.10),
            'physics_sha256':cfg['physics_sha256']}
    if not a.video_dir and not a.no_record:
        cfg['eval'][a.noise]=report; cfg['eval']['n']=len(rows)
        temp=directory/'config.json.tmp'; temp.write_text(json.dumps(cfg,indent=2)); temp.replace(directory/'config.json')
    (directory/f'eval_{a.noise}{"_video" if a.video_dir else ""}.json').write_text(json.dumps(report,indent=2))
    print('RESULT',report['success'],report['seconds'],flush=True)

if __name__=='__main__': main()
