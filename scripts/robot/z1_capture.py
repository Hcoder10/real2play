"""CPU rollout capture, then render its saved states separately on the leased GPU."""
import _bootstrap
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np

def capture(job):
    export,seed,out=job
    from r2p.policy import load
    from r2p.policy.eval import rollout
    from r2p.tasks import sample_scene
    policy=load(export); scene=sample_scene(policy.config['task_spec'],np.random.default_rng(seed)); frames=[]
    def record(row):
        if row['tick']%9==0:
            frames.append({'tick':row['tick'],'q':row['q'].tolist(),'poses':row['state']['objects'],'command':row['command']})
    result=rollout(policy,scene,seed=seed,noise='clean',on_step=record)
    path=Path(out)/f'policy_clean_{seed}.json'; path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({'seed':seed,'result':result,'scene':scene,'physics':policy.config['physics'],'frames':frames}))
    print(seed,result,flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--export',default='exports/bowl_v1'); p.add_argument('--seeds',type=int,nargs='+',default=[10003,10000]); p.add_argument('--out',default='out/z1/captures'); p.add_argument('--render')
    a=p.parse_args()
    if a.render:
        from z1_video import Recorder
        data=json.loads(Path(a.render).read_text()); result=data['result']; outcome='SUCCESS' if result['success'] else 'FAILURE'
        path=Path(a.out)/Path(a.render).with_suffix('.mp4').name
        rec=Recorder(data['scene'],data['physics'],path,f'Trained policy | seed {data["seed"]} | {outcome}')
        try:
            for f in data['frames']:rec.frame(f['tick'],f['q'],f['poses'],f['command'])
        finally:rec.close()
        path.with_suffix('.json').write_text(json.dumps({'seed':data['seed'],'result':result,'frames':rec.frames},indent=2))
        print(path,rec.frames,flush=True)
    else:
        with ProcessPoolExecutor(max_workers=min(2,len(a.seeds))) as pool:list(pool.map(capture,[(a.export,s,a.out) for s in a.seeds]))

if __name__=='__main__':main()
