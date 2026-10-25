"""P2 no-augmentation recipe with parallel preprocessing and resident GPU batches."""
import _bootstrap
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from r2p.policy.data import TraceDataset, Augmentation
from r2p.policy.features import export_config
from r2p.policy.runtime import clamp_constants
from r2p.tasks import load_task

def prepare(job):
    task, path, config, output = job
    ds=TraceDataset(task,path,config,Augmentation(0,0,0,6,0,0,4,1))
    np.savez(output,x=ds.features,y=ds.labels)
    return {'file':str(output),'start_q':ds.start_q.tolist(),'deployment':ds.deployment,
            'physics':ds.physics,'provenance':ds.provenance}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--task',required=True); p.add_argument('--traces',required=True)
    p.add_argument('--out',required=True); p.add_argument('--steps',type=int,default=60000)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]); p.add_argument('--workers',type=int,default=20)
    p.add_argument('--limit',type=int); p.add_argument('--width',type=int,default=512); p.add_argument('--blocks',type=int,default=3)
    p.add_argument('--final-lr',type=float,default=3e-4,help='Optional linear decay after half the updates; default retains P2 constant LR')
    a=p.parse_args(); task=load_task(a.task); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    cfg=export_config(task,out.name,seeds=a.seeds,width=a.width,blocks=a.blocks)
    cfg['qualification']={'ready_for_real':False,'reason':'Training export awaits recorded twin evaluations and guarded rehearsal.'}
    files=sorted(Path(a.traces).glob('*.jsonl'))
    if a.limit: files=[f for f in files if time.time()-f.stat().st_mtime>5][:a.limit]
    cache=out/'dataset'; cache.mkdir(exist_ok=True)
    jobs=[(task,str(f),cfg,str(cache/(f.stem+'.npz'))) for f in files]
    print('preparing',len(jobs),'traces',flush=True)
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,row in enumerate(pool.map(prepare,jobs)):
            rows.append(row)
            if (i+1)%32==0: print('prepared',i+1,flush=True)
    if not rows: raise ValueError('No traces')
    for row in rows:
        if any(row[k]!=rows[0][k] for k in ('start_q','deployment','physics')): raise ValueError('Mixed trace conventions')
    arrays=[np.load(row['file']) for row in rows]
    x=np.concatenate([z['x'] for z in arrays]); y=np.concatenate([z['y'] for z in arrays])
    for z in arrays: z.close()
    mean=x.mean(0,dtype=np.float64); std=x.std(0,dtype=np.float64)
    cfg['obs_const']=[{'index':int(i),'mean':float(mean[i]),'std':float(std[i]),'threshold':float(5*max(std[i],1e-3))} for i in np.flatnonzero(std<1e-6)]
    norm={'x_mean':mean.tolist(),'x_std':np.maximum(std,.01).tolist(),
          'y_mean':y.mean(0,dtype=np.float64).tolist(),'y_std':np.maximum(y.std(0,dtype=np.float64),.05).tolist()}
    cfg.update(start_q=rows[0]['start_q'],deployment=rows[0]['deployment'],physics=rows[0]['physics'])
    cfg['joint_limits']=deepcopy(cfg['deployment']['model_limits']); cfg['joint_limits'][4]=cfg['deployment']['roll_window_rad']
    cfg['gripper_open_threshold_rad']=cfg['deployment']['gripper_open_threshold_rad']; cfg['pick_lift_m']=.03
    cfg['physics_sha256']=hashlib.sha256(json.dumps(cfg['physics'],sort_keys=True).encode()).hexdigest()
    import torch
    from r2p.policy.network import StatePolicy
    from r2p.policy.train import _device
    device=_device(torch,'cuda:0',170/179.06); torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    x=clamp_constants(x,cfg['obs_const'])
    tx=torch.as_tensor((x-np.asarray(norm['x_mean']))/np.asarray(norm['x_std']),dtype=torch.float32,device=device)
    ty=torch.as_tensor((y-np.asarray(norm['y_mean']))/np.asarray(norm['y_std']),dtype=torch.float32,device=device)
    del x,y
    weights={}; losses=[]; started=time.time()
    for member,seed in enumerate(a.seeds):
        torch.manual_seed(seed); model=StatePolicy(cfg['input_dim'],width=a.width,blocks=a.blocks).to(device)
        learning_rate=torch.tensor(3e-4,device=device)
        opt=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=1e-4,capturable=True)
        def step():
            idx=torch.randint(len(tx),(256,),device=device)
            prediction=model(tx[idx]); loss=torch.nn.functional.smooth_l1_loss(prediction,ty[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
            return loss
        # Capture only after optimizer state and kernels are initialized.
        stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3): step()
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): loss=step()
        history=[]
        for i in range(a.steps-4):
            if i%1000==0:
                progress=max(0.,(i+4-a.steps/2)/(a.steps/2))
                learning_rate.fill_(3e-4+(a.final_lr-3e-4)*progress)
            graph.replay()
            if (i+5)%1000==0:
                value=float(loss); history.append({'step':i+5,'loss':value})
                print(json.dumps({'member':member,'step':i+5,'loss':value,'elapsed':time.time()-started}),flush=True)
        torch.cuda.synchronize()
        weights.update({f'member{member}.{k}':v.detach().cpu().numpy() for k,v in model.state_dict().items()})
        losses.append({'seed':seed,'loss':history})
        cfg['training']={'steps':a.steps,'batch_size':256,'augmentation':False,'recipe':'P2; resident CUDA batches; uniform sampling',
                         'learning_rate':3e-4,'final_learning_rate':a.final_lr,
                         'traces':[v for row in rows for v in row['provenance']], 'samples':len(tx),'losses':losses,
                         'lease_id':os.environ['LANDLORD_LEASE_ID']}
        # Each completed member is immediately a recoverable, explicitly unevaluated export.
        cfg['ensemble_size']=member+1; cfg['seeds']=a.seeds[:member+1]
        np.savez_compressed(out/'weights.npz',**weights)
        cfg['weights_sha256']=hashlib.sha256((out/'weights.npz').read_bytes()).hexdigest()
        (out/'norm.json').write_text(json.dumps(norm,indent=2))
        (out/'config.json').write_text(json.dumps(cfg,indent=2))
        shutil.copyfile('r2p/policy/runtime.py',out/'runtime.py')
        print('EXPORTED',member+1,flush=True)
        del graph,model,opt
    print('TRAIN_DONE',flush=True)

if __name__=='__main__': main()
