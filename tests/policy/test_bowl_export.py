import json
from pathlib import Path
import numpy as np
from r2p.tasks import load_task, sample_scene, commands, encode_slots
from r2p.traces import write_trace

def test_numpy_gelu_preserves_training_precision():
    import torch
    from r2p.policy.runtime import gelu
    x=np.linspace(-5,5,101,dtype=np.float32)
    actual=gelu(x)
    expected=torch.nn.functional.gelu(torch.from_numpy(x),approximate='tanh').numpy()
    assert actual.dtype==np.float32
    np.testing.assert_allclose(actual,expected,atol=5e-7,rtol=1e-5)

def test_free_roll_export_retains_window(tmp_path):
    from r2p.policy.train import train
    from r2p.policy import load
    root=Path(__file__).resolve().parents[2]
    task=load_task(root/'scripts/robot/Z1_BOWL_TASK.json')
    profile=json.loads((root/'scripts/robot/Z1_BOWL_PROFILE.json').read_text())
    q=np.asarray(profile['start_q']); scene=sample_scene(task,np.random.default_rng(0))
    poses={o['id']:o for o in scene['objects']}; command=commands(task)[0]
    ticks=[{'t':i/30,'q':q,'q_target':q,'poses':poses,'grip_closed':False,
            'command':command,'phase':'test_hold'} for i in range(4)]
    write_trace(tmp_path/'data/a.jsonl',{'task':task,'source':'expert','start_q':q,'deployment':profile},ticks)
    directory=train(task,tmp_path/'data','test',steps=1,device='cpu',width=16,blocks=1,batch_size=2,
                    out_dir=tmp_path/'export',eval_n=1,eval_max_ticks=1,smoke_only=True)
    policy=load(directory)
    np.testing.assert_allclose(policy.config['joint_limits'][4],profile['roll_window_rad'])
    assert policy.config['joint_limits'][4][0]<policy.config['joint_limits'][4][1]
    chunk=policy.act({'q':q,'slots':encode_slots(task,poses),'command':command})
    assert chunk.shape==(15,6) and np.isfinite(chunk).all()
