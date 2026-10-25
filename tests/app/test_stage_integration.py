"""Exercise the real orchestration adapters while replacing expensive component work.

No Gemini, hardware, GPU, torch training, or MuJoCo rollout is executed.
"""
import copy
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest

from r2p.pipeline.engine import Pipeline, RunOptions
from r2p.pipeline.storage import atomic_json, digest, read_json
from r2p.tasks import commands, load_task, sample_scene
from r2p.traces import write_trace


def module(monkeypatch, name, **members):
    value = ModuleType(name)
    value.__dict__.update(members)
    monkeypatch.setitem(sys.modules, name, value)
    return value


def episode(directory, task, *, source="expert", seed=0, frames=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    scene = sample_scene(task, np.random.default_rng(seed))
    poses = {o["id"]: {"xyz":o["xyz"],"yaw_deg":o["yaw_deg"]} for o in scene["objects"]}
    meta = {"schema":"r2p_trace/1","task":task,"task_name":task["name"],"source":source,
            "player":"fixture","level":task["name"],"seed":seed,"scene":scene,"success":True,
            "fail_reason":None,"commands":[[0,3/30,commands(task)[0]]],"rig":"so101_bench"}
    ticks = [{"t":i/30,"q":[0,-.3,.4,-.1,0,.5],"q_target":[0,-.3,.4,-.1,0,.5],
              "poses":copy.deepcopy(poses),"grip_closed":False,"command":commands(task)[0],"phase":"pick"}
             for i in range(3)]
    path = directory / f"episode_{seed}.jsonl"
    write_trace(path,meta,ticks)
    if frames:
        from PIL import Image
        (directory / "frames").mkdir(exist_ok=True)
        Image.new("RGB",(16,16)).save(directory / "frames/0000.jpg")
        atomic_json(directory / "camera.json",{"fixture":True})
    return path


def test_two_round_adapters_then_cache_resume(tmp_path, monkeypatch):
    import r2p.labeling  # Load the real deterministic labeler before replacing its expensive parent component.
    from r2p.pipeline import stages
    from r2p.pipeline import evaluation
    task = load_task("cube_in_bowl")
    atomic_json(tmp_path / "tasks/cube_in_bowl/task.json",task)
    episode(tmp_path / "input",task,frames=True)
    atomic_json(tmp_path / "config/physics/so101_bench.json",{"object":{"mass_kg":.03,"friction":.6},"latency_ticks":1})
    script = tmp_path / "scripts/robot/run_policy.py"
    script.parent.mkdir(parents=True)
    script.write_text("# test placeholder")
    incumbent = tmp_path / "exports/incumbent"
    incumbent.mkdir(parents=True)
    atomic_json(incumbent / "config.json",{"name":"incumbent"})
    calls = {"twins":[],"fits":[],"commands":[],"data":[],"evals":[]}

    def twin(source, *, task, out_dir, iterations):
        calls["twins"].append(str(source))
        scene = sample_scene(task,np.random.default_rng(7))
        atomic_json(Path(out_dir)/"run.json",{"schema":"r2s_live/1","status":"done","session_kind":"bench","iters":[]})
        return {"scene":scene}
    module(monkeypatch,"r2p.real2sim",run_twin=twin)

    gate = {"passed":False,"metrics":{"jaw_rms_mm":3},"thresholds":{"jaw_rms_mm":2},"failures":["jaw_rms_mm"]}
    def fit(attempts, physics, *, heldout, max_nfev):
        calls["fits"].append((attempts,heldout))
        return {"physics":physics,"gate":gate}
    module(monkeypatch,"r2p.sysid",validation_gate=lambda attempts,physics:gate,fit_parameters=fit,
           plan_experiments=lambda residuals,q0:[{"execute":False,"reason":"fixture"}])

    def generate(task,n,out_dir,workers,*,seed=0):
        calls["data"].append((n,seed))
        return [episode(out_dir,task,seed=seed+i) for i in range(n)]
    module(monkeypatch,"r2p.sim",gen_traces=generate)
    module(monkeypatch,"r2p.synthbench",World=lambda:None)
    module(monkeypatch,"r2p.policy.train")
    monkeypatch.setattr(stages,"validate_lease",lambda *a,**kw:{"LANDLORD_LEASE_ID":"fixture-lease"})

    def command(self, argv, folder, extra_env=None, **kwargs):
        calls["commands"].append(argv)
        if "--name" in argv:
            directory = tmp_path / "exports" / argv[argv.index("--name")+1]
            directory.mkdir(parents=True)
            atomic_json(directory/"config.json",{"name":directory.name,"task":task["name"],"start_q":[0]*6,
                                                 "obs_layout":"fixture","obs_const":[],"eval":{},"action":"absolute_q"})
            atomic_json(directory/"norm.json",{})
            (directory/"weights.npz").write_bytes(b"fixture-only")
            (directory/"runtime.py").write_text("# fixture-only")
        else:
            out = Path(argv[argv.index("--record")+1]) / "stamp/attempt_01"
            episode(out,task,source="bot",seed=len(calls["commands"]),frames=True)
    monkeypatch.setattr(stages.ComponentStages,"command",command)

    def paired(challenger,incumbent,task,plan):
        calls["evals"].append(plan)
        return tuple({"policy":str(policy),"plan_hash":digest(plan),"results":{
            noise:[{"scene_id":row["id"],"noise_seed":row["noise_seed"],"success":index==0}
                   for row in plan["episodes"]] for noise in plan["noise"]}}
            for index,policy in enumerate([challenger,incumbent]))
    monkeypatch.setattr(evaluation,"evaluate_pair",paired)
    monkeypatch.setattr("urllib.request.urlopen",lambda *a,**kw:pytest.fail("unexpected network access"))

    options = RunOptions(task="cube_in_bowl",root=tmp_path,video="input",mode="dry-run",rounds=2,traces=2,
                         eval_n=3,incumbent=str(incumbent))
    pipeline = Pipeline(options,emit=lambda _:None)
    assert pipeline.run(), read_json(pipeline.path)
    assert len(calls["twins"])==2 and "deploy" in calls["twins"][1]
    assert len(calls["fits"])==1 and len(calls["evals"])==2
    assert calls["evals"][0]["noise"]==["clean","real"]
    assert pipeline.result("r2","train")["incumbent"]==pipeline.result("r1","gate")["selected"]
    assert read_json(pipeline.result("r1","twin")["live"])["session_kind"]=="synthetic"
    before=copy.deepcopy(calls)
    assert Pipeline(options,emit=lambda _:None).run()
    assert calls==before


def test_instruction_uses_video_source_and_owned_inference_output(tmp_path, monkeypatch):
    task=load_task("cube_in_bowl")
    called={}
    def infer(source,instruction=None,*,out_root,fit_twin=True):
        called.update(source=source,instruction=instruction,out_root=out_root,fit_twin=fit_twin)
        return {"task":task}
    module(monkeypatch,"r2p.real2sim",task_from_video=infer)
    pipeline=Pipeline(RunOptions(task="put the cube in the bowl",video="video.mp4",root=tmp_path))
    folder=tmp_path/"runs/instruction/r1/session"
    result=pipeline.adapters.load_task(folder)
    assert result==task and called["source"]==tmp_path/"video.mp4"
    assert called["out_root"].is_relative_to(folder) and called["fit_twin"] is False
