from copy import deepcopy

import numpy as np
import pytest

from r2p.policy import load
from r2p.policy.data import TraceDataset
from r2p.policy.features import export_config
from r2p.robot.config import ROOT, load_joint_map
from r2p.robot.limits import OwnerLimits
from r2p.robot.rehearsal import placement_task
from r2p.sim.bench import load_profile
from r2p.tasks import commands, encode_slots, sample_scene
from r2p.traces import write_trace


def fixture_trace(path, profile):
    task = placement_task("r1c1")
    scene = sample_scene(task, np.random.default_rng(0))
    poses = {o["id"]: o for o in scene["objects"]}
    q = np.asarray(profile["start_q"])
    command = commands(task)[0]
    meta = {"task": task, "source": "expert", "start_q": q, "deployment": profile}
    ticks = [{"t": i/30, "q": q, "q_target": q, "poses": poses, "grip_closed": False,
              "command": command, "phase": "fixture_hold"} for i in range(4)]
    write_trace(path, meta, ticks)
    return task, poses


def test_profiles_cannot_be_mixed_in_training(tmp_path):
    profile = load_profile(ROOT/"scripts/robot/K_BENCH_PROFILE.json")
    task, _ = fixture_trace(tmp_path/"a.jsonl", profile)
    changed = deepcopy(profile)
    changed["roll_rad"] += .1
    fixture_trace(tmp_path/"b.jsonl", changed)
    with pytest.raises(ValueError, match="deployment conventions disagree"):
        TraceDataset(task, tmp_path, export_config(task, "mix"))


def test_cpu_export_preserves_real_convention_and_bounded_absolute_outputs(tmp_path):
    from r2p.policy.train import train
    profile = load_profile(ROOT/"scripts/robot/K_BENCH_PROFILE.json")
    task, poses = fixture_trace(tmp_path/"traces/a.jsonl", profile)
    directory = train(task, tmp_path/"traces", "bench_fixture", steps=1, device="cpu", width=16, blocks=1,
                      batch_size=2, out_dir=tmp_path/"export", eval_n=1, eval_max_ticks=1, smoke_only=True)
    policy = load(directory)
    assert policy.config["deployment"] == profile
    assert policy.config["gripper_open_threshold_rad"] == profile["gripper_open_threshold_rad"]
    limits, jm = OwnerLimits.from_state(profile["owner_state"]), load_joint_map()
    for command in commands(task):
        chunk = policy.act({"q": profile["start_q"], "slots": encode_slots(task, poses), "command": command})
        assert chunk.shape == (15, 6)
        assert not limits.violations(jm.to_owner(chunk)).any()
        np.testing.assert_allclose(chunk[:, 4], profile["roll_rad"], atol=1e-10)
