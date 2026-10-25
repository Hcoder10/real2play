from pathlib import Path
import numpy as np
import pytest

from r2p.tasks import (check, command_onehot, commands, encode_obs, encode_slots,
                       load_task, poses_from_scene, sample_scene, validate)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("path", sorted((ROOT / "tasks").glob("*/task.json")))
def test_task_smoke(path):
    task = load_task(path)
    assert not validate(task)
    scene = sample_scene(task, np.random.default_rng(7))
    assert scene == sample_scene(task, np.random.default_rng(7))
    poses = poses_from_scene(scene)
    k = task["limits"]["max_objects"]
    assert encode_slots(task, poses).shape == (k, 9)
    assert encode_obs(task, [0]*6, poses).shape == (6 + 9*k,)
    for command in commands(task):
        assert command_onehot(task, command).sum() == 1
    assert isinstance(check(task, {"objects": poses, "gripper_open": True})["ok"], bool)


def test_cube_goal():
    task = load_task("cube_in_bowl")
    state = {"objects": {"cube": {"xyz": [0.23, 0, 0.024]}, "bowl": {"xyz": [0.23, 0, 0.015]}},
             "gripper_open": True}
    assert check(task, state)["ok"]
    state["gripper_open"] = False
    assert not check(task, state)["ok"]
