import json
from pathlib import Path
import runpy
import socket
from types import SimpleNamespace

import numpy as np
import pytest


def script(monkeypatch):
    folder = Path(__file__).resolve().parents[2] / "scripts/argon"
    monkeypatch.syspath_prepend(str(folder))
    return runpy.run_path(str(folder / "tictactoe.py"))


def test_all_nine_measured_cells_have_safe_no_motion_plans(tmp_path, monkeypatch):
    entry = script(monkeypatch)
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("no-motion check used network"))
    output = tmp_path / "plan.json"
    assert entry["main"](["--check", "--check-out", str(output)]) == 0
    report = json.loads(output.read_text())
    assert not report["motion"] and report["safe"]
    from r2p.robot.config import load_joint_map
    from r2p.tasks import load_task
    assert report["joint_map"] == load_joint_map().name
    piece = next(o for o in load_task("place_on_cell")["objects"] if o["role"] == "manipulated")
    assert len(report["plans"]) == 9
    for row in report["plans"]:
        assert row["supply_xyz"] == [*piece.get("xyz", piece.get("xy"))[:2], .007]
        assert len(row["waypoints"]) == 8
        assert all(w["path_clear"] and w["within_limits"] for w in row["waypoints"])


def test_daemon_owner_blocks_direct_robot_open(monkeypatch):
    entry = script(monkeypatch)
    closed = []
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: SimpleNamespace(close=lambda: closed.append(True)))
    cfg = {"daemon_url": "http://127.0.0.1:8793"}
    with pytest.raises(ValueError, match="still owns"):
        entry["require_released_bus"](cfg, True)
    assert closed == [True]
    with pytest.raises(ValueError, match="arm-released"):
        entry["require_released_bus"](cfg, False)


def test_windows_nested_bus_config_and_distinct_pitch_gains():
    from r2p.robot.config import load_config, sag_gain
    from r2p.robot.gravity import GravitySag, GravityComp
    cfg = load_config(platform="win32")
    assert cfg["port"] == "COM3" and cfg["robot_id"] == "follower_1"
    arm = SimpleNamespace(gravity_torque=lambda q: np.ones(6))
    gains = sag_gain(cfg)
    assert gains[1] != gains[2]
    assert np.allclose(GravitySag(arm, gains).offset(np.zeros(6)), -gains)
    assert np.allclose(GravityComp(arm, gains).offset(np.zeros(6)), np.minimum(gains, np.deg2rad(3)))


def test_servo_export_is_not_double_compensated(flat_arm, cube_task):
    from r2p.robot.io import MockRobot
    from r2p.robot.runner import Session
    from r2p.robot.operator import Operator
    q = np.array([0., 0, 0, 0, 0, .9])
    policy = SimpleNamespace(config={"start_q": q.tolist(), "gravity_compensation": "included_in_expert_targets"},
                             act=lambda obs: np.tile(q, (15, 1)), done=True)
    robot = MockRobot(arm=flat_arm, q0=q)
    world = SimpleNamespace(update=lambda *a: None, poses=lambda: {"cube": {"xyz": [.15, 0, .019]}, "bowl": {"xyz": [.25, 0, .015]}},
                            truth_check=lambda: {"ok": False})
    result = Session(robot, policy, cube_task, arm=flat_arm, world=world, realtime=False,
                     operator=Operator(None, keyboard=False)).run(prepare=False)
    assert result["outcome"] == "policy_done"
    assert np.allclose(robot.target, q)


def test_near_open_pads_do_not_count_as_a_synthetic_grasp(cube_task):
    from r2p.model import Arm
    from r2p.frames import HOME_Q
    from r2p.robot.dryrun import make_world
    from r2p.robot.kinematics import solve_ik
    from r2p.robot.io import MockRobot
    arm = Arm()
    world, camera, _ = make_world(cube_task, arm, 0)
    try:
        pos = np.asarray(world.poses()["cube"]["xyz"])
        q = solve_ik(arm, pos + [0, 0, .006], HOME_Q, grip=.34)
        robot = MockRobot(arm=arm, q0=q)
        # This crosses the rendering fixture's closing threshold within 25 mm,
        # but the 45 mm gap has not contacted the 38 mm cube.
        assert np.linalg.norm(arm.tips(q)[2] - pos) < .025
        world.update(q, robot)
        assert world.world.held is None
        q[5] = .26
        world.update(q, robot)
        assert world.world.held is not None
        # Pad separation must release a held piece before the old fixed .35
        # angle would drag it along with the moving midpoint.
        q[5] = .34
        world.update(q, robot)
        assert world.world.held is None
    finally:
        camera.close()


def test_scripted_chunks_honor_requested_speed_and_compensation(cube_task):
    from r2p.model import Arm
    from r2p.robot.dryrun import ScriptedPolicy
    from r2p.tasks import encode_slots, sample_scene
    arm = Arm()
    policy = ScriptedPolicy(cube_task, arm, speed=.3, gravity_comp=False)
    scene = sample_scene(cube_task, np.random.default_rng(0))
    poses = {o["id"]: o for o in scene["objects"]}
    chunk = policy.act({"q": policy.start_q, "slots": encode_slots(cube_task, poses)})
    steps = np.diff(np.vstack([policy.start_q, chunk]), axis=0)
    assert np.max(np.abs(steps)) <= np.deg2rad(20*.3*.65)/30 + 1e-10
    assert np.all(policy.comp.offset(policy.start_q) == 0)


def test_exported_policy_advances_command_after_observed_pick(flat_arm, cube_task):
    from r2p.robot.io import MockRobot
    from r2p.robot.operator import Operator
    from r2p.robot.runner import Session
    from r2p.tasks import commands
    q = np.array([0., 0, 0, 0, 0, .1])
    observed = []
    ticks = [0]

    def act(obs):
        observed.append(obs["command"])
        return np.tile(q, (15, 1))

    def update(*args):
        ticks[0] += 1

    world = SimpleNamespace(update=update, truth_check=lambda: {"ok": False},
        poses=lambda: {"cube": {"xyz": [.15, 0, .08 if ticks[0] else .019]},
                       "bowl": {"xyz": [.25, 0, .015]}})
    policy = SimpleNamespace(config={"start_q": q.tolist(), "gravity_compensation": "included_in_expert_targets"}, act=act)
    result = Session(MockRobot(arm=flat_arm, q0=q), policy, cube_task,
                     arm=flat_arm, world=world, realtime=False,
                     operator=Operator(None, keyboard=False)).run(timeout_s=.4, prepare=False)
    vocab = commands(cube_task)
    assert result["error"] is None
    assert observed[0] == vocab[0]
    assert observed[-1] == vocab[1]
