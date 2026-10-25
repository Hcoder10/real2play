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
    assert report["joint_map"] == "vcal_oct25"
    assert len(report["plans"]) == 9
    for row in report["plans"]:
        assert row["supply_xyz"] == [.13, .13, .007]
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
