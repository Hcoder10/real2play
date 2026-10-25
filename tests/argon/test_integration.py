"""Small CPU integrations against the actual shared modules as they land."""
import copy
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from r2p.gemini import FakeGemini
from r2p.real2sim import SessionData, run_twin
from r2p.sysid import fit_sag, replay_attempt


def test_detect_fit_render_replay_with_shared_cpu_twin(tmp_path):
    from r2p.frames import HOME_Q
    from r2p.perception import CameraModel
    from r2p.synthbench import World

    task = {"schema": "argon_task/1", "name": "integration_pick", "instruction": "put the cube in the bowl",
            "objects": [{"id": "cube", "shape": "box", "role": "manipulated", "size_m": [.03]*3,
                         "xyz": [.22, -.08, .015], "color": "red", "mass_kg": .02, "friction": .6},
                        {"id": "bowl", "shape": "bowl", "role": "container", "size_m": [.10, .04],
                         "xyz": [.28, .07, .02], "color": "blue", "fixed": True}],
            "goal": [{"pred": "inside", "a": "cube", "b": "bowl"}],
            "skeleton": [{"skill": "pick", "obj": "cube"}, {"skill": "place_in", "obj": "cube", "target": "bowl"},
                         {"skill": "release"}, {"skill": "home"}], "randomize": {"min_sep_m": 0},
            "limits": {"max_objects": 2, "episode_s": 10}}
    camera = CameraModel.look_at(image_size=(320, 240))
    world = World(task, np.random.default_rng(0), camera)
    frames, rows, masks = [], [], {}
    try:
        image = world.render(board=False)
        boxes = []
        for obj in task["objects"]:
            y, x = np.where(world.last_masks[obj["id"]])
            assert len(x) > 20
            boxes.append({"id": obj["id"], "box": [(x.min()-3)/320, (y.min()-3)/240, (x.max()+4)/320, (y.max()+4)/240]})
        for i in range(4):
            path = tmp_path / f"frame_{i}.png"
            cv2.imwrite(str(path), image)
            frames.append(path)
            masks[i] = {oid: mask.copy() for oid, mask in world.last_masks.items() if oid != "arm"}
            rows.append({"t": i / 30, "q": HOME_Q.tolist(), "q_target": HOME_Q.tolist(), "poses": world.poses(),
                         "grip_closed": False, "phase": "home", "command": "home"})
    finally:
        world.close()
    session = SessionData(tmp_path, {"session_kind": "synthetic"}, rows, frames, [i / 30 for i in range(4)], camera, masks=masks)
    fake = FakeGemini([{"objects": boxes}, {"objects": [], "why": "Looks consistent", "done": True}])
    result = run_twin(session, task=task, out_dir=tmp_path / "live", ask=fake, iterations=1)
    assert result["stop_reason"] == "critic_done"
    assert [req["model"] for req in fake.requests] == ["gemini-3.8-flash", "gemini-4-argon"]
    assert result["metrics"]["objects"]["cube"]["iou"] > .65
    assert result["metrics"]["physics"]["replayed"]
    assert cv2.imread(str(tmp_path / "live/iter_0_physics.jpg")).shape == (480, 1280, 3)


def test_static_gravity_fit_recovers_compensation_k():
    class Arm:
        def gravity_torque(self, q):
            return np.array([0, np.cos(q[1]) * .3, np.cos(q[2]) * .2, np.cos(q[3]) * .1, 0, 0])
    arm = Arm()
    truth = np.array([0, .04, .06, .08, 0, 0])
    attempts = []
    for angle in (.1, .5, .8):
        q = np.array([0, angle, angle / 2, -angle, 0, 0])
        target = q + truth * arm.gravity_torque(q)
        attempts.append({"ticks": [{"t": i/30, "q": q.tolist(), "q_target": target.tolist()} for i in range(45)]})
    result = fit_sag(attempts, arm=arm, initial=[0]*6)
    assert result["identified"]
    assert np.allclose(result["sag_gain"], truth)


def test_recorded_attempt_replays_with_real_tasktwin_cpu():
    from r2p.frames import HOME_Q
    from r2p.sim.twin import load_physics

    attempt = {"scene": {"objects": [{"id": "cube", "role": "manipulated", "shape": "box", "size_m": [.03]*3,
                                       "xyz": [.25, .1, .015], "color": "red", "mass_kg": .03, "friction": .2}]},
               "ticks": [{"t": i/30, "q": HOME_Q.tolist(), "q_target": HOME_Q.tolist()} for i in range(5)]}
    initial = copy.deepcopy(attempt)
    result = replay_attempt(attempt, load_physics())
    assert result["q"].shape == (5, 6) and np.isfinite(result["jaw_xyz"]).all()
    assert "cube" in result["end_poses"] and not result["grasp_outcome"]
    assert attempt == initial


def test_live_game_routes_each_turn_through_session_without_hardware(tmp_path, monkeypatch):
    from r2p import ttt
    from r2p.perception import CameraModel
    import r2p.perception
    import r2p.robot
    import r2p.robot.config
    import r2p.robot.daemon
    import r2p.robot.policy

    script_dir = Path(__file__).resolve().parents[2] / "scripts/argon"
    monkeypatch.syspath_prepend(str(script_dir))
    namespace = runpy.run_path(str(script_dir / "tictactoe.py"))
    calibration = tmp_path / "camera.json"
    CameraModel.look_at().save(calibration)
    calls, closed = [], []
    camera = SimpleNamespace(read=lambda: np.zeros((480, 640, 3), np.uint8), close=lambda: closed.append("camera"))
    robot = SimpleNamespace(close=lambda **kwargs: closed.append(("robot", kwargs)))
    monkeypatch.setattr(r2p.robot.config, "load_config", lambda path: {"camera": {"calibration": str(calibration)},
                                                                     "daemon_url": "unused", "control_file": "unused"})
    monkeypatch.setattr(r2p.robot.config, "load_joint_map", lambda cfg: SimpleNamespace(verified=True))
    monkeypatch.setattr(r2p.perception, "open_camera", lambda cfg: camera)
    monkeypatch.setattr(r2p.robot.daemon, "DaemonRobot", lambda *args: robot)
    monkeypatch.setattr(r2p.robot.policy, "load_policy", lambda name: SimpleNamespace(config={"task": "place_on_cell", "start_q": [0]*6}))
    class Operator:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    class Perceiver:
        def __init__(self, *args, **kwargs):
            pass
        def setup(self, frame):
            pass
        def close(self):
            pass
    class Session:
        def __init__(self, robot_arg, policy, task, **kwargs):
            assert robot_arg is robot
            calls.append((task, kwargs))
        def run(self, **kwargs):
            return {"success": True, "outcome": "goal_reached"}
    monkeypatch.setattr(r2p.robot, "Operator", Operator)
    monkeypatch.setattr(r2p.robot, "Session", Session)
    monkeypatch.setattr(r2p.perception, "TaskPerceiver", Perceiver)
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    turns = iter([(0, 4), (1, 2), (3, 6)])
    def decide(frame, camera, history, *args):
        human, move = next(turns)
        board = ttt.history_board(history)
        board[human] = "X"
        return {"board": board, "move": ttt.CELL_IDS[move], "why": "test reasoning", "model": "fake", "source": "oracle"}
    monkeypatch.setattr(ttt, "decide", decide)
    args = SimpleNamespace(config=None, camera=str(calibration), policy="fake_export", games=1,
                           robot_desc="brown cubes", human_desc="blue discs", supply_xyz=[.18, -.12, .016],
                           piece_size=.032, record=str(tmp_path / "record"), status=str(tmp_path / "status.json"), timeout=45)
    assert namespace["live_games"](args) == 0
    assert len(calls) == 3 and closed == ["camera", ("robot", {"torque_off": False})]
    for task, options in calls:
        piece = next(obj for obj in task["objects"] if obj["role"] == "manipulated")
        assert piece["xyz"] == args.supply_xyz
        assert (options["record"] / "camera.json").exists()
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["result"] == "O" and status["argon_says"] == "test reasoning"
