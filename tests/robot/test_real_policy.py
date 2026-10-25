from copy import deepcopy
import json
from pathlib import Path
import runpy
import socket
from types import SimpleNamespace

import numpy as np
import pytest

from r2p.frames import HOME_Q
from r2p.model import Arm, JointMap
from r2p.policy.features import export_config
from r2p.robot.config import ROOT, load_config, load_joint_map
from r2p.robot.io import MockRobot
from r2p.robot.limits import LimitedRobot, OwnerLimits
from r2p.robot.operator import Operator
from r2p.robot.rehearsal import (SUPPLY_XY, audit, make_session, placement_task, rehearse,
                                require_released_bus, validate_profile)
from r2p.sim.bench import load_profile
from r2p.tasks import encode_slots, sample_scene


@pytest.fixture
def profile():
    return load_profile(ROOT/"scripts/robot/K_BENCH_PROFILE.json")


def hold_policy(task, start):
    config = export_config(task, "test_hold")
    config.update(start_q=list(start), obs_const=[], eval={"n": 0})
    return SimpleNamespace(config=config, act=lambda obs: np.tile(start, (15, 1)))


def test_real_map_ranges_signs_and_new_gripper(profile):
    jm = load_joint_map()
    limits = validate_profile(profile, jm)
    bounds = limits.model_bounds(jm)
    assert np.rad2deg(bounds[4, 1]) == pytest.approx(14.6)
    assert np.rad2deg(bounds[3, 1]) == pytest.approx(82.87)
    assert jm.to_owner(HOME_Q)[3] > limits.hi[3]
    assert jm.to_owner([0, 0, 0, 0, np.pi/2, 0])[4] > limits.hi[4]
    assert jm.to_owner([0, 0, 0, 0, profile["roll_rad"], profile["gripper_open_rad"]])[4:] == pytest.approx([81.9, 20])
    assert jm.to_owner([0, 0, 0, 0, 0, profile["gripper_close_rad"]])[5] == pytest.approx(6)
    flipped = JointMap(sign=-np.ones(5))
    transformed = limits.model_bounds(flipped)
    assert transformed[0, 0] < transformed[0, 1]
    assert limits.violations(flipped.to_owner(transformed[:, 0])).sum() == 0
    assert limits.violations(flipped.to_owner(transformed[:, 1])).sum() == 0


def test_stale_profile_is_rejected(profile):
    jm = deepcopy(load_joint_map())
    jm.go += .01
    with pytest.raises(ValueError, match="stale"):
        validate_profile(profile, jm)


def test_every_cell_and_supply_are_current_and_not_randomized():
    from r2p.tasks import load_task
    base = load_task("place_on_cell")
    for cell, xy in base["board"]["cells"].items():
        for cap, supply in enumerate(SUPPLY_XY):
            task = placement_task(cell, cap)
            for seed in (0, 19):
                poses = {o["id"]: o for o in sample_scene(task, np.random.default_rng(seed))["objects"]}
                assert poses["cell"]["xyz"][:2] == xy
                assert poses["piece"]["xyz"][:2] == supply
                for name, point in base["board"]["cells"].items():
                    assert poses[name]["xyz"][:2] == point
                assert not any(name.startswith("other") for name in poses)


def test_invalid_export_start_never_sends_a_command(profile):
    task = placement_task("r1c1")
    result = rehearse(hold_policy(task, HOME_Q), task, profile, timeout_s=.1)
    assert result["outcome"] == "start_refused"
    assert result["sent_commands"] == 0 and result["sent_owner_limit_hits"] == 0
    assert result["audit"]["start_limit_joints"] == ["wrist_flex"]
    assert result["placement_error_mm"] > 100
    assert not result["success"]


def test_policy_targets_and_compensation_stay_inside_owner_envelope(profile):
    task, jm, arm = placement_task("r1c1"), load_joint_map(), Arm()
    start = np.asarray(profile["start_q"])
    policy = hold_policy(task, start)
    target = start.copy()
    target[3], target[4] = np.deg2rad(95), np.pi/2
    policy.act = lambda obs: np.tile(target, (15, 1))
    robot = MockRobot(arm=arm, joint_map=jm, q0=start, sag=lambda q: q)
    sent = []
    original = robot.write
    robot.write = lambda owner: (sent.append(owner.copy()), original(owner))[1]
    poses = {o["id"]: o for o in sample_scene(task, np.random.default_rng(0))["objects"]}
    world = SimpleNamespace(update=lambda *a: None, poses=lambda: poses, truth_check=lambda: {"ok": False})
    session = make_session(robot, policy, task, profile, arm=arm, world=world, realtime=False,
                           operator=Operator(None, keyboard=False))
    result = session.run(timeout_s=.3, prepare=False)
    assert result["requested_limit_ticks"] == result["ticks"] > 0
    assert result["requested_limit_joints"][3:5] == [result["ticks"]]*2
    assert not np.any(OwnerLimits.from_state(profile["owner_state"]).violations(np.array(sent)))


def test_last_line_write_rejects_not_clips(profile):
    sent = []
    limits = OwnerLimits.from_state(profile["owner_state"])
    robot = LimitedRobot(SimpleNamespace(write=lambda q: sent.append(q)), limits)
    with pytest.raises(ValueError, match="wrist_roll"):
        robot.write([0, 0, 0, 0, 200, 20])
    assert not sent


def test_mock_latency_reads_are_pure_and_hold_clears_old_commands():
    robot = MockRobot(q0=HOME_Q, sag=lambda q: q, latency_ticks=2, tau_s=.001)
    changed = HOME_Q.copy()
    changed[0] += .1
    robot.write(robot.joint_map.to_owner(changed))
    for _ in range(20):
        assert robot.read()[0] == pytest.approx(np.rad2deg(HOME_Q[0]))
    robot.step()
    robot.step()
    assert robot.q[0] == pytest.approx(HOME_Q[0])
    robot.hold()
    for _ in range(4):
        robot.step()
    assert robot.q[0] == pytest.approx(HOME_Q[0])


def test_direct_bus_never_opens_when_daemon_ownership_is_uncertain(monkeypatch):
    cfg = load_config()
    with pytest.raises(ValueError, match="arm-released"):
        require_released_bus(cfg, False)
    monkeypatch.setattr(socket, "create_connection", lambda *a, **kw: SimpleNamespace(close=lambda: None))
    with pytest.raises(ValueError, match="still owns"):
        require_released_bus(cfg, True)
    def timeout(*a, **kw):
        raise TimeoutError()
    monkeypatch.setattr(socket, "create_connection", timeout)
    with pytest.raises(ValueError, match="cannot verify"):
        require_released_bus(cfg, True)


def test_streaming_handshake_is_get_only_and_rejects_legacy_or_wrong_map(monkeypatch):
    from r2p.robot.rehearsal import streaming_robot, validate_stream_state
    import r2p.robot.daemon
    profile = load_profile(ROOT/"scripts/robot/K_STREAM_PROFILE.json")
    state = deepcopy(profile["streaming_state"])
    # A synthetic settled handover, independent of the captured live daemon's
    # known out-of-owner-range target. No hardware is opened or commanded.
    state["q"] = profile["start_q"].copy()
    state["q_target"] = state["q"].copy()
    state["owner_deg"] = load_joint_map().to_owner(state["q"]).tolist()
    profile["owner_state"]["q"] = state["owner_deg"].copy()
    gets = []
    class FakeDaemon:
        def __init__(self, *args):
            pass
        def request(self, command=None):
            assert command is None, "handshake must not POST"
            gets.append(True)
            return state
    monkeypatch.setattr(r2p.robot.daemon, "DaemonRobot", FakeDaemon)
    assert isinstance(streaming_robot(load_config(), profile, load_joint_map()), FakeDaemon)
    assert gets == [True]
    previous_target = state["q_target"].copy()
    state["q_target"][0] += .2
    with pytest.raises(ValueError, match="not settled"):
        streaming_robot(load_config(), profile, load_joint_map())
    state["q_target"] = previous_target.copy()
    state["q_target"][3] = np.deg2rad(95)
    with pytest.raises(ValueError, match="owner limits"):
        streaming_robot(load_config(), profile, load_joint_map())
    state["q_target"] = previous_target.copy()
    with pytest.raises(ValueError, match="legacy"):
        validate_stream_state(profile["owner_state"], load_joint_map())
    state["q"][4] += 1
    with pytest.raises(ValueError, match="joint map"):
        streaming_robot(load_config(), profile, load_joint_map())
    state["owner_deg"][0] += 10
    state["q"] = load_joint_map().to_model(state["owner_deg"]).tolist()
    state["q_target"] = state["q"].copy()
    with pytest.raises(ValueError, match="pose changed"):
        streaming_robot(load_config(), profile, load_joint_map())


def test_captured_streaming_target_is_refused_before_any_policy_write():
    profile = load_profile(ROOT/"scripts/robot/K_STREAM_PROFILE.json")
    task = placement_task("r1c1")
    result = rehearse(hold_policy(task, profile["start_q"]), task, profile, timeout_s=.1, transport="armd")
    assert result["outcome"] == "handover_refused"
    assert result["sent_commands"] == 0
    assert "owner limits" in result["error"]


def test_stale_calibration_refuses_before_opening_serial(profile, tmp_path, monkeypatch):
    import r2p.robot.io
    from r2p.frames import JOINTS
    cfg = load_config()
    cfg.update(calibration_dir=str(tmp_path), robot_id="fake")
    (tmp_path/"fake.json").write_text(json.dumps({j: {"range_min": 0, "range_max": 4095} for j in JOINTS}))
    monkeypatch.setattr(r2p.robot.io, "ReadOnlyBus", lambda *a, **kw: pytest.fail("opened serial on stale calibration"))
    monkeypatch.syspath_prepend(str(ROOT/"scripts/robot"))
    checked = runpy.run_path(str(ROOT/"scripts/robot/run_real_policy.py"))["checked_rest"]
    with pytest.raises(ValueError, match="owner limits differ"):
        checked(cfg, profile, load_joint_map())


def test_cli_check_never_opens_network_or_hardware(profile, tmp_path, monkeypatch):
    import r2p.policy
    import r2p.robot.io
    task = placement_task("r1c1")
    monkeypatch.setattr(r2p.policy, "load", lambda path: hold_policy(task, HOME_Q))
    monkeypatch.setattr(socket, "create_connection", lambda *a, **kw: pytest.fail("network in no-motion check"))
    monkeypatch.setattr(r2p.robot.io, "RealRobot", lambda *a, **kw: pytest.fail("hardware in no-motion check"))
    monkeypatch.syspath_prepend(str(ROOT/"scripts/robot"))
    main = runpy.run_path(str(ROOT/"scripts/robot/run_real_policy.py"))["main"]
    assert main(["--export", "fake", "--check", "--cell", "r1c1"]) == 1
    monkeypatch.setattr(r2p.policy, "load", lambda path: hold_policy(task, profile["start_q"]))
    assert main(["--export", "fake", "--check", "--cell", "r1c1"]) == 0


def test_twin_preserves_true_folded_rest_without_mjcf_clipping(profile):
    from r2p.sim.twin import TaskTwin
    jm = load_joint_map()
    limits = OwnerLimits.from_state(profile["owner_state"])
    rest = jm.to_model(profile["owner_state"]["rest"])
    task = placement_task("r1c1")
    scene = sample_scene(task, np.random.default_rng(0))
    with TaskTwin(scene, q0=rest, joint_limits=limits.model_bounds(jm)) as twin:
        np.testing.assert_allclose(twin.q(), rest)
        assert rest[3] < Arm().range[3, 0]


def test_live_perception_requires_closing_pad_contact_and_releases_on_separation(profile):
    from r2p.robot.policy_vision import ContactPerceiver
    task, arm = placement_task("r1c1"), Arm()
    q = np.asarray(profile["start_q"])
    q[5] = .2
    pose = {"xyz": arm.tips(q)[2].tolist(), "seen": True, "source": "vision"}
    decisions = []
    p = SimpleNamespace(_held=None, _was_closed=False, poses={"piece": pose},
                        objects={o["id"]: o for o in task["objects"]})
    p.estimate = lambda *a: (decisions.append(p._was_closed), {"poses": p.poses})[1]
    adapter = ContactPerceiver(p, task, {})
    opened = q.copy()
    opened[5] = profile["gripper_open_rad"]
    adapter.estimate(None, opened)
    assert decisions[-1]  # no attachment merely because an object is nearby
    adapter.estimate(None, q)
    assert not decisions[-1]  # measured closing contact admits the FK latch
    p._held = ("piece", np.zeros(3), np.eye(3))
    adapter.estimate(None, opened)
    assert p._held is None


def test_saved_setup_can_change_cell_but_never_object_shape(tmp_path):
    from r2p.perception import CameraModel, TaskPerceiver
    from r2p.robot.policy_vision import load_placement_setup
    task = placement_task("r0c0", board=["", "X", "", "", "", "", "", "", ""])
    cam = CameraModel.look_at(image_size=(64, 48))
    p = TaskPerceiver(cam, task, detector=lambda *a, **kw: {})
    p.setup(np.zeros((48, 64, 3), np.uint8))
    path = tmp_path/"appearance.json"
    p.save_setup(path)
    changed = placement_task("r2c2")
    result = load_placement_setup(cam, changed, path)
    assert result.fixed["cell"]["xyz"][:2] == changed["board"]["cells"]["r2c2"]
    assert "other1" not in result.perceiver.poses
    changed["objects"][0]["size_m"][0] += .003
    with pytest.raises(ValueError, match="different objects"):
        load_placement_setup(cam, changed, path)
