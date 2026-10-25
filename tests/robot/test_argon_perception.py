"""Offline acceptance for the Argon survey/control-loop boundary."""
from copy import deepcopy
import json
import runpy
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from r2p import gemini
from r2p.model import Arm
from r2p.perception import CameraModel, StillCamera
from r2p.robot.argon_perception import ArgonPerceiver, RIM_HEIGHT_M
from r2p.robot.config import ROOT
from r2p.robot.io import MockRobot
from r2p.robot.rehearsal import placement_task


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now


class FakeArgon:
    def __init__(self, response):
        self.response = response
        self.requests = []
        self.delay_s = 0.
        self.gate = None
        self.started = threading.Event()

    def __call__(self, parts, **kwargs):
        data = parts[0]["inline_data"]["data"]
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        response, gate, delay = deepcopy(self.response), self.gate, self.delay_s
        self.requests.append({**kwargs, "frame_mean": float(frame.mean())})
        self.started.set()
        if gate is not None:
            assert gate.wait(5), "test did not release fake Argon request"
        time.sleep(delay)
        if isinstance(response, Exception):
            raise response
        return json.dumps(response), {"served_by": "fake-argon"}


@pytest.fixture(autouse=True)
def no_live_requests(monkeypatch):
    monkeypatch.setattr(gemini, "call", lambda *a, **k: pytest.fail("unmocked Gemini request"))


@pytest.fixture
def cam():
    return CameraModel.look_at(image_size=(640, 360))


@pytest.fixture
def frame():
    return np.zeros((360, 640, 3), np.uint8)


@pytest.fixture(scope="module")
def arm():
    return Arm()


@pytest.fixture
def contact_q():
    # Offline FK fixture: jaw midpoint is [.1748, -.1245, .0085], gap 34.14 mm.
    return np.array([.6414889412543895, .3890547734729014, -.04469537858767749,
                     1.4143545083388755, -1.44, .2])


def caps(cam, *points):
    values = []
    for xy, on_grid in points:
        uv = cam.project([*xy, RIM_HEIGHT_M])
        values.append({"rim_center": [float(uv[1]/cam.height*1000), float(uv[0]/cam.width*1000)],
                       "on_grid": on_grid})
    return {"caps": values}


def finish_refresh(perceiver):
    perceiver._worker.join(timeout=5)
    assert not perceiver._worker.is_alive()


def test_first_estimate_seeds_nearest_supply_at_rim_height_and_fixed_cells(cam, frame, arm, contact_q):
    board = ["X", "", "", "", "", "", "", "", ""]
    task = placement_task("r2c2", cap=1, board=board)
    observed = [.178, -.126]
    fake = gemini.FakeGemini([caps(cam, ([.1274, -.1272], False), (observed, False),
                                 ([.1748, -.1245], True), ([.27, -.13], False))])
    with ArgonPerceiver(cam, task, ask=fake.call, arm=arm) as p:
        result = p.estimate(frame, contact_q)
        pose = result["poses"]["piece"]
        np.testing.assert_allclose(pose["xyz"], [*observed, .0085], atol=1e-9)
        assert pose["source"] == "vision" and pose["seen"]
        assert result["holding"] == []  # no commanded target, so no inferred grasp
        assert result["masks"]["piece"].shape == frame.shape[:2]
        assert result["masks"]["piece"].any()
        assert "other1" in result["poses"] and "other2" not in result["poses"]
        for cell, xy in task["board"]["cells"].items():
            assert result["poses"][cell]["xyz"][:2] == xy
        assert result["poses"]["cell"]["xyz"][:2] == task["board"]["cells"]["r2c2"]
        for _ in range(5):
            p.estimate(frame, contact_q)
        assert len(fake.requests) == 1
        request = fake.requests[0]
        assert request["model"] == "gemini-4-argon"
        assert request["config"]["generation_config"]["thinking_level"] == "low"
        # Caller mutation cannot corrupt the cached pose or fixed targets.
        pose["xyz"][0] = 100
        assert p.estimate(frame, contact_q)["poses"]["piece"]["xyz"][0] == pytest.approx(observed[0])


def test_30hz_estimates_with_three_second_argon_latency(cam, frame, arm, contact_q):
    clock = Clock()
    fake = FakeArgon(caps(cam, ([.1748, -.1245], False)))
    robot = MockRobot(arm=arm, q0=contact_q, sag=lambda q: q)
    with ArgonPerceiver(cam, placement_task("r1c1", cap=1), ask=fake, arm=arm,
                        commanded_q=lambda: robot.target, clock=clock) as p:
        p.setup(frame)
        fake.delay_s = 3.
        fake.started.clear()
        clock.now = 4.01
        latest = np.full_like(frame, 83)
        durations = []
        loop_started = time.perf_counter()
        for tick in range(100):
            started = time.perf_counter()
            result = p.estimate(latest, robot.joint_map.to_model(robot.read()))
            durations.append(time.perf_counter()-started)
            assert result["poses"]["piece"]["source"] == "vision"
            assert result["poses"]["piece"]["seen"]
            robot.step()
            time.sleep(max(0, (tick+1)/30-(time.perf_counter()-loop_started)))
        elapsed = time.perf_counter()-loop_started
        assert fake.started.is_set()
        finish_refresh(p)
        assert len(fake.requests) == 2
        assert fake.requests[-1]["frame_mean"] == pytest.approx(83, abs=.1)
        assert fake.requests[-1]["role"] == "main" and fake.requests[-1]["thinking"] == "low"
        assert 100/elapsed > 25
        assert np.percentile(durations, 95) < 1/25
        assert max(durations) < .1
        print(f"Argon latency 3 s: {100/elapsed:.2f} Hz, estimate p95 {1000*np.percentile(durations, 95):.2f} ms")


def test_contact_needs_squeeze_and_nearby_pads(cam, frame, arm, contact_q):
    xy = arm.tips(contact_q)[2][:2]
    fake = FakeArgon(caps(cam, (xy, False)))
    with ArgonPerceiver(cam, placement_task("r1c1", supply_xy=xy), ask=fake, arm=arm) as p:
        p.set_commanded_q(contact_q)
        assert p.estimate(frame, contact_q)["holding"] == []
        squeeze = contact_q.copy()
        squeeze[5] = 0.
        p.set_commanded_q(squeeze)
        distant = contact_q.copy()
        distant[0] += .5
        assert p.estimate(frame, distant)["holding"] == []
        wide = contact_q.copy()
        wide[5] = .4
        assert p.estimate(frame, wide)["holding"] == []
        assert p.estimate(frame, contact_q)["holding"] == ["piece"]


def test_mock_grasp_follows_measured_rigid_frame_release_freezes_until_new_vision(cam, frame, arm, contact_q):
    clock = Clock()
    xyz = arm.tips(contact_q)[2] + [.004, .002, 0]
    fake = FakeArgon(caps(cam, (xyz[:2], False)))
    robot = MockRobot(arm=arm, q0=contact_q, sag=lambda q: q, tau_s=.001)
    robot.set_grasp(.2)
    with ArgonPerceiver(cam, placement_task("r1c1", supply_xy=xyz[:2]), ask=fake, arm=arm,
                        commanded_q=lambda: robot.target, clock=clock) as p:
        p.setup(frame)
        squeeze = contact_q.copy()
        squeeze[5] = 0.
        robot.write(robot.joint_map.to_owner(squeeze))
        robot.step()
        q0 = robot.joint_map.to_model(robot.read())
        mid0 = arm.tips(q0)[2]
        R0 = arm.fk(q0).xmat[arm.m.body("gripper").id].reshape(3, 3).copy()
        first = p.estimate(frame, q0)
        assert first["holding"] == ["piece"]
        initial_xyz = np.asarray(first["poses"]["piece"]["xyz"])
        moved = squeeze.copy()
        moved[0] += .12
        moved[2] -= .18
        moved[4] += .4
        robot.write(robot.joint_map.to_owner(moved))
        robot.step()
        q1 = robot.joint_map.to_model(robot.read())
        mid1 = arm.tips(q1)[2]
        R1 = arm.fk(q1).xmat[arm.m.body("gripper").id].reshape(3, 3).copy()
        clock.now = 12
        held = p.estimate(frame, q1)
        expected = mid1 + R1 @ R0.T @ (initial_xyz-mid0)
        np.testing.assert_allclose(held["poses"]["piece"]["xyz"], expected, atol=1e-10)
        assert np.linalg.norm(expected-initial_xyz) > .01
        assert held["poses"]["piece"]["source"] == "vision+fk"
        assert held["poses"]["piece"]["seen"]
        assert len(fake.requests) == 1  # no re-read while held, even after three periods

        fake.gate = threading.Event()
        fake.started.clear()
        confirmed_xy = expected[:2] + [.002, -.001]
        fake.response = caps(cam, (xyz[:2], False), (confirmed_xy, True))
        robot.set_grasp(None)
        opened = moved.copy()
        opened[5] = .6
        robot.write(robot.joint_map.to_owner(opened))
        robot.step()
        released = p.estimate(frame, robot.joint_map.to_model(robot.read()))
        assert fake.started.wait(1)
        assert released["holding"] == []
        assert released["poses"]["piece"]["source"] == "released"
        np.testing.assert_allclose(released["poses"]["piece"]["xyz"], expected)
        opened[0] += .3
        robot.write(robot.joint_map.to_owner(opened))
        robot.step()
        frozen = p.estimate(frame, robot.joint_map.to_model(robot.read()))
        np.testing.assert_allclose(frozen["poses"]["piece"]["xyz"], expected)
        fake.gate.set()
        finish_refresh(p)
        confirmed = p.estimate(frame, robot.joint_map.to_model(robot.read()))
        np.testing.assert_allclose(confirmed["poses"]["piece"]["xyz"], [*confirmed_xy, .0085])
        assert confirmed["poses"]["piece"]["source"] == "vision"


def test_pregrasp_request_cannot_overwrite_released_pose(cam, frame, arm, contact_q):
    clock = Clock()
    xy = arm.tips(contact_q)[2][:2]
    fake = FakeArgon(caps(cam, (xy, False)))
    with ArgonPerceiver(cam, placement_task("r1c1", supply_xy=xy), ask=fake, arm=arm, clock=clock) as p:
        p.setup(frame)
        fake.response = caps(cam, (xy+[.02, 0], False))
        fake.gate = threading.Event()
        fake.started.clear()
        clock.now = 4
        p.estimate(frame, contact_q)
        assert fake.started.wait(1)
        squeeze = contact_q.copy()
        squeeze[5] = 0
        p.set_commanded_q(squeeze)
        assert p.estimate(frame, contact_q)["holding"] == ["piece"]
        moved = contact_q.copy()
        moved[0] += .2
        moved[2] -= .1
        held_xyz = p.estimate(frame, moved)["poses"]["piece"]["xyz"]
        opened = moved.copy()
        opened[5] = .6
        p.set_commanded_q(opened)
        p.estimate(frame, opened)
        fake.gate.set()
        finish_refresh(p)
        # Keep the next fresh request in flight so this assertion tests the old result.
        fake.gate = threading.Event()
        fake.response = caps(cam, (np.asarray(held_xyz)[:2], True))
        released = p.estimate(frame, opened)
        np.testing.assert_allclose(released["poses"]["piece"]["xyz"], held_xyz)
        assert released["poses"]["piece"]["source"] == "released"
        fake.gate.set()
        finish_refresh(p)
        assert p.estimate(frame, opened)["poses"]["piece"]["source"] == "vision"


@pytest.mark.parametrize("error", [gemini.QuotaError("MAIN daily quota exhausted"),
                                  gemini.CacheMiss("cache-only miss")])
def test_background_quota_and_cache_only_errors_propagate(cam, frame, arm, contact_q, error):
    clock = Clock()
    fake = FakeArgon(caps(cam, ([.1748, -.1245], False)))
    with ArgonPerceiver(cam, placement_task("r1c1", cap=1), ask=fake, arm=arm, clock=clock) as p:
        p.setup(frame)
        fake.response = error
        clock.now = 4
        p.estimate(frame, contact_q)
        finish_refresh(p)
        with pytest.raises(type(error), match=str(error)):
            p.estimate(frame, contact_q)
        with pytest.raises(type(error)):
            p.estimate(frame, contact_q)
        assert len(fake.requests) == 2


def test_missing_initial_cap_fails_and_failed_refresh_retains_pose(cam, frame, arm, contact_q):
    clock = Clock()
    fake = FakeArgon({"caps": []})
    with ArgonPerceiver(cam, placement_task("r1c1", cap=1), ask=fake, arm=arm, clock=clock) as p:
        with pytest.raises(ValueError, match="cannot see"):
            p.setup(frame)
        fake.response = caps(cam, ([.1748, -.1245], False))
        before = p.setup(frame)["poses"]["piece"]
        fake.response = {"caps": []}
        clock.now = 4
        p.estimate(frame, contact_q)
        finish_refresh(p)
        after = p.estimate(frame, contact_q)
        assert after["poses"]["piece"] == before
        assert "cannot see" in after["redetect_error"]


def test_close_does_not_wait_for_api_or_accept_late_results(cam, frame, arm, contact_q):
    clock = Clock()
    fake = FakeArgon(caps(cam, ([.1748, -.1245], False)))
    p = ArgonPerceiver(cam, placement_task("r1c1", cap=1), ask=fake, arm=arm, clock=clock)
    p.setup(frame)
    fake.gate = threading.Event()
    fake.started.clear()
    clock.now = 4
    p.estimate(frame, contact_q)
    assert fake.started.wait(1)
    started = time.perf_counter()
    p.close()
    assert time.perf_counter()-started < .2
    fake.gate.set()
    finish_refresh(p)
    assert p._pending is None
    with pytest.raises(RuntimeError, match="closed"):
        p.estimate(frame, contact_q)


def runner_fixture(monkeypatch, cam, frame):
    import r2p.perception
    import r2p.policy
    import r2p.robot.io
    import r2p.robot.rehearsal
    from r2p.policy.features import export_config
    from r2p.sim.bench import load_profile

    profile = load_profile(ROOT/"scripts/robot/K_BENCH_PROFILE.json")
    task = placement_task("r1c1", cap=1)
    config = export_config(task, "test_hold")
    config.update(start_q=profile["start_q"], obs_const=[], eval={"n": 1})
    policy = SimpleNamespace(config=config, act=lambda obs: np.tile(profile["start_q"], (15, 1)))
    monkeypatch.setattr(r2p.policy, "load", lambda path: policy)
    camera = StillCamera(frame)
    camera.closed = False
    camera.close = lambda: setattr(camera, "closed", True)
    monkeypatch.setattr(r2p.perception, "open_camera", lambda cfg: camera)
    monkeypatch.setattr(CameraModel, "load", lambda path: cam)
    monkeypatch.setattr(r2p.robot.io, "RealRobot", lambda *a, **k: pytest.fail("opened real robot"))
    monkeypatch.setattr(r2p.robot.rehearsal, "streaming_robot", lambda *a: pytest.fail("opened daemon"))
    monkeypatch.setattr(r2p.robot.rehearsal, "require_released_bus", lambda *a: pytest.fail("checked serial ownership"))
    monkeypatch.syspath_prepend(str(ROOT/"scripts/robot"))
    main = runpy.run_path(str(ROOT/"scripts/robot/run_real_policy.py"))["main"]
    return main, camera, profile


def test_cli_argon_check_selects_cell_and_surveys_without_robot(cam, frame, monkeypatch, capsys):
    from r2p import ttt
    main, camera, profile = runner_fixture(monkeypatch, cam, frame)
    board = ["X", "", "", "", "", "", "", "", ""]
    fake = gemini.FakeGemini([
        {"board": board, "move": "r1c1", "why": "Take centre"},
        caps(cam, ([.1748, -.1245], False))])
    monkeypatch.setattr(gemini, "call", fake.call)
    assert main(["--export", "fake", "--cap", "1", "--argon-perception", "--check", "--transport", "armd"]) == 0
    assert camera.closed
    output = capsys.readouterr().out
    assert '"motion": false' in output and '"source": "vision"' in output
    assert '"move": "r1c1"' in output and '"source": "gemini"' in output
    assert '"joint_pose_source": "captured_bench_profile"' in output
    assert len(fake.requests) == 2


def test_cli_live_seeds_before_session_and_binds_accepted_target(cam, frame, monkeypatch, tmp_path):
    import r2p.robot.argon_perception
    import r2p.robot.io
    from r2p.robot.config import load_joint_map

    main, camera, profile = runner_fixture(monkeypatch, cam, frame)
    fake = FakeArgon(caps(cam, ([.1748, -.1245], False)))
    monkeypatch.setattr(gemini, "call", fake)
    robot = MockRobot(q0=profile["start_q"], joint_map=load_joint_map(), sag=lambda q: q)
    events = []
    globals_ = main.__globals__
    globals_["require_released_bus"] = lambda *a: events.append("ownership")
    globals_["checked_rest"] = lambda *a: events.append("rest")

    def real_robot(cfg):
        assert len(fake.requests) == 1, "cap must be surveyed before torque priming"
        events.append("prime")
        return robot

    monkeypatch.setattr(r2p.robot.io, "RealRobot", real_robot)

    def make_session(*args, **kwargs):
        p = kwargs["perceiver"]
        assert p.poses["piece"]["source"] == "vision"
        session = SimpleNamespace(last_sent=robot.target.copy())

        def run(**kw):
            assert kw["timeout_s"] == 60
            np.testing.assert_array_equal(p.commanded_q(), session.last_sent)
            session.last_sent = session.last_sent + .01
            np.testing.assert_array_equal(p.commanded_q(), session.last_sent)
            kwargs["record"].mkdir(parents=True)
            events.append("session")
            return {"outcome": "goal_reached", "success": True}

        session.run = run
        return session

    globals_["make_session"] = make_session
    from r2p.robot.operator import Operator
    monkeypatch.setattr(Operator, "__enter__", lambda self: self)
    monkeypatch.setattr(Operator, "__exit__", lambda *a: None)
    assert main(["--export", "fake", "--cap", "1", "--cell", "r1c1", "--argon-perception",
                 "--real", "--allow-unverified", "--arm-released", "--record", str(tmp_path)]) == 0
    assert events == ["ownership", "ownership", "rest", "prime", "session"]
    assert camera.closed and robot.closed
    assert len(list(tmp_path.glob("*/decision.json"))) == 1
