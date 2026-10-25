import json
import subprocess
import sys
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from r2p.robot.daemon import ArmController, DaemonRobot, make_server
from r2p.robot.io import MockRobot
from r2p.robot.gravity import GravitySag
from r2p.robot.record import TraceWriter
from r2p.robot.calibration import fit_sag, fit_touch, stable_pose
from r2p.robot.config import ROOT, load_config
from r2p.robot.preflight import Report, check_bus


def controller(flat_arm):
    robot = MockRobot(arm=flat_arm, sag=GravitySag(flat_arm, 0), q0=[0,0,0,0,0,.9])
    return ArmController(robot, arm=flat_arm, realtime=False), robot


def test_daemon_interpolates_and_releases_only_at_rest(flat_arm):
    ctrl, robot = controller(flat_arm)
    writes = []
    original = robot.write
    def write(owner):
        writes.append(robot.joint_map.to_model(owner))
        original(owner)
    robot.write = write
    destination = np.array([.25, 0, 0, 0, 0, .9])
    ctrl.command({"joints": destination})
    assert len(writes) > 20
    assert np.max(np.abs(np.diff(np.array(writes), axis=0))) <= np.deg2rad(20)/30 + 1e-10
    with pytest.raises(ValueError, match="rest"):
        ctrl.command({"release": True})
    assert robot.torque_enabled
    ctrl.command({"rest": True})
    for _ in range(20):
        robot.step()
    ctrl.command({"release": True})
    assert not robot.torque_enabled
    with pytest.raises(RuntimeError, match="torque is off"):
        ctrl.command({"joints": destination})


def test_daemon_load_lag_and_invalid_command_abort(flat_arm):
    ctrl, robot = controller(flat_arm)
    robot.load_values["elbow_flex"] = 600
    with pytest.raises(RuntimeError, match="load"):
        ctrl.command({"gripper": .8})
    assert robot.torque_enabled and np.allclose(robot.q, robot.target)
    robot.load_values["elbow_flex"] = 0
    robot.q[0] = -.5
    with pytest.raises(RuntimeError, match="lag"):
        ctrl.command({"gripper": .8})
    with pytest.raises(ValueError, match="exactly one"):
        ctrl.command({"joints": np.zeros(6), "xyz": [0,0,0]})
    with pytest.raises(ValueError, match="finite"):
        ctrl.command({"joints": [float("nan")]*6})


def test_daemon_http_mock_round_trip(flat_arm):
    ctrl, robot = controller(flat_arm)
    server = make_server(ctrl, ("127.0.0.1", 0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = DaemonRobot(f"http://127.0.0.1:{server.server_port}", robot.joint_map)
    try:
        state = client.request()
        assert state["at_rest"] and state["torque_enabled"]
        client.write(robot.joint_map.to_owner([.05,0,0,0,0,.9]))
        assert client.read().shape == (6,)
        assert len(client.loads()) == 6
        client.hold()
        request = Request(client.url+"/cmd", data=b'{"release":true}', headers={"Origin":"https://example.com"})
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert robot.torque_enabled


def test_trace_rows_and_frames_pair_even_when_queue_full(tmp_path, monkeypatch):
    import cv2
    entered, release = threading.Event(), threading.Event()
    original = cv2.imwrite
    def blocked_write(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)
    monkeypatch.setattr(cv2, "imwrite", blocked_write)
    writer = TraceWriter(tmp_path / "attempt", {"task":"cube_in_bowl", "source":"real"}, capacity=1)
    frame = np.zeros((12,20,3), np.uint8)
    tick = {"t":0, "q":np.zeros(6), "q_target":np.ones(6), "poses":{}, "grip_closed":False, "command":"home", "phase":"home"}
    assert writer.submit(tick, frame)
    assert entered.wait(5)
    assert writer.submit({**tick,"t":1/30}, frame)
    assert not writer.submit({**tick,"t":2/30}, frame)
    release.set()
    writer.close({"success":True,"outcome":"goal_reached"})
    lines = [json.loads(x) for x in (writer.directory/"trace.jsonl").read_text().splitlines()]
    assert lines[0]["meta"]["schema"] == "r2p_trace/1"
    assert lines[0]["meta"]["dropped"] == 1
    assert len(lines) == 3
    assert {x["frame"] for x in lines[1:]} == {"frames/0000.jpg","frames/0001.jpg"}
    assert len(list((writer.directory/"frames").glob("*.jpg"))) == 2
    assert np.array_equal(lines[1]["q_target"], np.ones(6))


def test_sag_fit_known_gain_and_touch_identifiability(flat_arm):
    samples = [{"commanded":np.array([v,0,0,0,0,.9]), "measured":np.array([v,0,0,0,0,.9])+GravitySag(flat_arm,.07).offset(np.zeros(6))} for v in (.1,.2,.3)]
    assert fit_sag(samples, flat_arm)["k"] == pytest.approx(.07)
    with pytest.raises(ValueError, match="six"):
        from r2p.model import JointMap
        fit_touch([], flat_arm, JointMap())
    assert np.allclose(stable_pose(lambda:np.ones(6), sleep=lambda _:None), 1)


def test_touch_fit_recovers_known_joint_offsets():
    from r2p.model import Arm, JointMap
    from r2p.frames import HOME_Q
    arm = Arm()
    truth = JointMap(offset=np.array([1., -2, 3, -4, 5]))
    rng = np.random.default_rng(4)
    qs = np.clip(HOME_Q+rng.uniform(-.4,.4,(12,6)), arm.range[:,0]+.04, arm.range[:,1]-.04)
    samples = [{"owner_deg":truth.to_owner(q), "xyz":arm.tips(q)[0]} for q in qs]
    fitted = fit_touch(samples, arm, JointMap())
    assert fitted["usable"] and not fitted["verified"]
    assert fitted["rank"] == 5
    assert np.allclose(fitted["offset_deg"], truth.offset, atol=.001)
    assert fitted["rms_m"] < 1e-6


def test_read_only_preflight_register_checks(follower):
    from r2p.frames import JOINTS
    from r2p.robot.io import ReadOnlyBus
    values = {"Homing_Offset":10, "Min_Position_Limit":0, "Max_Position_Limit":4095,
              "Present_Voltage":120, "Present_Temperature":30, "Torque_Enable":0, "Status":0}
    follower.bus.sync_read = lambda name, normalize=True: dict.fromkeys(JOINTS, values[name])
    bus = ReadOnlyBus({}, follower=follower)
    calibration = {j:{"homing_offset":10,"range_min":0,"range_max":4095} for j in JOINTS}
    report = Report()
    check_bus(bus, calibration, report)
    assert not report.failed
    values["Present_Voltage"] = 100
    values["Status"] = 1
    values["Homing_Offset"] = 11
    failed = Report()
    check_bus(bus, calibration, failed)
    assert sum(status == "FAIL" for status,_,_ in failed.lines) == 3
    assert follower.bus.events == [("connect",)]
    bus.close()


def test_scripts_import_and_calibration_smoke(tmp_path):
    for name in ("armd", "preflight", "fit_sag", "run_policy", "record_session", "mirror", "read_pose"):
        process = subprocess.run([sys.executable, str(ROOT/"scripts/robot"/f"{name}.py"), "--help"],
                                 capture_output=True, text=True, timeout=20)
        assert process.returncode == 0, process.stderr
    output = tmp_path/"sag.json"
    process = subprocess.run([sys.executable, str(ROOT/"scripts/robot/fit_sag.py"), "--simulate", ".05", "--out", str(output)],
                             capture_output=True, text=True, timeout=20)
    assert process.returncode == 0, process.stderr
    result = json.loads(output.read_text())
    assert result["k"] == pytest.approx(.05)
    assert result["n"] == 8 and result["simulated"]
