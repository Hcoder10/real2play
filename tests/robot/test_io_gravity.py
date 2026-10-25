import gc
import numpy as np
import pytest

from r2p.frames import HOME_Q
from r2p.robot.io import RealRobot, ReadOnlyBus, MockRobot, load_magnitude, BusLease
from r2p.robot.gravity import GravitySag, GravityComp
from r2p.robot.config import load_config, load_joint_map
from r2p.robot.runner import observation_q


def test_priming_order_and_torque_preserving_disconnect(follower):
    robot = RealRobot({}, follower=follower)
    events = follower.bus.events
    assert events[:5] == [("connect",), ("torque_off",), ("read", "Present_Position", False),
                          ("write", "Goal_Position", follower.bus.present, False), ("configure_torque_on",)]
    robot.hold()
    robot.close()
    assert events.count(("torque_off",)) == 1
    assert events[-1] == ("disconnect", False)
    assert not follower.config.disable_torque_on_disconnect


def test_only_explicit_close_disables_torque(follower):
    robot = RealRobot({}, follower=follower)
    robot.close(torque_off=True)
    assert follower.bus.events[-2:] == [("torque_off",), ("disconnect", False)]


def test_calibration_mismatch_never_configures_or_toggles(follower):
    follower.bus.is_calibrated = False
    with pytest.raises(RuntimeError, match="EEPROM"):
        RealRobot({}, follower=follower)
    assert follower.bus.events == [("connect",), ("disconnect", False)]


def test_read_only_no_writes_and_retains_follower(follower):
    bus = ReadOnlyBus({}, follower=follower)
    del follower
    gc.collect()
    assert len(bus.read()) == 6
    assert set(bus.loads().values()) == {32}
    with pytest.raises(ValueError, match="never writes"):
        bus.close(torque_off=True)
    bus.close()
    assert all(e[0] in ("connect", "read", "disconnect") for e in bus.bus.events)
    assert bus.bus.events[-1] == ("disconnect", False)


@pytest.mark.parametrize("value, expected", [(-32, 32), (32, 32), (-1023, 1023), (0, 0)])
def test_signed_load_magnitude(value, expected):
    assert load_magnitude(value) == expected


def test_missing_load_telemetry_is_not_a_safe_zero():
    with pytest.raises(ValueError, match="finite"):
        load_magnitude(float("nan"))


def test_bus_has_one_owner(tmp_path):
    first = BusLease(str(tmp_path / "fake-port"))
    try:
        with pytest.raises(RuntimeError, match="already owned"):
            BusLease(str(tmp_path / "fake-port"))
    finally:
        first.close()
    BusLease(str(tmp_path / "fake-port")).close()


def test_gravity_sign_compensation_and_cap(flat_arm):
    q = np.array([0, 0, 0, 0, 0, .9])
    sag = GravitySag(flat_arm, .05)
    comp = GravityComp(flat_arm, .05)
    assert np.allclose(sag(comp(q)), q)
    assert sag(q)[1] < q[1]
    huge = GravityComp(flat_arm, 100).offset(q)
    assert np.max(np.abs(huge)) == pytest.approx(np.deg2rad(3))
    assert np.all(huge[[0, 4, 5]] == 0)


def test_servo_and_grip_floor(flat_arm):
    q0 = np.array([0., 0, 0, 0, 0, .9])
    robot = MockRobot(arm=flat_arm, q0=q0)
    target = q0.copy()
    target[0], target[5] = .3, -.12
    robot.write(robot.joint_map.to_owner(target))
    robot.set_grasp(.3)
    first = robot.step()
    assert 0 < first[0] < .3
    for _ in range(100):
        robot.step()
    assert robot.q[5] == .3
    robot.close()
    assert robot.torque_enabled


def test_commanded_observation_prevents_measured_delta_sag_integration(flat_arm):
    start = np.array([0., 0, 0, 0, 0, .9])
    stable = MockRobot(arm=flat_arm, q0=start)
    failing = MockRobot(arm=flat_arm, q0=start)
    commanded = start.copy()
    for _ in range(180):
        obs, reset = observation_q(commanded, stable.q)
        assert not reset
        # A zero-delta residual policy returns its observation as absolute target.
        stable.write(stable.joint_map.to_owner(obs))
        stable.step()
        failing.write(failing.read())
        failing.step()
    assert abs(stable.q[1]) < .03
    assert failing.q[1] < -.8


def test_divergence_resets_whole_vector():
    commanded = np.arange(6)*.1
    measured = commanded.copy()
    measured[2] += np.deg2rad(4.1)
    result, reset = observation_q(commanded, measured)
    assert reset and np.array_equal(result, measured)


def test_platform_overlay_and_unverified_candidates():
    cfg = load_config(platform="darwin")
    assert cfg["port"] == "/dev/tty.usbmodem5B3D0496341"
    assert cfg["camera"]["width"] == 640
    jm = load_joint_map(cfg)
    assert not jm.verified
    # The bench's active calibration is fitted independently of the platform.
    from r2p.model import JointMap
    from r2p.robot.config import config_path
    active = JointMap.load(config_path(cfg["joint_map"]))
    assert np.allclose(jm.offset, active.offset)
    assert np.allclose(jm.to_model(jm.to_owner(HOME_Q)), HOME_Q)
