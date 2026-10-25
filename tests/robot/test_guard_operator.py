import json
import numpy as np
import pytest

from r2p.robot.guard import Guard
from r2p.robot.operator import Operator
from r2p.robot.policy import Ensembler, check_chunk


def test_guard_speed_slow_zone_and_joint_margin(flat_arm):
    guard = Guard(flat_arm)
    q = np.zeros(6)
    wanted = np.ones(6)*10
    result = guard.project(q, wanted)
    assert result.accepted and result.clipped
    assert np.max(result.target) <= np.deg2rad(20)/30 + 1e-10
    q[1] = -.08
    slow = guard.project(q, wanted)
    assert np.max(slow.target-q) <= np.deg2rad(8)/30 + 1e-10
    assert guard.limits[0, 1] == pytest.approx(3-np.deg2rad(2))


def test_guard_refusal_holds_and_retreat_improves(flat_arm):
    guard = Guard(flat_arm)
    q = np.zeros(6)
    q[1] = -.099
    down = q.copy()
    down[1] -= .1
    result = guard.project(q, down)
    assert not result.accepted
    assert np.array_equal(result.target, q)
    q[1] = -.12  # already penetrating: retreat may progressively recover
    up = q.copy()
    up[1] += .1
    result = guard.project(q, up, retreat=True)
    assert result.accepted
    assert result.target[1] > q[1]


def test_guard_checks_all_geometry_and_roll(flat_arm):
    guard = Guard(flat_arm, {"roll_excursion_deg": 1})
    q = np.zeros(6)
    q[4] = np.deg2rad(1)
    target = q.copy()
    target[4] += .1
    assert guard.project(q, target).reason == "roll_excursion"
    flat_arm.geometry = lambda q: {"link": .005-q[0], "gripper": .1, "finger": .1, "finger_r": .2}
    q = np.zeros(6)
    target = q.copy()
    target[0] = 1
    assert not guard.project(q, target).accepted
    assert not guard.project(q, [float("nan")]*6).accepted


def test_full_path_catches_middle_collision(flat_arm):
    flat_arm.geometry = lambda q: {"link": .2, "gripper": .1, "finger": -.1 if .04 < q[0] < .06 else .1, "finger_r": .2}
    guard = Guard(flat_arm)
    assert not guard.path_clear(np.zeros(6), [.1, 0, 0, 0, 0, 0])


def test_operator_sequenced_file_and_two_torque_requests(tmp_path):
    path = tmp_path / "control.json"
    path.write_text(json.dumps({"seq": 4, "cmd": "stop"}))
    now = [10.]
    op = Operator(path, keyboard=False, clock=lambda: now[0])
    assert op.poll() == []  # stale command is ignored
    for seq, command, expected in [(5, "p", "pause"), (6, "g", "go"), (7, "y", "yes"), (8, "T", "torque_off_armed"), (9, "T", "torque_off")]:
        path.write_text(json.dumps({"seq": seq, "cmd": command}))
        assert op.poll() == [expected]
        assert op.poll() == []
    assert op.score is True
    assert op.feed("T") == "torque_off_armed"
    now[0] += 3.01
    assert op.feed("T") == "torque_off_armed"
    path.write_text('{"seq":')
    assert op.poll() == []


def test_act_oldest_gets_highest_weight_and_zero_is_valid():
    ensemble = Ensembler(.1)
    ensemble.add(0, np.zeros((15, 6)))
    ensemble.add(1, np.ones((15, 6)))
    expected = np.exp(-.1)/(1+np.exp(-.1))
    assert np.allclose(ensemble.target(1), expected)
    assert np.allclose(ensemble.target(15), 1)
    with pytest.raises(ValueError, match="covers"):
        ensemble.target(16)
    with pytest.raises(ValueError, match="15, 6"):
        check_chunk(np.zeros((6, 15)))
