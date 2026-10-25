from copy import deepcopy
import numpy as np
import pytest

from r2p.robot.io import MockRobot
from r2p.robot.runner import Session
from r2p.robot.guard import GuardResult
from r2p.robot.operator import Operator


class HoldPolicy:
    def __init__(self, q):
        self.config = {"start_q": np.asarray(q).tolist()}
        self.obs = []
        self.done = False

    def act(self, obs):
        self.obs.append(deepcopy(obs))
        return np.tile(self.config["start_q"], (15, 1))


class World:
    def __init__(self, task, good=False):
        self.task, self.good = task, good

    def update(self, q, robot):
        self.q = q.copy()

    def poses(self):
        return {"cube": {"xyz": [.25 if self.good else .15, 0, .024], "yaw_deg": 0},
                "bowl": {"xyz": [.25, 0, .015], "yaw_deg": 0}}

    def truth_check(self):
        return {"ok": self.good, "per_predicate": []}


def session(flat_arm, cube_task, good=False):
    q = np.array([0., 0, 0, 0, 0, .9])
    robot = MockRobot(arm=flat_arm, q0=q)
    policy = HoldPolicy(q)
    runner = Session(robot, policy, cube_task, arm=flat_arm, world=World(cube_task, good),
                     gravity_comp=False, realtime=False, operator=Operator(None, keyboard=False))
    return runner, robot, policy


def test_goal_requires_full_second_open_and_elevated(flat_arm, cube_task):
    runner, robot, _ = session(flat_arm, cube_task, good=True)
    result = runner.run(timeout_s=3, prepare=False)
    assert result["outcome"] == "goal_reached"
    assert result["ticks"] == 30
    assert result["success"] and robot.torque_enabled
    runner, robot, _ = session(flat_arm, cube_task, good=True)
    flat_arm.tips = lambda q: (np.zeros(3), np.zeros(3), np.array([.25, 0, .06]))
    assert runner.run(timeout_s=1.2, prepare=False)["outcome"] == "timeout"


def test_sixty_refusals_resend_last_goal(flat_arm, cube_task):
    runner, robot, _ = session(flat_arm, cube_task)
    initial = robot.target.copy()
    runner.guard.project = lambda current, target, **kw: GuardResult(current.copy(), False, reason="test collision")
    result = runner.run(timeout_s=3, prepare=False)
    assert result["outcome"] == "held_after_refusals"
    assert result["refusals"] == 60
    assert np.allclose(robot.target, initial) and robot.torque_enabled


def test_load_stop_and_policy_done(flat_arm, cube_task):
    runner, robot, policy = session(flat_arm, cube_task)
    robot.load_values["shoulder_pan"] = -501
    assert runner.run(prepare=False)["outcome"] == "load_stop"
    assert robot.torque_enabled
    runner, robot, policy = session(flat_arm, cube_task)
    policy.done = True
    assert runner.run(prepare=False)["outcome"] == "policy_done"


def test_policy_observes_command_not_sagging_measurement(flat_arm, cube_task):
    runner, robot, policy = session(flat_arm, cube_task)
    runner.run(timeout_s=.6, prepare=False)
    assert len(policy.obs) == 18
    assert all(np.allclose(obs["q"], policy.config["start_q"]) for obs in policy.obs)
    assert not np.allclose(robot.q, policy.obs[-1]["q"])


def test_start_from_export_and_operator_override(flat_arm, cube_task):
    runner, robot, policy = session(flat_arm, cube_task)
    runner.start_q[0] = .1
    policy.config["start_q"][0] = .1
    assert runner.move_start()
    assert robot.target[0] == pytest.approx(.1, abs=.005)
    original = runner.operator.poll
    called = [False]
    def poll():
        if not called[0]:
            called[0] = True
            return [runner.operator.feed("yes")]
        return original()
    runner.operator.poll = poll
    result = runner.run(timeout_s=.1, prepare=False)
    assert result["operator_override"] is True and result["success"]
    assert not result["truth_check"]["ok"]


def test_error_keeps_torque_and_holds(flat_arm, cube_task):
    runner, robot, policy = session(flat_arm, cube_task)
    policy.act = lambda obs: np.full((15,6), np.nan)
    result = runner.run(prepare=False)
    assert result["outcome"] == "error"
    assert "finite" in result["error"]
    assert robot.torque_enabled
    assert np.allclose(robot.target, robot.q)


def test_recorded_session_has_frames_clean_labels_and_commands(flat_arm, cube_task, tmp_path):
    import json
    runner, robot, policy = session(flat_arm, cube_task, good=True)
    class Camera:
        def read(self):
            return np.zeros((16,24,3),np.uint8)
    runner.camera = Camera()
    runner.record = tmp_path/"attempt"
    result = runner.run(timeout_s=2, prepare=False)
    rows = [json.loads(line) for line in (runner.record/"trace.jsonl").read_text().splitlines()]
    assert result["outcome"] == "goal_reached"
    assert rows[0]["meta"]["success"]
    assert rows[0]["meta"]["commands"] == [[0,1,"pick(cube)"]]
    for row in rows[1:]:
        assert (runner.record/row["frame"]).is_file()
        assert len(row["q_target"]) == 6
        assert row["q_target"] == row["q_exec"]  # compensation disabled in this fixture


def test_synthbench_stub_with_sag_and_compensation(cube_task):
    from r2p.model import Arm
    from r2p.robot.dryrun import ScriptedPolicy, make_world
    arm = Arm()
    policy = ScriptedPolicy(cube_task, arm)
    robot = MockRobot(arm=arm, q0=policy.start_q)
    world, camera, _ = make_world(cube_task, arm, 0)
    try:
        result = Session(robot, policy, cube_task, arm=arm, world=world, realtime=False).run()
        assert result["outcome"] == "goal_reached", result
        assert result["truth_check"]["ok"]
        assert result["perception_check"]["ok"]
        assert result["clipped_fraction"] < .05
        assert result["commanded_q_resets"] == 0
    finally:
        if camera:
            camera.close()


def test_truth_fallback_stub_with_rigid_grasp(cube_task):
    from r2p.model import Arm
    from r2p.robot.dryrun import ScriptedPolicy, TruthWorld
    arm = Arm()
    policy = ScriptedPolicy(cube_task, arm)
    robot = MockRobot(arm=arm, q0=policy.start_q)
    world = TruthWorld(cube_task, arm, 0)
    result = Session(robot, policy, cube_task, arm=arm, world=world, realtime=False).run()
    assert result["outcome"] == "goal_reached", result
    assert result["truth_check"]["ok"] and result["success"]
    assert world.held is None
