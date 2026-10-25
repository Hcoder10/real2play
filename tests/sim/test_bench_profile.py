import numpy as np
import pytest

from r2p.model import Arm
from r2p.robot.config import ROOT, load_joint_map
from r2p.robot.limits import OwnerLimits
from r2p.robot.rehearsal import placement_task
from r2p.sim.bench import load_profile
from r2p.sim.expert import Expert
from r2p.sim.ik import JawIK
from r2p.sim.twin import TaskTwin
from r2p.tasks import sample_scene


def test_locked_roll_ik_uses_real_bounds_and_never_takes_the_opposite_branch():
    profile = load_profile(ROOT/"scripts/robot/K_BENCH_PROFILE.json")
    arm = Arm()
    arm.range = np.asarray(profile["model_limits"])
    solver = JawIK(arm, roll=profile["roll_rad"])
    for yaw in (-np.pi, 0, np.pi):
        q = solver.solve([.17, 0, .05], profile["start_q"], grip=profile["gripper_open_rad"], yaw=yaw, iterations=100)
        assert q[4] == profile["roll_rad"]
        assert np.all(q >= arm.range[:, 0]) and np.all(q <= arm.range[:, 1])


def test_profile_expert_caps_ignore_yaw_labels_map_into_real_owner_limits():
    profile = load_profile(ROOT/"scripts/robot/K_BENCH_PROFILE.json")
    task = placement_task("r1c1")
    expert = Expert(task, deployment=profile)
    scene = sample_scene(task, np.random.default_rng(0))
    with TaskTwin(scene, q0=profile["start_q"]) as twin:
        expert._bind(twin)
        assert expert._grasp_yaw("piece") is None
        for _ in range(3):
            expert._servo([.20, -.10, .05], expert.release_grip)
        home = expert.ik.compensate(expert.start_q, twin.physics)
        home[4] += .1  # labels must retain the declared roll even at home
        expert._tick(home)
        jm = load_joint_map()
        limits = OwnerLimits.from_state(profile["owner_state"])
        for row in expert.ticks:
            assert row["q_target"][4] == profile["roll_rad"]
            assert not limits.violations(jm.to_owner(row["q_target"])).any()
        assert expert.release_grip == pytest.approx(.4038)
        assert expert.close_grip == pytest.approx(.00214)


def test_locked_roll_profile_cannot_silently_train_rectangular_grasps():
    profile = load_profile(ROOT/"scripts/robot/K_BENCH_PROFILE.json")
    with pytest.raises(ValueError, match="round caps"):
        Expert("cube_in_bowl", deployment=profile)
