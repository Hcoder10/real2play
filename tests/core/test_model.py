import numpy as np

from r2p.frames import HOME_Q
from r2p.model import Arm, JointMap


def test_home_pose_geometry():
    arm = Arm()
    _, _, mid = arm.tips(HOME_Q)
    g = arm.geometry(HOME_Q)
    assert 0.10 < g["finger"] < 0.14          # fingers ~12 cm above the table at the training home pose
    assert 0.2 < mid[0] < 0.35 and abs(mid[1]) < 0.01


def test_jacobian_matches_finite_difference():
    arm = Arm()
    q = HOME_Q.copy()
    jp, _, axis = arm.jac_tip(q)
    _, _, m0 = arm.tips(q)
    for j in range(5):
        dq = q.copy(); dq[j] += 1e-5
        _, _, m1 = arm.tips(dq)
        assert np.allclose((m1 - m0) / 1e-5, jp[:, j], atol=2e-3)
    assert abs(np.linalg.norm(axis) - 1) < 1e-6


def test_jointmap_roundtrip_and_gravity():
    jm = JointMap(offset=np.array([0, -11.5, 0, 0, -76.0]))
    o = np.array([3.0, -40, 50, 60, 10, 42])
    assert np.allclose(jm.to_owner(jm.to_model(o)), o)
    tau = Arm().gravity_torque(HOME_Q)
    assert tau.shape == (6,) and abs(tau[1]) > 0.05     # the shoulder lift carries the arm's weight
