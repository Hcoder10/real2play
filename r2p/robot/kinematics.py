"""Damped least-squares IK for the jaw midpoint, with fingers down."""
import numpy as np
from .io import vector


def solve_ik(arm, xyz, q0, *, grip=None, tolerance=0.003, fingers_down=True):
    q = vector(q0)
    xyz = np.asarray(xyz, float)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError("xyz must be three finite spec-frame metres")
    if grip is not None:
        q[5] = float(grip)
    limits = np.asarray(arm.range) + np.deg2rad([2, -2])
    for _ in range(180):
        mid = arm.tips(q)[2]
        jp, jr, axis = arm.jac_tip(q)
        weight = 0.07 if fingers_down else 0.0
        err = np.r_[xyz - mid, weight * (np.array([0., 0., -1.]) - axis)]
        if np.linalg.norm(xyz - mid) < tolerance and (not fingers_down or axis[2] < -0.98):
            return q
        # d(axis)/dq = angular_jacobian x axis
        ja = np.cross(jr.T, axis).T
        jac = np.vstack([jp[:, :4], weight * ja[:, :4]])
        dq = jac.T @ np.linalg.solve(jac @ jac.T + 2e-5 * np.eye(6), err)
        q[:4] = np.clip(q[:4] + np.clip(dq, -0.15, 0.15), limits[:4, 0], limits[:4, 1])
    raise ValueError(f"unreachable fingers-down target {xyz.tolist()} (error {np.linalg.norm(xyz-arm.tips(q)[2]):.4f} m)")
