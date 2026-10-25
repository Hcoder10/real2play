"""Damped jaw-midpoint inverse kinematics, with fingers-down in the null space."""
from __future__ import annotations

import numpy as np

from r2p.frames import HOME_Q
from r2p.model import Arm


class JawIK:
    def __init__(self, arm=None):
        self.arm = arm or Arm()

    def solve(self, xyz, q=None, grip=0.3, yaw=None, iterations=60):
        q = np.array(HOME_Q if q is None else q, float).copy()
        q[5] = grip
        target = np.asarray(xyz, float)
        for _ in range(iterations):
            a, b, mid = self.arm.tips(q)
            J, R, axis = self.arm.jac_tip(q)
            err = target - mid
            inv = J.T @ np.linalg.solve(J @ J.T + 0.000015 * np.eye(3), np.eye(3))
            # Tilt the finger axis just enough to keep both actual fingertips level.
            gap = b-a
            across = gap / max(np.linalg.norm(gap), 1e-8)
            az = np.arctan2(across[1], across[0]) if yaw is None else yaw
            desired_across = np.array([np.cos(az), np.sin(az), 0.0])
            rot_err = np.cross(across, desired_across)
            # A down-facing finger, projected perpendicular to the jaw axis.
            flat_axis = axis - across * np.dot(axis, across)
            flat_axis /= max(np.linalg.norm(flat_axis), 1e-8)
            rot_err += np.cross(flat_axis, [0, 0, -1])
            orient = R.T @ np.linalg.solve(R @ R.T + .01*np.eye(3), rot_err)
            if yaw is not None:
                # The cross-product error is zero at an upside-down yaw and can
                # strand the wrist against its limit after a long carry. Resolve
                # that branch using the SO-101's down-facing roll convention.
                roll = yaw+q[0]+.06
                candidates = [roll+turn*2*np.pi for turn in (-1, 0, 1)
                              if self.arm.range[4, 0] < roll+turn*2*np.pi < self.arm.range[4, 1]]
                if candidates:
                    roll = min(candidates, key=lambda value: abs(value-q[4]))
                    orient[4] += .6*np.clip(roll-q[4], -.8, .8)
            delta = inv @ np.clip(err, -.025, .025) + (np.eye(5)-inv@J) @ (0.6*orient)
            q[:5] += np.clip(delta, -.15, .15)
            q = np.clip(q, self.arm.range[:, 0]+1e-5, self.arm.range[:, 1]-1e-5)
            if np.linalg.norm(err) < .00015 and np.linalg.norm(delta) < .001:
                break
        return q

    def compensate(self, q, physics):
        from r2p.frames import JOINTS
        kp = np.array([physics["joints"][j]["kp"] for j in JOINTS])
        target = q + self.arm.gravity_torque(q) / kp
        target[5] = q[5]
        return np.clip(target, *self.arm.range.T)
