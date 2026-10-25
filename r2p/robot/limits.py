"""Calibrated owner limits, independent of the MJCF's nominal joint ranges."""
from dataclasses import dataclass

import numpy as np

from r2p.frames import JOINTS
from .io import vector


@dataclass
class OwnerLimits:
    lo: np.ndarray
    hi: np.ndarray

    def __post_init__(self):
        self.lo, self.hi = vector(self.lo), vector(self.hi)
        if np.any(self.lo >= self.hi):
            raise ValueError("owner lo must be below hi for every joint")

    @classmethod
    def from_state(cls, state):
        return cls(state["lo"], state["hi"])

    def model_bounds(self, joint_map, model_range=None, margin_deg=0):
        if not np.isfinite(margin_deg) or margin_deg < 0:
            raise ValueError("joint margin must be finite and nonnegative")
        bounds = np.sort(np.stack([joint_map.to_model(self.lo), joint_map.to_model(self.hi)], axis=1), axis=1)
        if model_range is not None:
            bounds[:, 0] = np.maximum(bounds[:, 0], np.asarray(model_range)[:, 0])
            bounds[:, 1] = np.minimum(bounds[:, 1], np.asarray(model_range)[:, 1])
        bounds += np.deg2rad(float(margin_deg)) * np.array([1, -1])
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("owner/MJCF ranges have no usable intersection")
        return bounds

    def violations(self, owner, tolerance=1e-8):
        q = np.asarray(owner, float)
        if q.shape[-1:] != (6,) or not np.isfinite(q).all():
            raise ValueError("owner targets must have six finite values")
        bad = (q < self.lo-tolerance) | (q > self.hi+tolerance)
        return bad

    def require(self, owner):
        q = vector(owner)
        bad = self.violations(q)
        if np.any(bad):
            names = ", ".join(j for j, hit in zip(JOINTS, bad) if hit)
            raise ValueError(f"target outside calibrated owner limits: {names}")
        return q


class LimitedRobot:
    """Last-line owner-space rejection; never silently clip an actuator command."""
    def __init__(self, robot, limits):
        self.robot, self.owner_limits = robot, limits

    def __getattr__(self, name):
        return getattr(self.robot, name)

    def write(self, owner):
        return self.robot.write(self.owner_limits.require(owner))

    def hold(self):
        # A measured encoder can be fractionally outside a calibrated bound.
        # Session checks geometry/rate before moves; hold stays at that boundary.
        return self.write(np.clip(self.robot.read(), self.owner_limits.lo, self.owner_limits.hi))
