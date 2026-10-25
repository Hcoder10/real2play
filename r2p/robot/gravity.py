"""Gravity torque is in model radians/Nm; only the three pitch joints sag."""
import numpy as np


class GravitySag:
    def __init__(self, arm, k=None):
        from .config import sag_gain
        self.arm = arm
        self.k = sag_gain() if k is None else float(k)
        if not np.isfinite(self.k) or self.k < 0:
            raise ValueError("sag gain must be finite and nonnegative")

    def offset(self, q):
        result = np.zeros(6)
        result[1:4] = -self.k * np.asarray(self.arm.gravity_torque(q))[1:4]
        return result

    def apply(self, q):
        return np.asarray(q, float) + self.offset(q)

    __call__ = apply


class GravityComp(GravitySag):
    def offset(self, q):
        return np.clip(-super().offset(q), -np.deg2rad(3), np.deg2rad(3))
