"""Gravity torque is in model radians/Nm; only the three pitch joints sag."""
import numpy as np


class GravitySag:
    def __init__(self, arm, k=None):
        from .config import sag_gain
        self.arm = arm
        self.k = np.asarray(sag_gain() if k is None else k, dtype=float)
        if self.k.shape not in ((), (6,)) or not np.isfinite(self.k).all() or np.any(self.k < 0):
            raise ValueError("sag gain must be finite and nonnegative")

    def offset(self, q):
        result = np.zeros(6)
        gain = self.k if self.k.ndim == 0 else self.k[1:4]
        result[1:4] = -gain * np.asarray(self.arm.gravity_torque(q))[1:4]
        return result

    def apply(self, q):
        return np.asarray(q, float) + self.offset(q)

    __call__ = apply


class GravityComp(GravitySag):
    def offset(self, q):
        return np.clip(-super().offset(q), -np.deg2rad(3), np.deg2rad(3))
