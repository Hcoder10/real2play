"""Small, inspectable safety projection in model radians and the spec frame."""
from dataclasses import dataclass
import numpy as np
from .io import vector


@dataclass
class GuardResult:
    target: np.ndarray
    accepted: bool
    clipped: bool = False
    reason: str = ""


class Guard:
    def __init__(self, arm, cfg=None, *, hz=30, speed=1.0, start_q=None, joint_limits=None):
        from .config import load_config
        self.arm = arm
        self.cfg = {**load_config()["guard"], **(cfg or {})}
        self.hz = float(hz)
        self.speed = float(speed)
        if not 0 < self.speed <= 1:
            raise ValueError("speed must be in (0, 1]")
        margin = np.deg2rad(self.cfg["margin_deg"])
        bounds = arm.range if joint_limits is None else joint_limits
        self.limits = np.asarray(bounds) + np.array([margin, -margin])
        if self.limits.shape != (6, 2) or not np.isfinite(self.limits).all() or np.any(self.limits[:, 0] >= self.limits[:, 1]):
            raise ValueError("invalid guard joint limits")
        self.roll_origin = float(start_q[4]) if start_q is not None else 0.0

    def violations(self, q):
        geometry = self.arm.geometry(q)
        floors = self.cfg["clearance_m"]
        values = [max(0.0, self.cfg["table_z"] + floors[k] - geometry[k]) for k in ("link", "gripper", "finger")]
        values.append(max(0.0, geometry["finger_r"] - self.cfg["max_radius_m"]))
        return np.asarray(values)

    def improves(self, before, after):
        a, b = self.violations(before), self.violations(after)
        # A recovery may leave a collision, but must improve it without a new one.
        return bool(np.all(b <= a + 1e-9) and np.any(b < a - 1e-9))

    def path_clear(self, start, target):
        start, target = vector(start), vector(target)
        n = max(1, int(np.ceil(np.max(np.abs(target - start)) / np.deg2rad(self.cfg["path_resolution_deg"]))))
        previous = start
        for alpha in np.linspace(0, 1, n + 1)[1:]:
            q = start + alpha * (target - start)
            bad = self.violations(q)
            if np.any(bad > 1e-9) and not self.improves(previous, q):
                return False
            previous = q
        return True

    def project(self, current, target, *, measured=None, retreat=False):
        current = vector(current)
        try:
            wanted = vector(target)
        except ValueError:
            return GuardResult(current, False, reason="nonfinite_or_shape")
        bounded = np.clip(wanted, self.limits[:, 0], self.limits[:, 1])
        geo = self.arm.geometry(current if measured is None else measured)
        speed = self.cfg["speed_deg_s"]
        if geo["finger"] - self.cfg["table_z"] < self.cfg["slow_zone_m"] or retreat:
            speed = min(speed, self.cfg["slow_speed_deg_s"])
        step = np.deg2rad(speed) * self.speed / self.hz
        candidate = current + np.clip(bounded - current, -step, step)
        improving = self.improves(current, candidate)
        roll_cap = np.deg2rad(self.cfg["roll_excursion_deg"])
        if abs(candidate[4] - self.roll_origin) > roll_cap and not improving:
            return GuardResult(current, False, reason="roll_excursion")
        if not self.path_clear(current, candidate):
            return GuardResult(current, False, reason="table_or_reach")
        # Measured lag must not turn a safe commanded step into an unsafe path.
        if measured is not None and not self.path_clear(measured, candidate):
            return GuardResult(current, False, reason="measured_path")
        return GuardResult(candidate, True, not np.allclose(candidate, wanted, atol=1e-10))
