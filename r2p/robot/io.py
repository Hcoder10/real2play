"""Owner-space IO (degrees, gripper 0..100), with lazy LeRobot imports.

Connection deliberately bypasses SOFollower.connect(): configure enables torque.
The register sequence is tested against an instrumented bus, without hardware.
"""
from pathlib import Path
import hashlib
import os
import tempfile

import numpy as np

from r2p.frames import JOINTS


def load_magnitude(value):
    """LeRobot has already decoded the direction bit. Never mask a signed load."""
    magnitude = abs(float(value))
    if not np.isfinite(magnitude):
        raise ValueError("load telemetry must be finite")
    return magnitude


def vector(value):
    q = np.asarray(value, float)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError("expected six finite joint values")
    return q.copy()


class BusLease:
    """OS-held advisory lock; automatically released if a process exits."""
    def __init__(self, port):
        key = hashlib.sha256(str(port).encode()).hexdigest()[:24]
        self.file = (Path(tempfile.gettempdir()) / f"r2p-bus-{key}.lock").open("a+b")
        self.file.seek(0)
        self.file.write(b"0")
        self.file.flush()
        self.file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.file.close()
            raise RuntimeError(f"bus {port} is already owned; use armd") from exc

    def close(self):
        if not self.file.closed:
            self.file.close()


def make_follower(cfg):
    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
    if not cfg.get("port"):
        raise ValueError("configure a follower port before opening hardware")
    return SOFollower(SOFollowerRobotConfig(
        port=cfg["port"], id=cfg.get("robot_id", "follower_1"),
        calibration_dir=Path(cfg["calibration_dir"]).expanduser(),
        use_degrees=True, disable_torque_on_disconnect=False, cameras={},
    ))


class ReadOnlyBus:
    """No register writes, including connect/disconnect. Retains the follower."""
    def __init__(self, cfg=None, *, follower=None):
        from .config import load_config
        self.cfg = cfg or load_config()
        self._lease = None
        self._closed = False
        if follower is None:
            self._lease = BusLease(self.cfg.get("port"))
        try:
            self.follower = follower if follower is not None else make_follower(self.cfg)
            self.bus = self.follower.bus
            self.follower.config.disable_torque_on_disconnect = False
            self.bus.connect()  # bus handshake only: never SOFollower.connect/configure
        except BaseException:
            if self._lease:
                self._lease.close()
            raise

    def read_register(self, name, *, normalize=False):
        return self.bus.sync_read(name, normalize=normalize)

    def read(self):
        result = self.read_register("Present_Position", normalize=True)
        return vector([result[j] for j in JOINTS])

    def loads(self):
        return {j: load_magnitude(v) for j, v in self.read_register("Present_Load").items()}

    def close(self, torque_off=False):
        if torque_off:
            raise ValueError("ReadOnlyBus never writes; use armd release at rest")
        if not self._closed:
            try:
                self.bus.disconnect(disable_torque=False)
            finally:
                self._closed = True
                if self._lease:
                    self._lease.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class RealRobot(ReadOnlyBus):
    def __init__(self, cfg=None, *, follower=None):
        super().__init__(cfg, follower=follower)
        try:
            if not self.bus.is_calibrated:
                raise RuntimeError("EEPROM does not match calibration; run read-only preflight")
            self.bus.disable_torque()
            present = self.read_register("Present_Position", normalize=False)
            self.bus.sync_write("Goal_Position", present, normalize=False)
            self.follower.configure()
            self.last_target = self.read()
        except BaseException:
            super().close()
            raise

    def write(self, owner):
        q = vector(owner)
        if self._closed:
            raise RuntimeError("robot is closed")
        self.bus.sync_write("Goal_Position", dict(zip(JOINTS, q)))
        self.last_target = q
        return q.copy()

    def hold(self):
        self.write(self.read())

    def torque_off(self):
        self.bus.disable_torque()

    def close(self, torque_off=False):
        if not self._closed:
            if torque_off:
                self.torque_off()
            super().close()


class MockRobot:
    """Deterministic first-order servo. Call step once per tick; reads are pure."""
    def __init__(self, sag=None, *, arm=None, joint_map=None, q0=None, tau_s=0.07, latency_ticks=0):
        from r2p.model import Arm, JointMap
        from r2p.frames import HOME_Q
        from .gravity import GravitySag
        self.arm = arm or Arm()
        self.joint_map = joint_map or JointMap()
        self.sag = sag if sag is not None else GravitySag(self.arm)
        self.q = vector(HOME_Q if q0 is None else q0)
        self.target = self.q.copy()
        from collections import deque
        if not isinstance(latency_ticks, int) or latency_ticks < 0:
            raise ValueError("latency_ticks must be a nonnegative integer")
        self.latency_ticks = latency_ticks
        self.pending = deque([self.target.copy() for _ in range(latency_ticks)])
        self.tau_s = float(tau_s)
        self.torque_enabled = True
        self.holding = False
        self.grip_floor = -0.17
        self.closed = False
        self.load_values = dict.fromkeys(JOINTS, 0.0)

    def read(self):
        return self.joint_map.to_owner(self.q)

    def write(self, owner):
        if self.closed or not self.torque_enabled:
            raise RuntimeError("mock torque is off or connection closed")
        self.target = self.joint_map.to_model(vector(owner))
        return self.joint_map.to_owner(self.target)

    def step(self, dt=1/30):
        if self.torque_enabled:
            self.pending.append(self.target.copy())
            equilibrium = self.sag(self.pending.popleft())
            self.q += (1 - np.exp(-dt / self.tau_s)) * (equilibrium - self.q)
            if self.holding:
                self.q[5] = max(self.q[5], self.grip_floor)
        return self.q.copy()

    def set_grasp(self, floor_rad=None):
        self.holding = floor_rad is not None
        if self.holding:
            self.grip_floor = float(floor_rad)

    def loads(self):
        return {j: load_magnitude(v) for j, v in self.load_values.items()}

    def hold(self):
        self.target = self.q.copy()
        self.pending.clear()
        self.pending.extend(self.target.copy() for _ in range(self.latency_ticks))

    def torque_off(self):
        self.torque_enabled = False

    def close(self, torque_off=False):
        if torque_off:
            self.torque_off()
        self.closed = True
