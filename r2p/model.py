"""SO-101 kinematics on the official MJCF, in the spec frame (r2p.frames). Used by sims, the real-arm guard, perception
and the policies' FK features.

  arm = Arm()
  arm.tips(q)          -> (fixed_tip, moving_tip, mid)  spec frame, metres
  arm.jac_tip(q)       -> (Jpos 3x5 of the jaw midpoint, Jrot 3x5, finger axis unit vector)
  arm.geometry(q)      -> {"link", "gripper", "finger": min z (m), "finger_r", "finger_zmax"}
  arm.range            -> (6, 2) joint limits (rad)
  JointMap             -> owner degrees (LeRobot so_follower, use_degrees=True, gripper 0-100) <-> MJCF rad
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from r2p.frames import BASE_ORIGIN_MJCF, FIXED_TIP, JOINTS, MJCF, MOVING_TIP

FINGER_LOCAL_Z = -0.062      # gripper-body vertices below this (towards the tips) are the fixed finger
MOVING_FINGER_R = 0.030      # moving-jaw vertices further than this from its pivot are the moving finger
STATIC_BODIES = ("base", "shoulder")


def _hull(v):
    try:
        from scipy.spatial import ConvexHull
        return v[ConvexHull(v).vertices]
    except Exception:  # noqa: BLE001 - degenerate mesh: keep a decimated cloud
        return v[:: max(1, len(v) // 400)]


class Arm:
    def __init__(self, mjcf: Path = MJCF):
        self.m = mujoco.MjModel.from_xml_path(str(mjcf))
        self.d = mujoco.MjData(self.m)
        self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(j).id] for j in JOINTS])
        self.dofadr = np.array([self.m.jnt_dofadr[self.m.joint(j).id] for j in JOINTS])
        self.range = np.array([self.m.jnt_range[self.m.joint(j).id] for j in JOINTS])
        self.parts = []           # (body_id, hull verts in body frame, kind)
        for b in range(1, self.m.nbody):
            name = self.m.body(b).name
            if name in STATIC_BODIES:
                continue
            vs = []
            for g in range(self.m.ngeom):
                if self.m.geom_bodyid[g] != b or self.m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                    continue
                if self.m.geom_contype[g] == 0 and self.m.geom_conaffinity[g] == 0 and self.m.geom_group[g] != 3:
                    continue
                mid = self.m.geom_dataid[g]
                v = self.m.mesh_vert[self.m.mesh_vertadr[mid]: self.m.mesh_vertadr[mid] + self.m.mesh_vertnum[mid]]
                R = np.zeros(9)
                mujoco.mju_quat2Mat(R, self.m.geom_quat[g])
                vs.append(_hull(v) @ R.reshape(3, 3).T + self.m.geom_pos[g])
            if not vs:
                continue
            v = np.vstack(vs)
            if name == "gripper":
                fin = v[:, 2] < FINGER_LOCAL_Z
                self.parts += [(b, v[fin], "finger"), (b, v[~fin], "gripper")]
            elif name == "moving_jaw_so101_v1":
                fin = np.linalg.norm(v, axis=1) > MOVING_FINGER_R
                self.parts += [(b, v[fin], "finger"), (b, v[~fin], "gripper")]
            else:
                self.parts.append((b, v, "link"))

    def fk(self, q):
        self.d.qpos[:] = 0
        self.d.qpos[self.qadr] = np.asarray(q, float)[: len(JOINTS)]
        mujoco.mj_kinematics(self.m, self.d)
        return self.d

    def point(self, body, local):
        """A body-local point in the spec frame (call fk first)."""
        b = self.m.body(body).id
        return self.d.xpos[b] + self.d.xmat[b].reshape(3, 3) @ local - BASE_ORIGIN_MJCF

    def tips(self, q):
        self.fk(q)
        a, b = self.point(*FIXED_TIP), self.point(*MOVING_TIP)
        return a, b, (a + b) / 2

    def geometry(self, q) -> dict:
        self.fk(q)
        out = {"finger": np.inf, "gripper": np.inf, "link": np.inf, "finger_r": 0.0, "finger_zmax": -np.inf}
        for b, v, kind in self.parts:
            w = v @ self.d.xmat[b].reshape(3, 3).T + self.d.xpos[b] - BASE_ORIGIN_MJCF
            out[kind] = min(out[kind], float(w[:, 2].min()))
            if kind == "finger":
                out["finger_r"] = max(out["finger_r"], float(np.hypot(w[:, 0], w[:, 1]).max()))
                out["finger_zmax"] = max(out["finger_zmax"], float(w[:, 2].max()))
        return out

    def jac_tip(self, q):
        """Jacobian of the jaw midpoint w.r.t. the 5 arm joints (3x5 pos, 3x5 rot) and the finger axis (gripper -z)."""
        self.fk(q)
        mujoco.mj_comPos(self.m, self.d)
        b = self.m.body(FIXED_TIP[0]).id
        _, _, mid = self.tips(q)
        mujoco.mj_comPos(self.m, self.d)
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        mujoco.mj_jac(self.m, self.d, jp, jr, mid + BASE_ORIGIN_MJCF, b)
        axis = -self.d.xmat[b].reshape(3, 3)[:, 2]
        return jp[:, self.dofadr[:5]], jr[:, self.dofadr[:5]], axis

    def gravity_torque(self, q):
        """Joint-space gravity torque (Nm) at q (qfrc_bias with zero velocity), JOINTS order."""
        self.fk(q)
        self.d.qvel[:] = 0
        mujoco.mj_forward(self.m, self.d)
        return self.d.qfrc_bias[self.dofadr].copy()


@dataclass
class JointMap:
    """model_deg = sign * owner_deg + offset_deg (arm joints); gripper model_rad = closed + g/100 * (open - closed)."""
    sign: np.ndarray = field(default_factory=lambda: np.ones(5))
    offset: np.ndarray = field(default_factory=lambda: np.zeros(5))
    gc: float = -0.17
    go: float = 1.745
    name: str = "identity"
    verified: bool = False

    @classmethod
    def load(cls, path: Path, candidate: str | None = None) -> "JointMap":
        d = json.loads(Path(path).read_text())
        name = candidate or d["active"]
        c = d["candidates"][name]
        j = [c["joints"][k] for k in JOINTS[:5]]
        return cls(np.array([x.get("sign", 1) for x in j], float), np.array([x.get("offset_deg", 0.0) for x in j], float),
                   c["gripper"]["closed_rad"], c["gripper"]["open_rad"], name,
                   all(x.get("measured", False) for x in j))

    def to_model(self, owner):
        x = np.asarray(owner, float)
        out = np.empty_like(x)
        out[..., :5] = np.deg2rad(self.sign * x[..., :5] + self.offset)
        out[..., 5] = self.gc + x[..., 5] / 100.0 * (self.go - self.gc)
        return out

    def to_owner(self, model):
        x = np.asarray(model, float)
        out = np.empty_like(x)
        out[..., :5] = (np.rad2deg(x[..., :5]) - self.offset) * self.sign
        out[..., 5] = (x[..., 5] - self.gc) / (self.go - self.gc) * 100.0
        return out
