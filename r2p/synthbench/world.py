"""MuJoCo FK bench with rigid gripper-frame attachment and deterministic contacts.

This is a perception fixture, not the dynamic reference twin in r2p.sim.
Commands are absolute MJCF radians. Object centres and truth use spec metres.
"""
from __future__ import annotations

import copy
import numpy as np

from r2p.frames import HOME_Q
from r2p.model import Arm
from r2p.perception.camera import CameraModel
from r2p.perception.geometry import dimensions, rotation
from .render import CpuRenderer


class World:
    def __init__(self, task, rng=None, cam=None):
        from r2p.tasks import load_task, sample_scene
        self.task = load_task(task) if not isinstance(task, dict) else copy.deepcopy(task)
        self.rng = np.random.default_rng(0) if rng is None else rng
        self.cam = cam if cam is not None else CameraModel.look_at()
        self.scene = sample_scene(self.task, self.rng)
        self.objects = copy.deepcopy(self.scene["objects"])
        self._poses = {o["id"]: {"xyz": list(o["xyz"]), "yaw_deg": float(o.get("yaw_deg", 0)),
                                "R": rotation(o.get("yaw_deg", 0)), "size_m": o["size_m"]} for o in self.objects}
        # Scene samplers need only carry contract keys; preserve appearance/shape details.
        templates = {o["id"]: o for o in self.task["objects"]}
        self.objects = [{**templates[o["id"]], **o} for o in self.objects]
        self.arm = Arm()
        self._q = HOME_Q.copy()
        self.held = None
        self._closed = False
        self.t = 0.0
        self._renderer = None
        self.last_masks = {}

    def q(self):
        return self._q.copy()

    def poses(self):
        return {oid: {k: copy.deepcopy(v) for k, v in pose.items() if k != "R"}
                for oid, pose in self._poses.items()}

    @property
    def truth(self):
        return self.poses()

    def set_pose(self, oid, xyz, yaw_deg=0):
        self._poses[oid].update(xyz=list(xyz), yaw_deg=float(yaw_deg), R=rotation(yaw_deg))

    def _gripper(self, q):
        a, b, mid = self.arm.tips(q)
        R = self.arm.d.xmat[self.arm.m.body("gripper").id].reshape(3, 3).copy()
        return mid, R, np.linalg.norm(a-b)

    def _jaw_floor(self, q, width):
        # Contact prevents the synthetic jaws closing through a held object.
        low, high = self.arm.range[5]
        for _ in range(18):
            value = (low+high)/2
            trial = q.copy()
            trial[5] = value
            if self._gripper(trial)[2] < width:
                low = value
            else:
                high = value
        return max(q[5], high)

    def _release(self, oid):
        obj = next(o for o in self.objects if o["id"] == oid)
        pose = self._poses[oid]
        size = dimensions(obj)
        floor = 0.
        for other in self.objects:
            if other["id"] == oid:
                continue
            support = self._poses[other["id"]]
            extent = dimensions(other)
            delta = (np.asarray(pose["xyz"])-support["xyz"]) @ rotation(support["yaw_deg"])
            if other.get("shape") == "bowl":
                fits = np.linalg.norm(delta[:2])+np.linalg.norm(size[:2])/2 <= extent[0]/2
                height = support["xyz"][2]-extent[2]/2+other.get("floor_m", other.get("wall_m", .005))
            else:
                fits = np.all(np.abs(delta[:2]) + size[:2]/2 <= extent[:2]/2+.002)
                height = support["xyz"][2]+extent[2]/2
            if fits and height <= pose["xyz"][2]+.02:
                floor = max(floor, height)
        self.set_pose(oid, [*pose["xyz"][:2], floor+size[2]/2], pose["yaw_deg"])

    def step(self, q_target):
        target = np.asarray(q_target, float)
        if target.shape != (6,) or not np.isfinite(target).all():
            raise ValueError("Expected six finite absolute joint targets")
        target = np.clip(target, self.arm.range[:, 0], self.arm.range[:, 1])
        old_mid, _, _ = self._gripper(self._q)
        closed = bool(target[5] < .35)
        if self.held is not None:
            oid, offset, local_R, width = self.held
            if closed:
                target[5] = self._jaw_floor(target, width)
            else:
                self._release(oid)
                self.held = None
        mid, R, gap = self._gripper(target)
        if closed and not self._closed and self.held is None:
            candidates = [(np.linalg.norm(np.asarray(self._poses[o["id"]]["xyz"])-mid), o)
                          for o in self.objects if not o.get("fixed")
                          and o.get("role") not in ("container", "target", "fixture")]
            if candidates:
                distance, obj = min(candidates, key=lambda x: x[0])
                if distance < .025:
                    oid = obj["id"]
                    size = dimensions(obj)
                    width = min(size[:2])
                    target[5] = self._jaw_floor(target, width)
                    mid, R, gap = self._gripper(target)
                    pose = self._poses[oid]
                    self.held = (oid, R.T @ (np.asarray(pose["xyz"])-mid), R.T @ pose["R"], width)
        if self.held is not None:
            oid, offset, local_R, _ = self.held
            global_R = R @ local_R
            self._poses[oid].update(xyz=(mid+R@offset).tolist(), R=global_R,
                                   yaw_deg=float(np.rad2deg(np.arctan2(global_R[1, 0], global_R[0, 0]))))
        else:
            self._push(old_mid, mid)
        self._q, self._closed = target, closed
        self.t += 1/30
        return self.poses()

    def _push(self, old_mid, mid):
        movement = mid-old_mid
        distance = np.linalg.norm(movement[:2])
        if distance < 1e-8 or distance > .08:
            return
        for obj in self.objects:
            if obj.get("fixed") or obj.get("role") in ("container", "fixture"):
                continue
            pose = self._poses[obj["id"]]
            size = dimensions(obj)
            if min(old_mid[2], mid[2]) > pose["xyz"][2]+size[2]/2+.008:
                continue
            xy = np.asarray(pose["xyz"][:2])
            delta = movement[:2]
            fraction = np.clip(np.dot(xy-old_mid[:2], delta)/np.dot(delta, delta), 0, 1)
            nearest = old_mid[:2]+fraction*delta
            radius = min(size[:2])/2+.006  # finger-pad thickness
            if np.linalg.norm(xy-nearest) < radius and np.dot(xy-old_mid[:2], delta) >= 0:
                new_xy = xy+delta*(1-fraction)
                self.set_pose(obj["id"], [*new_xy, pose["xyz"][2]], pose["yaw_deg"])

    def grip_closed(self):
        return self._closed

    def truth_check(self):
        from r2p.tasks import check
        return check(self.task, {"objects": self.poses(), "gripper_open": not self._closed,
                                 "site_xyz": self._gripper(self._q)[0].tolist()})

    def render_layers(self, **kwargs):
        if self._renderer is None:
            self._renderer = CpuRenderer(self.arm, self.cam)
        frame, self.last_masks, shadow = self._renderer.render(self._q, self.objects, self._poses, **kwargs)
        return frame, self.last_masks, shadow

    def render(self, **kwargs):
        return self.render_layers(**kwargs)[0]

    def close(self):
        self._renderer = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class OracleDetector:
    """Test-only oracle supplies visible boxes, never masks or poses to perception."""
    def __init__(self, world, padding=3):
        self.world, self.padding = world, padding
        self.calls = []

    def detect(self, frame, task, *, role="small"):
        self.calls.append(role)
        if not self.world.last_masks:
            self.world.render()
        result = {}
        for obj in task["objects"]:
            mask = self.world.last_masks.get(obj["id"])
            if mask is None or not mask.any():
                continue
            y, x = np.nonzero(mask)
            p = self.padding
            result[obj["id"]] = [max(0, x.min()-p), max(0, y.min()-p),
                                  min(frame.shape[1], x.max()+p+1), min(frame.shape[0], y.max()+p+1)]
        return result
