"""Active identification using bounded experiments and held-out command replays.

All positions are spec-frame metres and all targets are absolute MJCF radians.
The legacy argon-hack armd also uses the spec frame (Kin.point subtracts
BASE_ORIGIN_MJCF). Its blocking state lock cannot supply a command trace; live
identification therefore requires the repository daemon's streaming protocol.
No hardware or network connection is opened by importing this module.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time
from typing import Literal, Protocol
from urllib.request import Request, urlopen

import cv2
import numpy as np
from scipy.optimize import least_squares

from r2p import gemini
from r2p.frames import HOME_Q, HZ, JOINTS, ROOT
from r2p.model import Arm, JointMap
from r2p.perception import CameraModel
from r2p.perception.geometry import fit_pose
from r2p.sim import TaskTwin
from r2p.sim.ik import JawIK
from r2p.sim.twin import load_physics
from r2p.tasks import load_task
from r2p.traces import write_trace
from .evaluate import mask_metrics, overlay, physics_replay
from .loop import atomic_image
from .patch import PATCH_SCHEMA, apply_patch
from .session import SessionData, load_session


class UnsafeExperiment(ValueError):
    """No motion may be sent for an experiment without a complete safe plan."""


@dataclass(frozen=True)
class Experiment:
    kind: Literal["hold_pose", "joint_sweep", "push", "pick_lift_place"]
    xyz: tuple[float, float, float] | None = None
    joint: int | None = None
    amplitude_deg: float = 5.0
    speed: float = 8.0  # maximum joint speed, degrees/sec
    obj: str | None = None
    direction_deg: float = 0.0
    distance_m: float = .025
    xy: tuple[float, float] | None = None
    lift_m: float = .07
    duration_s: float = 1.0

    @classmethod
    def parse(cls, value):
        if not isinstance(value, dict) or set(value) - set(cls.__dataclass_fields__):
            raise UnsafeExperiment("Unknown experiment fields")
        try:
            result = cls(**value)
            result.check_shape()
            return result
        except (TypeError, ValueError) as exc:
            raise UnsafeExperiment(str(exc)) from exc

    def check_shape(self):
        if self.kind not in ("hold_pose", "joint_sweep", "push", "pick_lift_place"):
            raise UnsafeExperiment("Unknown experiment kind")
        for key in ("speed", "amplitude_deg", "direction_deg", "distance_m", "lift_m", "duration_s"):
            v = getattr(self, key)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise UnsafeExperiment(f"{key} must be finite")
        if not 1 <= self.speed <= 20 or not .3 <= self.duration_s <= 4:
            raise UnsafeExperiment("speed must be 1..20 deg/s; duration .3..4 seconds")
        if not 1 <= self.amplitude_deg <= 12 or not .005 <= self.distance_m <= .045:
            raise UnsafeExperiment("amplitude must be 1..12 deg; push distance .005..045 m")
        if not .04 <= self.lift_m <= .12 or not -180 <= self.direction_deg <= 180:
            raise UnsafeExperiment("lift must be .04..12 m; direction -180..180 degrees")
        for key, size in (("xyz", 3), ("xy", 2)):
            v = getattr(self, key)
            if v is not None and (np.asarray(v).shape != (size,) or
                                  not np.isfinite(np.asarray(v, float)).all()):
                raise UnsafeExperiment(f"Invalid {key}")
        if self.kind == "hold_pose" and self.xyz is None:
            raise UnsafeExperiment("hold_pose needs xyz")
        if self.kind == "joint_sweep" and (type(self.joint) is not int or not 0 <= self.joint < 5):
            raise UnsafeExperiment("sweep joint must be 0..4; gripper sweeps are forbidden")
        if self.kind in ("push", "pick_lift_place") and not isinstance(self.obj, str):
            raise UnsafeExperiment("Contact experiments need an object id")
        if self.kind == "pick_lift_place" and self.xy is None:
            raise UnsafeExperiment("pick_lift_place needs destination xy")

    @property
    def key(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


EXPERIMENT_SCHEMA = {"type": "object", "required": ["kind"], "additionalProperties": False,
                     "properties": {
    "kind": {"type": "string", "enum": ["hold_pose", "joint_sweep", "push", "pick_lift_place"]},
    "xyz": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}},
    "xy": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}},
    "joint": {"type": "integer", "minimum": 0, "maximum": 4}, "obj": {"type": "string"},
    **{k: {"type": "number", "minimum": lo, "maximum": hi} for k, lo, hi in (
        ("amplitude_deg", 1, 12), ("speed", 1, 20), ("direction_deg", -180, 180),
        ("distance_m", .005, .045), ("lift_m", .04, .12), ("duration_s", .3, 4))}}}


@dataclass
class Safety:
    min_z: float = .003
    safe_z: float = .09
    max_z: float = .25
    reach: tuple[float, float] = (.10, .32)
    speed_cap: float = 20
    trajectory_speed_cap: float = 20
    joint_margin_deg: float = 0
    grid_margin: float = .024  # finger envelope beyond a cell edge
    # Real-bench contact depths (jaw midpoint, commanded). Servo sag lowers the real jaw ~3 mm at these poses, so a
    # commanded 8 mm push ran the fingertips ~4 mm over the paper and dragged the sheet. Grasp/place depths are the
    # ones that picked and placed caps on the real arm today.
    push_contact_z: float = .013
    push_standoff: float = .030
    grasp_z: float = .011
    place_z: float = .013
    closed_grip: float = -.10
    open_grip: float = .5        # ~60 mm jaw gap with vcal_oct25b: wide for a 33 mm cap, keeps the moving jaw high
    # Live abort: measured jaw below this outside grasp/place phases means the fingers are on the table.
    live_min_jaw_z: float = .007
    finish_tolerance: float = .025   # servo sag at the safe height is real (measured 12 mm) and is what we fit
    max_duration_s: float = 150   # vertical approach/retreat at the daemon's 8 deg/s near-table cap
    clearance: dict = field(default_factory=lambda: {"link": .005, "gripper": .004, "finger": .003})
    owner_lo: list | None = None
    owner_hi: list | None = None
    joint_map: JointMap | None = None


@dataclass
class Plan:
    experiment: Experiment
    q0: np.ndarray
    targets: np.ndarray
    phases: list[str]
    waypoints: list[list[float]]


def bench_scene(task=None):
    """Nominal supply/grid for dry runs; live execution requires a surveyed scene."""
    task = task or load_task("place_on_cell")
    objects = []
    for obj in task["objects"]:
        if "xy" not in obj or isinstance(obj["xy"], str):
            continue
        item = deepcopy(obj)
        size = item["size_m"]
        item["xyz"] = [*item.pop("xy"), size[-1] / 2]
        if not item.get("fixed"):
            item.update(friction=.6, mass_kg=.004)
        objects.append(item)
    # Lower-friction tabletop makes cap friction observable; MuJoCo combines
    # contacts by max(friction). The live table value must be independently fit.
    return {"task": task["name"], "objects": objects, "table": {"friction": .25},
            "board": deepcopy(task.get("board", {}))}


def scene_task(scene):
    """Freeze surveyed grid geometry with the scene, including older recordings."""
    task = load_task(scene.get("task", "place_on_cell"))
    cells = [o for o in scene["objects"] if re.fullmatch(r"r[0-2]c[0-2]", o["id"])]
    if scene.get("board"):
        task["board"] = deepcopy(scene["board"])
    elif len(cells) == 9:
        task["board"] = {"cell_m": max(o["size_m"][0] for o in cells),
                         "cells": {o["id"]: o["xyz"][:2] for o in cells}}
    return task


def away_from_board(scene, obj):
    """Pushes run toward the robot (180 deg): the pusher starts on the far side of the cap, so it never starts over
    the grid beside the lane (pushing away from the grid would start over it) or too close to the base to reach at
    contact height (pushing +x). The validator still checks every waypoint against the grid and clearances."""
    return 180


def push_length(obj, wanted):
    """Pushes run toward the robot; fingers-down contact stops being reachable ~12.7 cm from the base, so a cap close
    to the robot gets a shorter push (never below 8 mm) instead of an unreachable plan."""
    return round(float(np.clip(min(wanted, obj["xyz"][0] - .127), .008, wanted)), 3)


def experiment_menu(scene):
    obj = next(o for o in scene["objects"] if not o.get("fixed", False))
    x, y = obj["xyz"][:2]
    lane = round(y, 2)
    away = away_from_board(scene, obj)
    return [Experiment("hold_pose", xyz=(.18, lane, .12)),
            Experiment("joint_sweep", joint=1, amplitude_deg=7, speed=12),
            Experiment("push", obj=obj["id"], direction_deg=away, distance_m=push_length(obj, .025), speed=12),
            Experiment("hold_pose", xyz=(.25, lane, .11)),
            Experiment("joint_sweep", joint=2, amplitude_deg=8, speed=16),
            Experiment("push", obj=obj["id"], direction_deg=180, distance_m=push_length(obj, .012), speed=8),
            Experiment("pick_lift_place", obj=obj["id"], xy=(x+.025, y), lift_m=.08)]


def heldout_menu(scene):
    obj = next(o for o in scene["objects"] if not o.get("fixed", False))
    oid = obj["id"]
    lane = round(obj["xyz"][1]-.01, 2)
    return [Experiment("hold_pose", xyz=(.225, lane, .13)),
            Experiment("joint_sweep", joint=1, amplitude_deg=4, speed=10),
            Experiment("push", obj=oid, direction_deg=away_from_board(scene, obj), distance_m=push_length(obj, .035), speed=10)]


class Validator:
    def __init__(self, task=None, safety=None):
        self.task = task or load_task("place_on_cell")
        self.safety = safety or Safety()
        self.arm = Arm()
        self.ik = JawIK(self.arm)

    def _point(self, p):
        s = self.safety
        if not s.reach[0] <= np.linalg.norm(p[:2]) <= s.reach[1] or not s.min_z <= p[2] <= s.max_z:
            raise UnsafeExperiment("Outside reach band 0.10..0.32 m or height bounds")
        board = self.task.get("board", {})
        radius = board.get("cell_m", .0361) / 2 + s.grid_margin
        if any(np.all(np.abs(p[:2] - np.asarray(xy)) < radius) for xy in board.get("cells", {}).values()):
            raise UnsafeExperiment("Trajectory crosses protected grid cells")

    def plan(self, experiment, scene, q0=HOME_Q):
        e = experiment if isinstance(experiment, Experiment) else Experiment.parse(experiment)
        e.check_shape()
        s, q = self.safety, np.array(q0, float).copy()
        if q.shape != (6,) or not np.isfinite(q).all() or e.speed > s.speed_cap:
            raise UnsafeExperiment("Invalid start joints or excessive speed")
        qstart = q.copy()
        targets, phases, points = [], [], []
        objects = {o["id"]: o for o in scene["objects"]}
        self._point(self.arm.tips(q)[2])

        def joint(goal, phase, dwell=0):
            nonlocal q
            goal = np.asarray(goal, float)
            n = max(1, math.ceil(np.max(abs(goal-q)) / np.deg2rad(min(e.speed, s.trajectory_speed_cap)) * HZ))
            for alpha in np.linspace(0, 1, n+1)[1:]:
                targets.append(q + alpha*(goal-q))
                phases.append(phase)
            q = goal.copy()
            for _ in range(round(dwell*HZ)):
                targets.append(q.copy())
                phases.append(phase)

        def xyz(p, phase, grip=None, dwell=0):
            p = np.asarray(p, float)
            self._point(p)
            goal = _solve(self.arm, p, q, q[5] if grip is None else grip, margin_deg=s.joint_margin_deg)
            if np.linalg.norm(self.arm.tips(goal)[2]-p) > .003:
                raise UnsafeExperiment("IK residual exceeds 3 mm")
            points.append(p.tolist())
            joint(goal, phase, dwell)

        start = self.arm.tips(q)[2]
        xyz([start[0], start[1], max(s.safe_z, start[2])], "raise")
        if e.kind == "hold_pose":
            xyz(e.xyz, "hold", dwell=e.duration_s)
        elif e.kind == "joint_sweep":
            center = q.copy()
            for sign in (1, -1, 0):
                target = center.copy()
                target[e.joint] += sign*np.deg2rad(e.amplitude_deg)
                joint(target, "sweep", e.duration_s/3)
        else:
            if e.obj not in objects or objects[e.obj].get("fixed", False):
                raise UnsafeExperiment("Unknown or fixed contact object")
            obj = objects[e.obj]
            p = np.array(obj["xyz"], float)
            if obj["shape"] != "cylinder" or not .025 <= obj["size_m"][0] <= .04:
                raise UnsafeExperiment("Contact menu is calibrated for 25..40 mm caps only")
            def vertical(xy, z_from, z_to, phase, n=8, grip=None):
                # Straight down/up in Cartesian space: joint-space interpolation at low height sweeps the
                # fingers sideways across the cap and the paper (Oct 25 bench: it dragged the sheet).
                for z in np.linspace(z_from, z_to, n+1)[1:]:
                    xyz([*xy, z], phase, grip=grip)

            hover = .045
            if e.kind == "push":
                direction = np.array([math.cos(math.radians(e.direction_deg)), math.sin(math.radians(e.direction_deg)), 0])
                # Closed jaw = compact pusher; 30 mm beyond the rim so neither finger lands on the cap.
                beside = p - direction*(obj["size_m"][0]/2 + s.push_standoff)
                contact_z = max(s.push_contact_z, .6*obj["size_m"][-1])
                xyz([*beside[:2], s.safe_z], "approach", grip=s.closed_grip)
                xyz([*beside[:2], hover], "hover")
                vertical(beside[:2], hover, contact_z, "lower")
                end = p + direction*e.distance_m
                for alpha in np.linspace(0, 1, 13)[1:]:
                    xyz([*(beside[:2] + alpha*(end[:2]-beside[:2])), contact_z], "push")
                vertical(end[:2], contact_z, hover, "retreat")
                xyz([*end[:2], s.safe_z], "retreat", dwell=e.duration_s)
            else:
                xyz([*p[:2], s.safe_z], "approach", grip=s.open_grip)
                xyz([*p[:2], hover], "hover")
                vertical(p[:2], hover, s.grasp_z, "lower")
                xyz([*p[:2], s.grasp_z], "close", grip=-.10, dwell=.5)
                vertical(p[:2], s.grasp_z, hover, "lift")
                lift = max(s.safe_z, p[2]+e.lift_m)
                xyz([*p[:2], lift], "lift", dwell=.4)
                xyz([*e.xy, lift], "carry")
                xyz([*e.xy, hover], "hover")
                vertical(e.xy, hover, s.place_z, "place")
                xyz([*e.xy, s.place_z], "release", grip=s.open_grip, dwell=.5)
                vertical(e.xy, s.place_z, hover, "retreat")
                xyz([*e.xy, s.safe_z], "retreat", dwell=e.duration_s)
        last = self.arm.tips(q)[2]
        xyz([last[0], last[1], max(s.safe_z, last[2])], "safe_finish", dwell=.5)
        if len(targets)/HZ > s.max_duration_s:
            raise UnsafeExperiment("Experiment exceeds duration cap")
        limits = self.arm.range + np.deg2rad([s.joint_margin_deg, -s.joint_margin_deg])
        # Validate the complete interpolated path, not just IK endpoints.
        for index, target in enumerate([qstart, *targets]):
            if np.any(target < limits[:, 0]+1e-7) or np.any(target > limits[:, 1]-1e-7):
                raise UnsafeExperiment("Joint limit")
            p = self.arm.tips(target)[2]
            self._point(p)
            geo = self.arm.geometry(target)
            if any(geo[k] < v for k, v in s.clearance.items()):
                phase = "start" if index == 0 else phases[index-1]
                raise UnsafeExperiment(f"Mesh/table clearance during {phase}: "+str({k: round(geo[k], 4) for k in s.clearance}))
            if s.joint_map is not None and s.owner_lo is not None:
                owner = s.joint_map.to_owner(target)
                if np.any(owner < s.owner_lo) or np.any(owner > s.owner_hi):
                    raise UnsafeExperiment("Calibrated owner joint limit")
            for oid, obj in objects.items():
                if oid == e.obj or obj.get("role") == "target":
                    continue
                radius = max(obj["size_m"][:2]) / 2 + .025
                if p[2] < obj["xyz"][2]+.045 and np.linalg.norm(p[:2]-np.asarray(obj["xyz"])[:2]) < radius:
                    raise UnsafeExperiment("Path intersects another object")
        return Plan(e, qstart, np.asarray(targets), phases, points)

    def verdict(self, experiment, scene, q0=HOME_Q):
        try:
            plan = self.plan(experiment, scene, q0)
            return {"safe": True, "duration_s": len(plan.targets)/HZ, "finish_xyz": self.arm.tips(plan.targets[-1])[2].tolist()}
        except (ValueError, TypeError, KeyError) as exc:
            return {"safe": False, "reason": str(exc)}


def _solve(arm, xyz, q, grip, *, margin_deg=0):
    """Position-priority bounded IK; table/mesh validation remains mandatory."""
    initial = np.asarray(q, float).copy()
    initial[5] = grip
    def residual(joints):
        candidate = np.r_[joints, grip]
        a, b, mid = arm.tips(candidate)
        axis = arm.jac_tip(candidate)[2]
        return np.r_[mid-xyz, .004*(axis-[0, 0, -1]), .02*(a[2]-b[2])]
    margin = np.deg2rad(margin_deg)+1e-5
    limits = arm.range[:5] + [margin, -margin]
    result = least_squares(residual, np.clip(initial[:5], *limits.T), bounds=limits.T,
                           max_nfev=100, ftol=1e-8, xtol=1e-8, gtol=1e-8)
    return np.r_[result.x, grip]


def start_q():
    return _solve(Arm(), np.array([.18, -.105, .12]), HOME_Q, .9)


@dataclass
class Observation:
    experiment: Experiment
    session: SessionData
    split: Literal["train", "heldout"]
    plan: Plan


class Mover(Protocol):
    scene: dict
    camera: CameraModel
    validator: Validator
    def execute(self, experiment: Experiment, destination: Path, *, split="train") -> Observation: ...


def _trace_row(t, q, target, poses, phase):
    return {"t": t, "q": np.asarray(q).tolist(), "q_target": np.asarray(target).tolist(),
            "poses": poses, "grip_closed": bool(target[5] < .15), "command": None, "phase": phase}


def _save_observation(directory, e, plan, scene, camera, ticks, images, masks, *, source, split, extra=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    frame_times = []
    for i, (t, frame) in enumerate(images):
        atomic_image(directory / "frames" / f"{i:06d}.png", frame)
        frame_times.append(t)
    meta = {"task": scene.get("task", "place_on_cell"), "source": source, "scene": scene, "rig": "so101_bench",
            "session_kind": "synthetic" if source == "bot" else "bench", "frame_times": frame_times,
            "experiment": asdict(e), "split": split, "fps": HZ, **(extra or {})}
    write_trace(directory / "trace.jsonl", meta, ticks)
    gemini.atomic_json(directory / "camera.json", camera.to_dict())
    session = load_session(directory)
    session.masks = masks
    return Observation(e, session, split, plan)


class SimMover:
    """Physical MuJoCo truth; fitting sees only joint sensors and rendered masks.

    Perfect segmentation is an explicit synthetic camera assumption. Object
    coordinates are recovered with fit_pose, never read from truth for fitting.
    No GPU renderer, analytic friction shortcut, or proximity grasp is used.
    """
    def __init__(self, scene=None, *, truth_physics=None, truth_scene=None, camera=None, validator=None):
        self.scene = deepcopy(scene or bench_scene())
        self.__truth_physics = load_physics(truth_physics or "so101_bench")
        self.__truth_scene = deepcopy(truth_scene or self.scene)
        self.camera = camera or CameraModel.look_at(position=(.43, -.42, .43), target=(.18, -.06, .035), image_size=(320, 240))
        self.validator = validator or Validator(scene_task(self.scene))
        self.q0 = start_q()

    def execute(self, experiment, destination, *, split="train"):
        from .evaluate import CalibratedRenderer
        plan = self.validator.plan(experiment, self.scene, self.q0)
        twin = TaskTwin(self.__truth_scene, self.__truth_physics, plan.q0)
        renderer = CalibratedRenderer(twin, self.camera)
        indices = set(np.linspace(0, len(plan.targets), 5, dtype=int))
        ticks, images, masks = [], [], {}
        try:
            for i in range(len(plan.targets)+1):
                target = plan.targets[min(i, len(plan.targets)-1)]
                poses = {}
                if i in indices:
                    frame, observed_masks = renderer.render()
                    for obj in self.scene["objects"]:
                        mask = observed_masks.get(obj["id"])
                        if mask is not None and mask.sum() >= 8 and not obj.get("fixed"):
                            # Only table-supported final poses are scored. Mid-lift
                            # masks remain available for visual/grasp diagnostics.
                            if i in (0, len(plan.targets)):
                                poses[obj["id"]] = fit_pose(self.camera, obj, mask)
                                poses[obj["id"]].update(seen=True, source="rendered_mask_fit")
                    masks[len(images)] = observed_masks
                    images.append((i/HZ, frame))
                ticks.append(_trace_row(i/HZ, twin.q(), target, poses, plan.phases[min(i, len(plan.phases)-1)]))
                if i < len(plan.targets):
                    twin.step(target)
        finally:
            renderer.close()
            twin.close()
        return _save_observation(destination, experiment, plan, self.scene, self.camera, ticks, images, masks,
                                 source="bot", split=split, extra={"observations": "joint sensors + fit_pose on rendered segmentation"})


def http_json(url, payload=None, timeout=3):
    request = Request(url, data=None if payload is None else json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


class ArmdMover:
    """HTTP-only mover. Read-only construction; execute is the sole write boundary.

    A separate state poller runs at 30 Hz. Raw sensor/capture timestamps are kept;
    the trace is explicitly resampled to 30 Hz, with zero-order-held commands.
    Missing telemetry, stale camera, moved supply, or refused commands abort the
    campaign. Abort holds in place: blind automatic retreat is unsafe.
    """
    def __init__(self, scene, camera, *, url="http://127.0.0.1:8793", camera_url="http://127.0.0.1:8792",
                 joint_map=None, validator=None, perceiver=None, request=http_json, capture=None,
                 reset_timeout_s=60):
        self.scene, self.camera = deepcopy(scene), camera
        self.url, self.camera_url = url.rstrip("/"), camera_url.rstrip("/")
        self.joint_map = joint_map or JointMap.load(ROOT / "config/joint_map.json")
        self.validator = validator or Validator(scene_task(self.scene))
        self.perceiver, self.request, self.capture = perceiver, request, capture or self._capture
        if not math.isfinite(reset_timeout_s) or not 0 <= reset_timeout_s <= 300:
            raise ValueError("reset_timeout_s must be 0..300")
        self.reset_timeout_s = reset_timeout_s
        self.initial_state = self.request(self.url+"/state")
        self.modern = "owner_deg" in self.initial_state and "q_target" in self.initial_state
        if self.modern and validator is None:
            # The repository daemon reserves two degrees at joint limits and
            # caps near-table motion at 8 deg/s. Plan within both limits rather
            # than relying on streaming projection to change the experiment.
            self.validator.safety.joint_margin_deg = 2
            self.validator.safety.trajectory_speed_cap = 8
        if not self.modern:
            if self.initial_state.get("joint_map") != self.joint_map.name:
                raise UnsafeExperiment("Daemon and client joint maps differ")
            self.validator.safety.owner_lo = self.initial_state["lo"]
            self.validator.safety.owner_hi = self.initial_state["hi"]
            self.validator.safety.joint_map = self.joint_map
            # Legacy min_z is measured geometry, NOT a configurable clearance.
            self.validator.safety.clearance = {"link": .03, "gripper": .02, "finger": .015}
            self.validator.safety.min_z = .015

    def measured_q(self, state):
        return np.asarray(state["q"], float) if "owner_deg" in state else self.joint_map.to_model(state["q"])

    def _capture(self):
        started = time.monotonic()
        listing = self.request(self.camera_url+"/list")
        age = float(listing["0"]["age_s"])
        if not math.isfinite(age) or not 0 <= age <= 2.5:
            raise UnsafeExperiment("Camera is stale")
        with urlopen(self.camera_url+"/frame/0.jpg", timeout=3) as response:
            frame = cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.shape[1::-1] != self.camera.image_size:
            raise UnsafeExperiment("Camera frame/calibration dimensions differ")
        return frame, started-age, {"timestamp_source": "camserver_age_estimate", "age_s": age,
                                   "received_at": time.monotonic(), "sequence": listing["0"].get("frames")}

    def execute(self, experiment, destination, *, split="train"):
        state = self.request(self.url+"/state")
        plan = self.validator.plan(experiment, self.scene, self.measured_q(state))
        if not self.modern:
            raise UnsafeExperiment("Legacy armd blocks /state during moves and exposes no q_target; cannot record >=10 Hz. Use the repository streaming daemon; no POST sent.")
        if self.perceiver is None:
            raise UnsafeExperiment("Live recording needs a calibrated, saved perception setup")
        deadline = time.monotonic()+self.reset_timeout_s
        announced = False
        while True:
            frame, captured, _ = self.capture()
            observed = self.perceiver.estimate(frame, self.measured_q(state))
            reset = True
            for obj in self.scene["objects"]:
                if obj.get("fixed"):
                    continue
                pose = observed["poses"].get(obj["id"], {})
                reset &= (pose.get("source") == "vision" and bool(pose.get("seen")) and
                          np.linalg.norm(np.asarray(pose.get("xyz", [9,9,9]))-obj["xyz"]) <= .004)
            if reset:
                break
            if time.monotonic() >= deadline:
                raise UnsafeExperiment("Supply reset/vision check failed; restore surveyed scene before continuing")
            if not announced:
                print("Waiting for observed supply reset to surveyed positions; no motion commands are being sent.", flush=True)
                announced = True
            time.sleep(min(.2, max(0, deadline-time.monotonic())))
            state = self.request(self.url+"/state")
        # An operator may have changed the arm while restoring the supply.
        state = self.request(self.url+"/state")
        plan = self.validator.plan(experiment, self.scene, self.measured_q(state))
        started = time.monotonic()
        stop = threading.Event()
        sensors, captures, errors, commands = [], [], [], []

        def poll_state():
            while not stop.is_set():
                begin = time.monotonic()
                try:
                    row = self.request(self.url+"/state")
                    sensors.append({"t": time.monotonic()-started, "q": self.measured_q(row).tolist(),
                                    "q_target": row["q_target"]})
                except Exception as exc:
                    errors.append(str(exc)); stop.set()
                stop.wait(max(0, 1/HZ-(time.monotonic()-begin)))

        def poll_camera():
            previous = None
            while not stop.is_set():
                try:
                    image, timestamp, metadata = self.capture()
                    if timestamp >= started and metadata.get("sequence", timestamp) != previous:
                        captures.append((timestamp-started, image, metadata))
                        previous = metadata.get("sequence", timestamp)
                except Exception as exc:
                    errors.append(str(exc)); stop.set()
                stop.wait(.1)

        workers = [threading.Thread(target=fn, daemon=True) for fn in (poll_state, poll_camera)]
        for worker in workers:
            worker.start()
        try:
            for i, target in enumerate(plan.targets):
                if errors or stop.is_set():
                    raise UnsafeExperiment("Recording failed: "+"; ".join(errors))
                if i > HZ and (not sensors or time.monotonic()-started-sensors[-1]["t"] > .2):
                    raise UnsafeExperiment("State telemetry stalled")
                begin = time.monotonic()
                if sensors and plan.phases[i] not in ("close", "place", "release"):
                    jaw = self.validator.arm.tips(sensors[-1]["q"])[2]
                    if jaw[2] < self.validator.safety.live_min_jaw_z:
                        raise UnsafeExperiment(f"Live abort: measured jaw {jaw[2]*1000:.1f} mm above the table during {plan.phases[i]}")
                reply = self.request(self.url+"/cmd", {"joints": target.tolist(), "stream": True})
                if reply.get("ok") is False or any(k in reply for k in ("refused", "abort", "error")):
                    raise UnsafeExperiment("armd refused/aborted experiment")
                commands.append({"t": time.monotonic()-started, "q_target": reply["q_target"]})
                stop.wait(max(0, 1/HZ-(time.monotonic()-begin)))
            # Streaming ends where the guard-limited arm still lags: finish with one blocking move to the last target.
            final = self.request(self.url+"/cmd", {"joints": plan.targets[-1].tolist()})
            if any(k in final for k in ("refused", "abort", "error")):
                raise UnsafeExperiment("armd refused the final settle")
            # Camera has 1-2 s latency: hold high while final observations arrive.
            stop.wait(3)
            if (errors or len(sensors) < 2 or np.max(np.diff([r["t"] for r in sensors])) > .15 or
                    (len(sensors)-1)/(sensors[-1]["t"]-sensors[0]["t"]) < 10):
                raise UnsafeExperiment("State sampling did not sustain >=10 Hz")
            if not captures or captures[-1][0] < commands[-1]["t"]:
                raise UnsafeExperiment("No final camera observation after retreat")
            if self.validator.arm.tips(sensors[-1]["q"])[2][2] < self.validator.safety.safe_z-self.validator.safety.finish_tolerance:
                raise UnsafeExperiment("Measured finish is below the safe height")
        except BaseException:
            self.request(self.url+"/cmd", {"hold": True})
            raise
        finally:
            stop.set()
            for worker in workers:
                worker.join(timeout=4)
            # Preserve raw evidence even when quality gates refuse a fit.
            gemini.atomic_json(Path(destination).parent/(Path(destination).name+"_raw.json"),
                               {"sensors": sensors, "commands": commands, "errors": errors,
                                "captures": [{"t": t, **m} for t, _, m in captures]})
        times = np.array([r["t"] for r in sensors])
        measured = np.asarray([r["q"] for r in sensors])
        # No synthetic pre-telemetry targets. Start at the first measured sample.
        origin = times[0]
        end = times[-1]-origin
        ticks = []
        for i in range(int(end*HZ)+1):
            t = origin+i/HZ
            q = [np.interp(t, times, measured[:, j]) for j in range(6)]
            command_times = [row["t"] for row in commands]
            idx = np.searchsorted(command_times, t, side="right")-1
            target = commands[idx]["q_target"] if idx >= 0 else sensors[0]["q_target"]
            ticks.append(_trace_row(i/HZ, q, target, {}, "recorded"))
        images, masks = [], {}
        for t, image, metadata in sorted(captures, key=lambda item: item[0]):
            if not origin <= t <= times[-1]:
                continue
            index = min(round((t-origin)*HZ), len(ticks)-1)
            result = self.perceiver.estimate(image, ticks[index]["q"])
            poses = {k: v for k, v in result["poses"].items() if v.get("source") == "vision" and v.get("seen")}
            ticks[index]["poses"] = poses
            masks[len(images)] = result["masks"]
            images.append((t-origin, image))
        return _save_observation(destination, experiment, plan, self.scene, self.camera, ticks, images, masks,
                                 source="real", split=split, extra={"resampled": True, "sensor_hz": len(sensors)/(times[-1]-times[0]),
                                 "capture_timestamps": [m for _, _, m in captures]})


PARAMETERS = {
    "stiffness_scale": (.6, 1.8), "damping_scale": (.5, 1.8),
    "cap_friction": (.2, .9), "cap_mass_kg": (.002, .008), "table_friction": (.15, .8),
}


def initial_parameters(scene):
    cap = next(o for o in scene["objects"] if not o.get("fixed"))
    return {"stiffness_scale": 1., "damping_scale": 1., "cap_friction": float(cap.get("friction", .6)),
            "cap_mass_kg": float(cap.get("mass_kg", .004)), "table_friction": float(scene.get("table", {}).get("friction", .6))}


def materialize(base_physics, base_scene, parameters):
    physics, scene = deepcopy(base_physics), deepcopy(base_scene)
    for name in JOINTS:
        physics["joints"][name]["kp"] *= parameters["stiffness_scale"]
        physics["joints"][name]["kv"] *= parameters["damping_scale"]
    # sag_gain documents the same servo compliance, not a second injected force.
    physics["sag_gain"] = (np.asarray(physics["sag_gain"])/parameters["stiffness_scale"]).tolist()
    physics["object"].update(friction=parameters["cap_friction"], mass_kg=parameters["cap_mass_kg"])
    scene.setdefault("table", {})["friction"] = parameters["table_friction"]
    for obj in scene["objects"]:
        if not obj.get("fixed"):
            obj.update(friction=parameters["cap_friction"], mass_kg=parameters["cap_mass_kg"])
    return physics, scene


def _replay(observation, physics, scene, *, render=False):
    session = observation.session
    if render:
        return physics_replay(session, scene, physics=physics)
    # Sample the full measured trajectory, including ramps/overshoot, without
    # conflating sparse camera frame indices with 30 Hz command timestamps.
    indices = sorted(set(range(0, len(session.ticks), 10)) | {len(session.ticks)-1} |
                     {i for i, tick in enumerate(session.ticks) if tick["poses"]})
    sampled = SessionData(session.path, session.meta, session.ticks, indices,
                          [session.ticks[i]["t"] for i in indices], session.camera)
    snapshots = physics_replay(sampled, scene, physics=physics, render=False)
    return [(session.ticks[index], snapshots[i]) for i, index in enumerate(indices)]


def _vector(observation, physics, scene, *, joints=True, objects=True):
    out = []
    samples = _replay(observation, physics, scene)
    for tick, predicted in samples:
        if joints:
            # Sensor resolution is ~0.1 degree; silhouette fits carry millimetre
            # calibration/raster uncertainty. Do not fit pixel quantization at
            # the expense of the independently measured contact trajectory.
            out.extend((np.rad2deg(np.asarray(predicted["q"])-tick["q"])/.1).tolist())
    if objects:
        observed = [(real, predicted) for real, predicted in samples if real["poses"]]
        if not observed:
            raise ValueError("No independently observed object pose")
        real, predicted = observed[-1]
        required = {o["id"] for o in scene.get("objects", []) if not o.get("fixed")}
        if not required.issubset(real["poses"]):
            raise ValueError("Final observation is missing a manipulated object")
        for oid, pose in real["poses"].items():
            if required and oid not in required:
                continue
            if oid in predicted["poses"]:
                out.extend(((np.asarray(predicted["poses"][oid]["xyz"])[:2]-np.asarray(pose["xyz"])[:2])/.005).tolist())
    return np.asarray(out, float)


def fit_parameters(training, base_physics, base_scene, current, *, max_nfev=9):
    """Bounded numerical identification. Held-out data is rejected at the API."""
    if not training or any(o.split != "train" for o in training):
        raise ValueError("Fitting requires training experiments only")
    parameters = dict(current)
    uncertainty = {key: {"std": None, "identified": False, "bounds": list(bounds)} for key, bounds in PARAMETERS.items()}
    diagnostics = []
    groups = [("servo", ["stiffness_scale", "damping_scale"],
               [o for o in training if o.experiment.kind in ("hold_pose", "joint_sweep")]),
              ("contact", ["cap_friction", "cap_mass_kg", "table_friction"],
               [o for o in training if o.experiment.kind in ("push", "pick_lift_place")])]
    for group, keys, observations in groups:
        if not observations:
            continue
        scale = np.array([PARAMETERS[k][1]-PARAMETERS[k][0] for k in keys])
        lower = np.array([PARAMETERS[k][0] for k in keys])
        x0 = (np.array([parameters[k] for k in keys])-lower)/scale
        cache = {}
        def residual(x):
            cache_key = tuple(x)
            if cache_key not in cache:
                proposed = {**parameters, **dict(zip(keys, lower+x*scale))}
                physics, scene = materialize(base_physics, base_scene, proposed)
                cache[cache_key] = np.concatenate([_vector(o, physics, scene, objects=group == "contact") for o in observations])
            return cache[cache_key]
        # Explicit sensitivity probe exposes friction-max plateaus instead of
        # reporting a spurious point estimate with zero uncertainty.
        initial = residual(x0)
        if group == "contact":
            # Stick/slip transitions are nonsmooth. A local derivative from the
            # prior can stay in a false basin, so bracket cap friction globally
            # before the bounded multivariate refinement.
            candidates = []
            for friction in np.linspace(PARAMETERS["cap_friction"][0], PARAMETERS["cap_friction"][1], 15):
                trial = x0.copy()
                trial[0] = (friction-lower[0])/scale[0]
                values = residual(trial)
                candidates.append((float(values@values), trial))
            x0 = min(candidates, key=lambda item: item[0])[1]
        probe_base = residual(x0)
        probes = []
        for j in range(len(keys)):
            shifted = x0.copy()
            shifted[j] = shifted[j]+.025 if shifted[j] < .975 else shifted[j]-.025
            probes.append((residual(shifted)-probe_base)/(shifted[j]-x0[j]))
        jac0 = np.column_stack(probes)
        active = np.flatnonzero(np.linalg.norm(jac0, axis=0) > 1e-5)
        if len(active):
            def objective(x):
                full = x0.copy(); full[active] = x
                return residual(full)
            result = least_squares(objective, np.clip(x0[active], .0001, .9999), bounds=(0, 1),
                                   diff_step=.01, max_nfev=max_nfev, ftol=1e-5, xtol=1e-5, gtol=1e-5)
            final = x0.copy(); final[active] = result.x
            rank = int(np.linalg.matrix_rank(result.jac, tol=1e-5))
            covariance = np.linalg.pinv(result.jac.T@result.jac) * max(1., float(np.mean(result.fun**2)))
            for i, j in enumerate(active):
                key = keys[j]
                parameters[key] = float(lower[j]+final[j]*scale[j])
                std = float(np.sqrt(max(0, covariance[i, i]))*scale[j])
                uncertainty[key].update(std=std if rank == len(active) else None,
                                        identified=bool(rank == len(active) and std < scale[j]/2))
            diagnostics.append({"group": group, "evaluations": len(cache), "jacobian_rank": rank,
                                "active": [keys[i] for i in active], "cost_before": float(initial@initial),
                                "cost_after": float(result.fun@result.fun)})
    return parameters, uncertainty, diagnostics


def score_observations(observations, physics, scene, *, previews=None):
    """Independent joint, jaw, camera pose, and per-object visual residuals."""
    arm, rows, object_ious = Arm(), [], {}
    for number, observation in enumerate(observations):
        pairs = _replay(observation, physics, scene)
        joint, jaw, sag = [], [], []
        for measured, simulated in pairs:
            joint.extend(np.rad2deg(np.asarray(measured["q"])-simulated["q"])[:5])
            delta = arm.tips(measured["q"])[2]-arm.tips(simulated["q"])[2]
            jaw.append(float(np.linalg.norm(delta)*1000))
            sag.append(float(delta[2]*1000))
        observed = [(real, predicted) for real, predicted in pairs if real["poses"]]
        if not observed:
            raise ValueError("No independently observed object pose")
        measured, simulated = observed[-1]
        pose_errors, yaw_errors = [], []
        objects = {o["id"]: o for o in scene["objects"]}
        required = {oid for oid, obj in objects.items() if not obj.get("fixed")}
        if not required.issubset(measured["poses"]):
            raise ValueError("Final observation is missing a manipulated object")
        for oid, pose in measured["poses"].items():
            if oid not in required:
                continue
            if oid not in simulated["poses"]:
                continue
            predicted = simulated["poses"][oid]
            pose_errors.append(float(np.linalg.norm(np.asarray(pose["xyz"])-predicted["xyz"])*1000))
            # An unmarked circular cap has no observable yaw; do not fabricate it.
            if objects[oid]["shape"] != "cylinder":
                yaw_errors.append(abs((pose["yaw_deg"]-predicted["yaw_deg"]+180)%360-180))
        snapshots = _replay(observation, physics, scene, render=True)
        ious = []
        for i, snapshot in snapshots.items():
            for oid, real in observation.session.masks.get(i, {}).items():
                predicted = snapshot["masks"].get(oid, np.zeros_like(real))
                value = mask_metrics(real, predicted)["iou"]
                object_ious.setdefault(oid, []).append(value)
                ious.append(value)
        if not ious or not pose_errors:
            raise ValueError("Missing independent camera masks or final object observations; cannot score twin")
        row = {"experiment": asdict(observation.experiment), "session": str(observation.session.path),
               "split": observation.split, "joint_deg": float(np.sqrt(np.mean(np.square(joint)))),
               "jaw_mm": float(np.sqrt(np.mean(np.square(jaw)))), "sag_mm": float(np.sqrt(np.mean(np.square(sag)))),
               "object_mm": float(np.mean(pose_errors)), "object_deg": float(np.mean(yaw_errors)) if yaw_errors else None,
               "iou": float(np.mean(ious))}
        rows.append(row)
        if previews is not None:
            last = len(observation.session.frames)-1
            atomic_image(Path(previews)/f"experiment_{number:02d}.jpg", overlay(observation.session.frame(last), snapshots[last]))
    result = {key: float(np.mean([row[key] for row in rows])) for key in ("joint_deg", "jaw_mm", "sag_mm", "object_mm", "iou")}
    result["error"] = result["joint_deg"]/.5 + result["jaw_mm"]/2 + result["object_mm"]/5 + 2*(1-result["iou"])
    result["objects"] = {oid: {"iou": float(np.mean(values))} for oid, values in object_ious.items()}
    result["rows"] = rows
    return result


def keep_heldout(candidate, incumbent, *, min_gain=.001):
    reasons = []
    if not all(math.isfinite(candidate[k]) for k in ("error", "joint_deg", "jaw_mm", "object_mm", "iou")):
        reasons.append("nonfinite")
    if incumbent["error"]-candidate["error"] <= min_gain:
        reasons.append("insufficient_heldout_gain")
    for key, slack in (("joint_deg", .02), ("jaw_mm", .1), ("object_mm", .2)):
        if candidate[key] > incumbent[key]+slack:
            reasons.append(key+"_regression")
    for oid, row in incumbent["objects"].items():
        value = candidate["objects"].get(oid, {}).get("iou", -1)
        if not math.isfinite(value) or value < row["iou"]-.01:
            reasons.append("iou_regression:"+oid)
    return {"kept": not reasons, "reasons": reasons, "basis": "held-out experiments only"}


def choose_experiment(client, menu, tried, parameters, uncertainty, history, validator, scene, q0):
    available = [e for e in menu if e.key not in tried and validator.verdict(e, scene, q0)["safe"]]
    if not available:
        return None, {"source": "exhausted"}
    # The menu alternates complementary excitations; largest standardized
    # uncertainty promotes the first untried informative experiment.
    def priority(e):
        keys = ["cap_friction", "cap_mass_kg", "table_friction"] if e.kind in ("push", "pick_lift_place") else ["stiffness_scale", "damping_scale"]
        return sum(1 if not uncertainty[k]["identified"] else .1 for k in keys)/len(keys)
    available.sort(key=priority, reverse=True)
    fallback = available[0]
    prompt = {"instruction": "Choose one experiment from this menu to reduce uncertainty. Return its JSON fields only. All positions spec frame, metres; speed deg/s. No motion outside the menu.",
              "parameters": parameters, "uncertainty": uncertainty, "training_residual_history": history,
              "menu": [{k: v for k, v in asdict(e).items() if v is not None} for e in available],
              "bounds": asdict(validator.safety) if validator.safety.joint_map is None else {"reach": [.1, .32], "safe_z": validator.safety.safe_z}}
    try:
        answer, record = client.call([json.dumps(prompt)], role="main", tag="active_choose", schema=EXPERIMENT_SCHEMA)
        proposed = Experiment.parse(gemini.parse_json(answer))
        if proposed.key not in {e.key for e in available}:
            raise UnsafeExperiment("Choice is outside the validated, untried menu")
        validator.plan(proposed, scene, q0)
        return proposed, {"source": "main", "record": record}
    except (gemini.JSONError, UnsafeExperiment, TypeError, KeyError) as exc:
        return fallback, {"source": "informative_untried_fallback", "reason": str(exc)}


def propose_patch(client, training, physics, scene):
    if any(o.split != "train" for o in training):
        raise ValueError("Patch proposals cannot inspect held-out experiments")
    residual = score_observations(training[-1:], physics, scene)
    observation = training[-1]
    snapshots = _replay(observation, physics, scene, render=True)
    last = len(observation.session.frames)-1
    image = overlay(observation.session.frame(last), snapshots[last])
    answer, record = client.call([
        "Propose a conservative PATCH_SCHEMA scene correction from this TRAINING residual table and real|sim image. "
        "Left is observed, right simulated. Positions are spec-frame metres. Empty objects is valid. "
        "No camera changes or invented observations.\n"+json.dumps({"residuals": residual, "scene": scene}),
        gemini.image_part(image)], role="main", tag="active_patch", schema=PATCH_SCHEMA)
    patch = gemini.parse_json(answer)
    candidate = apply_patch(scene, patch)
    # Existing patch schema is intentionally broad; narrow the active bench
    # material bounds before allowing a proposal to reach the hold-out gate.
    for obj in candidate["objects"]:
        if not obj.get("fixed") and (not .2 <= obj.get("friction", .6) <= .9 or not .002 <= obj.get("mass_kg", .004) <= .008):
            raise ValueError("Patch exceeds active cap material bounds")
    return candidate, patch, record, residual


def offline_client():
    """Real schema/cache path, scripted chooser, zero Google requests."""
    def respond(**request):
        if request["config"]["response_json_schema"] == PATCH_SCHEMA:
            return {"objects": [], "why": "Offline numeric identification campaign", "done": False}
        prompt = json.loads(request["parts"][0])
        return prompt["menu"][0]
    return gemini.FakeGemini(responder=respond)


def run_campaign(mover, *, budget=6, out_dir=None, physics="so101_bench", client=None, patience=3,
                 candidate_dir=None, max_nfev=9):
    """Budget counts ALL executed experiments, including the frozen hold-out set.

    No candidate is activated. Live scene reset is checked by the mover before
    every experiment. Quota/cache failures propagate; there is no role fallback.
    """
    if type(budget) is not int or not 4 <= budget <= 30 or patience < 1:
        raise ValueError("budget must be 4..30 experiments (3 held-out + >=1 fit)")
    out = Path(out_dir or ROOT/"runs/active_twin"/time.strftime("%Y%m%d-%H%M%S"))
    if not re.fullmatch(r"[A-Za-z0-9_-]+", out.name):
        raise ValueError("Run name must contain only letters, numbers, underscore, hyphen")
    out.mkdir(parents=True, exist_ok=False)
    base_physics, base_scene = load_physics(physics), deepcopy(mover.scene)
    gemini.atomic_json(out/"prior_physics.json", base_physics)
    gemini.atomic_json(out/"prior_scene.json", base_scene)
    parameters = initial_parameters(base_scene)
    uncertainty = {k: {"std": None, "identified": False, "bounds": list(v)} for k, v in PARAMETERS.items()}
    client = client or (offline_client() if isinstance(mover, SimMover) else gemini.Gemini(max_requests=30, retries=1))
    status = {"schema": "r2p_active_twin/1", "status": "collecting_holdout", "run": out.name,
              "budget": budget, "experiments_used": 0, "iterations": [], "parameters": parameters,
              "synthetic": isinstance(mover, SimMover), "started": time.time(), "candidate_activated": False}
    def publish(stage):
        status["status"] = stage
        gemini.atomic_json(out/"status.json", status)
    publish("collecting_holdout")
    training, heldout, tried, history = [], [], set(), []
    stalled = 0
    try:
        for i, e in enumerate(heldout_menu(base_scene)):
            heldout.append(mover.execute(e, out/f"heldout_{i:02d}", split="heldout"))
            status["experiments_used"] += 1
            publish("collecting_holdout")
        best_physics, best_scene = materialize(base_physics, base_scene, parameters)
        before = best = score_observations(heldout, best_physics, best_scene, previews=out/"before")
        status.update(before=before, heldout=best)
        while status["experiments_used"] < budget:
            q0 = mover.q0 if isinstance(mover, SimMover) else mover.measured_q(mover.request(mover.url+"/state"))
            e, choice = choose_experiment(client, experiment_menu(base_scene), tried, parameters, uncertainty,
                                         history, mover.validator, base_scene, q0)
            if e is None:
                status["stop_reason"] = "safe_menu_exhausted"
                break
            tried.add(e.key)
            iteration = len(training)+1
            publish("executing")
            observation = mover.execute(e, out/f"train_{iteration:02d}", split="train")
            training.append(observation)
            status["experiments_used"] += 1
            publish("fitting")
            fitted, estimated_uncertainty, diagnostics = fit_parameters(training, base_physics, base_scene, parameters, max_nfev=max_nfev)
            proposed_physics, proposed_scene = materialize(base_physics, base_scene, fitted)
            numeric = score_observations(heldout, proposed_physics, proposed_scene, previews=out/f"iter_{iteration:02d}"/"numeric")
            verdict = keep_heldout(numeric, best)
            kept = verdict["kept"]
            if kept:
                parameters, uncertainty = fitted, estimated_uncertainty
                best_physics, best_scene, best = proposed_physics, proposed_scene, numeric
            patch_info = {}
            residual = None
            publish("argon_patch")
            try:
                patched_scene, patch, record, residual = propose_patch(client, training, best_physics, best_scene)
                patched = best if patched_scene == best_scene else score_observations(heldout, best_physics, patched_scene)
                patch_verdict = keep_heldout(patched, best)
                patch_info = {"patch": patch, "record": record, "verdict": patch_verdict, "heldout": patched}
                if patch_verdict["kept"]:
                    best_scene, best = patched_scene, patched
                    # Persist accepted geometry in future numeric candidates;
                    # material parameters remain explicitly represented.
                    base_scene = deepcopy(best_scene)
                    values = initial_parameters(best_scene)
                    for key in ("cap_friction", "cap_mass_kg", "table_friction"):
                        parameters[key] = values[key]
                    kept = True
                    residual = None
            except (ValueError, gemini.JSONError) as exc:
                patch_info = {"rejected": str(exc)}
            residual = residual or score_observations(training[-1:], best_physics, best_scene)
            history.append(residual)
            row = {"iteration": iteration, "experiment": asdict(e), "choice": choice, "fit": diagnostics,
                   "estimated_parameters": fitted, "estimated_uncertainty": estimated_uncertainty,
                   "numeric_verdict": verdict, "numeric_heldout": numeric, "patch": patch_info,
                   "kept": kept, "heldout": best, "training": residual}
            status["iterations"].append(row)
            status.update(parameters=parameters, uncertainty=uncertainty, heldout=best)
            gemini.atomic_json(out/f"iter_{iteration:02d}.json", row)
            gemini.atomic_json(out/"residuals.json", {"training": history, "heldout": best["rows"]})
            publish("evaluated")
            stalled = 0 if kept else stalled+1
            if stalled >= patience:
                status["stop_reason"] = "heldout_plateau"
                break
        status.setdefault("stop_reason", "experiment_budget")
        destination = Path(candidate_dir or ROOT/"config/physics")/f"so101_bench.{out.name}.json"
        if destination.exists():
            raise FileExistsError(f"Candidate already exists: {destination}")
        candidate = {**best_physics, "active_twin": {"run": str(out), "activated": False,
                     "synthetic": status["synthetic"], "parameters": parameters, "uncertainty": uncertainty,
                     "before": before, "after": best, "scene_path": str(out/"candidate_scene.json")}}
        gemini.atomic_json(destination, candidate)
        gemini.atomic_json(out/"candidate_scene.json", best_scene)
        status["candidate_physics"] = str(destination)
        report = [f"# Active twin {out.name}", "", "Synthetic physical MuJoCo campaign; no transfer claim." if status["synthetic"] else "Recorded hardware campaign; candidate not activated.",
                  "", "| Held-out metric | Before | After |", "|---|---:|---:|"]
        report += [f"| {key} | {before[key]:.6f} | {best[key]:.6f} |" for key in ("error", "joint_deg", "jaw_mm", "sag_mm", "object_mm", "iou")]
        report += ["", f"Experiments: {status['experiments_used']}/{budget}; stop: {status['stop_reason']}.",
                   f"Candidate: `{destination}`. Apply its companion `candidate_scene.json` explicitly; per-object and table properties override physics defaults.",
                   "", "Parameters and uncertainty:", "```json", json.dumps({"parameters": parameters, "uncertainty": uncertainty}, indent=2), "```", "",
                   "Circular cap yaw is unobservable. Segmentation is perfect only in synthetic campaigns; object positions are fitted from rendered masks. Hold-out experiments never enter the optimizer or patch prompt. A repeated hold-out gate is model selection, not an independent transfer audit."]
        temporary = out/".report.md.tmp"
        temporary.write_text("\n".join(report)+"\n", encoding="utf-8")
        temporary.replace(out/"report.md")
        publish("complete")
        return status
    except BaseException as exc:
        status.update(error=f"{type(exc).__name__}: {exc}", candidate_activated=False)
        publish("failed")
        raise
