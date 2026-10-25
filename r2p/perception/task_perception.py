"""Task-driven 30 Hz tracking with one-call setup and bounded async recovery.

Detector protocol: detect(frame_bgr, task, *, role) -> {id: [x0,y0,x1,y1]}.
An injected callable with the same signature is also accepted. Detector boxes
are pixel xyxy; Gemini's normalized yxyx boxes are converted by GeminiDetector.
"""
from __future__ import annotations

import copy
import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .appearance import Appearance, lab_indices, seed_mask
from .geometry import dimensions, fit_pose, rotation, is_screwdriver


class GeminiDetector:
    def detect(self, frame, task, *, role="small"):
        from r2p import gemini  # no network client is imported until explicitly used
        prompt = ("Locate every visible task object in this image. Return JSON {\"objects\": "
                  "[{\"id\":task_object_id,\"box_2d\":[ymin,xmin,ymax,xmax]}]}. "
                  "Coordinates are 0..1000, relative to the full image. Omit absent objects. Task: "
                  + json.dumps(task))
        schema = {"type": "object", "required": ["objects"], "properties": {"objects": {
            "type": "array", "items": {"type": "object", "required": ["id", "box_2d"], "properties": {
                "id": {"type": "string"}, "box_2d": {"type": "array", "minItems": 4, "maxItems": 4,
                                                        "items": {"type": "number", "minimum": 0, "maximum": 1000}}}}}}}
        response, _ = gemini.call([prompt, gemini.image_part(frame)], role=role, tag="perception_boxes", schema=schema)
        if isinstance(response, str):
            response = response.strip()
            if response.startswith("```"):
                response = response.split("\n", 1)[1].rsplit("```", 1)[0]
            response = json.loads(response)
        h, w = frame.shape[:2]
        boxes = {}
        for obj in response["objects"]:
            y0, x0, y1, x1 = obj["box_2d"]
            if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
                raise ValueError(f"Invalid detector box for {obj['id']}")
            if str(obj["id"]) in boxes:
                raise ValueError(f"Duplicate detector id: {obj['id']}")
            boxes[str(obj["id"])] = [x0*w/1000, y0*h/1000, x1*w/1000, y1*h/1000]
        return boxes


def _known_pose(obj):
    if "xy" in obj and not isinstance(obj["xy"], str):
        xyz = [*obj["xy"], dimensions(obj)[2]/2]
    elif obj.get("fixed") and "xyz" in obj:
        xyz = obj["xyz"]
    else:
        return None
    return {"xyz": list(xyz), "yaw_deg": float(obj.get("yaw_deg", 0)),
            "size_m": obj.get("size_m"), "seen": True, "source": "known"}


class TaskPerceiver:
    def __init__(self, cam_model, task, detector=None, *, clock=time.monotonic, arm=None):
        if not isinstance(task, dict):
            from r2p.tasks import load_task
            task = load_task(task)
        self.cam = cam_model
        self.task = copy.deepcopy(task)
        self.objects = {obj["id"]: copy.deepcopy(obj) for obj in task["objects"]}
        for oid, obj in self.objects.items():
            scale = task.get("randomize", {}).get(oid, {}).get("size_scale")
            if isinstance(scale, (list, tuple)) and scale[0] < scale[1]:
                obj["_size_scale_range"] = list(scale)
        self.detector = detector if detector is not None else GeminiDetector()
        self.clock = clock
        self.arm = arm
        self.models = {}
        self.poses = {}
        self.initial = {}
        self.centers = {}
        self.areas = {}
        self.last_seen = {}
        self._setup = False
        self.replay = False
        self.redetect_count = 0
        self.last_redetect = -float("inf")
        self.redetect_error = None
        self._worker = None
        self._pending = None
        self._pending_lock = threading.Lock()
        self._closed = False
        self._held = None
        self._was_closed = False
        self.last_result = None

    def _detect(self, frame, role):
        method = getattr(self.detector, "detect", self.detector)
        boxes = method(frame, self.task, role=role)
        if not isinstance(boxes, dict):
            raise ValueError("Detector must return an object-id to xyxy-pixel-box mapping")
        return boxes

    def _fit_setup(self, frame, boxes):
        indices = lab_indices(frame)
        fitted = {}
        for oid, obj in self.objects.items():
            known = _known_pose(obj)
            if known is not None:
                fitted[oid] = (None, known, None, 0)
                continue
            if oid not in boxes:
                continue
            mask = seed_mask(frame, boxes[oid], obj.get("color"))
            model = Appearance.fit(frame, mask, indices)
            y, x = np.nonzero(mask)
            hint = None
            if is_screwdriver(obj):
                box = np.asarray(boxes[oid], float)
                # The shaft extends beyond the coloured handle: the complete
                # detector box resolves its otherwise ambiguous 180-degree end.
                pixels = np.array([[(box[0]+box[2])/2, (box[1]+box[3])/2], [x.mean(), y.mean()]])
                endpoints = self.cam.deproject(pixels, dimensions(obj)[2]/2)
                direction = endpoints[0]-endpoints[1]
                hint = {"directed_yaw_deg": float(np.rad2deg(np.arctan2(direction[1], direction[0])))}
            pose = fit_pose(self.cam, obj, mask, previous=hint)
            pose.update(seen=pose["fit_px"] < 8, source="vision" if pose["fit_px"] < 8 else "stale")
            fitted[oid] = (model, pose, [float(x.mean()), float(y.mean())], int(mask.sum()))
        return fitted

    def _apply_setup(self, fitted, now):
        for oid, (model, pose, center, area) in fitted.items():
            if self._held is not None and self._held[0] == oid:
                continue
            if model is not None:
                if not self.objects[oid].get("fixed"):
                    self.models[oid] = model
                self.centers[oid] = center
                self.areas[oid] = area
            if self.objects[oid].get("fixed") and pose.get("seen"):
                pose["source"] = "known"
            self.poses[oid] = pose
            self.last_seen[oid] = now

    def setup(self, frame):
        self._validate_frame(frame)
        if self.replay:
            raise RuntimeError("Saved setup loaded; call estimate directly for zero-request replay")
        if self._setup:
            raise RuntimeError("Already set up; create a new perceiver for a new run")
        fitted = self._fit_setup(frame, self._detect(frame, "small"))
        now = self.clock()
        self._apply_setup(fitted, now)
        for oid in self.objects:
            self.last_seen.setdefault(oid, now)
        self.initial = copy.deepcopy(self.poses)
        self._setup = True
        return {"poses": copy.deepcopy(self.poses)}

    def _validate_frame(self, frame):
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("Expected uint8 BGR camera frame")
        if frame.shape[1::-1] != self.cam.image_size:
            raise ValueError("Frame size differs from camera calibration; resize the calibration too")

    def _start_redetect(self, frame, now):
        if (self.replay or self._closed or os.environ.get("R2P_CACHE_ONLY") == "1"
                or self.redetect_count >= 6 or now-self.last_redetect < 8
                or (self._worker is not None and self._worker.is_alive())):
            return
        self.redetect_count += 1
        self.last_redetect = now
        image = frame.copy()
        def work():
            try:
                fitted = self._fit_setup(image, self._detect(image, "main"))
                pending = (fitted, None, now)
            except Exception as exc:
                pending = ({}, f"{type(exc).__name__}: {exc}", now)
            with self._pending_lock:
                if not self._closed:
                    self._pending = pending
        self._worker = threading.Thread(target=work, name="perception-redetect", daemon=True)
        self._worker.start()

    def _gripper(self, q):
        if self.arm is None:
            from r2p.model import Arm
            self.arm = Arm()
        a, b, mid = self.arm.tips(q)
        data = self.arm.fk(q)
        R = data.xmat[self.arm.m.body("gripper").id].reshape(3, 3).copy()
        return mid, R, float(np.linalg.norm(a-b))

    def _support_candidates(self, oid):
        obj = self.objects[oid]
        result = [0.0]
        if oid not in self.poses:
            return result
        xy = np.asarray(self.poses[oid]["xyz"])[:2]
        for sid, support in self.poses.items():
            if sid == oid or not support.get("seen"):
                continue
            other = self.objects[sid]
            size = dimensions(other)
            if np.linalg.norm(xy-np.asarray(support["xyz"])[:2]) > max(size[:2])*.7 + .025:
                continue
            if other.get("shape") == "bowl":
                height = support["xyz"][2] - size[2]/2 + other.get("floor_m", other.get("wall_m", .005))
            else:
                height = support["xyz"][2] + size[2]/2
            if height < self.poses[oid]["xyz"][2] + dimensions(obj)[2] + .05:
                result.append(height)
        return result

    def estimate(self, frame, q):
        if not self._setup:
            raise RuntimeError("Call setup(frame) or load_setup(path) first")
        self._validate_frame(frame)
        started = time.perf_counter()
        now = self.clock()
        with self._pending_lock:
            pending, self._pending = self._pending, None
        if pending is not None:
            fitted, self.redetect_error, captured_at = pending
            fitted = {oid: value for oid, value in fitted.items() if self.last_seen.get(oid, -float("inf")) < captured_at}
            self._apply_setup(fitted, now)
        q = np.asarray(q, float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("q must contain six finite MJCF radians")
        mid, grip_R, gap = self._gripper(q)
        closed = bool(q[5] < .35)
        if self._held is not None and not closed:
            oid = self._held[0]
            pose = self.poses[oid]
            bottoms = self._support_candidates(oid)
            bottom = max((h for h in bottoms if h <= pose["xyz"][2]+.02), default=0.)
            pose["xyz"][2] = bottom + dimensions(self.objects[oid])[2]/2
            angle = np.deg2rad(pose["yaw_deg"])/2
            pose["quat"] = [float(np.cos(angle)), 0., 0., float(np.sin(angle))]
            pose["source"] = "known" if _known_pose(self.objects[oid]) is not None else "released"
            self._held = None
        if closed and not self._was_closed and self._held is None:
            options = [(np.linalg.norm(np.asarray(p["xyz"])-mid), oid) for oid, p in self.poses.items()
                       if p.get("seen") and not self.objects[oid].get("fixed")
                       and self.objects[oid].get("role") not in ("container", "target", "fixture")]
            if options:
                distance, oid = min(options)
                if distance < .025:
                    p = self.poses[oid]
                    self._held = (oid, grip_R.T @ (np.asarray(p["xyz"])-mid), grip_R.T @ rotation(p["yaw_deg"]))
        self._was_closed = closed
        indices = lab_indices(frame)
        masks = {}
        for oid, obj in self.objects.items():
            if self._held is not None and self._held[0] == oid:
                _, offset, local_R = self._held
                R = grip_R @ local_R
                from scipy.spatial.transform import Rotation
                quat = np.roll(Rotation.from_matrix(R).as_quat(), 1).tolist()
                self.poses[oid].update(xyz=(mid + grip_R @ offset).tolist(),
                                       yaw_deg=float((np.rad2deg(np.arctan2(R[1, 0], R[0, 0]))+90)%180-90),
                                       quat=quat, seen=True, source="fk")
                self.centers[oid] = self.cam.project(self.poses[oid]["xyz"]).tolist()
                self.last_seen[oid] = now
                continue
            if _known_pose(obj) is not None or (obj.get("fixed") and oid in self.poses):
                # Explicit known locations are authoritative, never colour tracked.
                self.poses[oid] = self.poses.get(oid) or _known_pose(obj)
                self.last_seen[oid] = now
                continue
            if oid not in self.models:
                continue
            mask = self.models[oid].mask(indices=indices)
            n, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
            choices = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= max(8, .18*self.areas[oid])]
            if choices:
                prev = np.asarray(self.centers[oid])
                choice = min(choices, key=lambda i: np.linalg.norm(centers[i]-prev))
                distance = np.linalg.norm(centers[choice]-prev)
                # Reject distant locks, but allow ordinary 30 Hz motion.
                radius = max(35., 2*np.sqrt(self.areas[oid]))
                if distance > radius:
                    choices = []
            if not choices:
                if oid in self.poses:
                    self.poses[oid].update(seen=False, source="stale")
                continue
            selected = (labels == choice).astype(np.uint8)
            fits = [fit_pose(self.cam, obj, selected, bottom, self.poses.get(oid))
                    for bottom in self._support_candidates(oid)]
            pose = min(fits, key=lambda p: p["fit_px"])
            if pose["fit_px"] > 8:
                self.poses[oid].update(seen=False, source="stale")
                continue
            pose.update(seen=True, source="vision")
            self.poses[oid] = pose
            self.centers[oid] = centers[choice].tolist()
            self.last_seen[oid] = now
            masks[oid] = selected
            if pose["fit_px"] < 2 and distance < max(10, np.sqrt(self.areas[oid])):
                distractors = (labels != 0) & (labels != choice)
                self.models[oid].learn_background(indices, distractors, selected)
        lost = [oid for oid in self.objects if now-self.last_seen.get(oid, now) > 1.0
                and _known_pose(self.objects[oid]) is None
                and not (self.objects[oid].get("fixed") and oid in self.poses)
                and (oid in self.poses or self.task.get("randomize", {}).get(oid, {}).get("p_present", 1) >= 1)]
        if lost:
            self._start_redetect(frame, now)
        result = {"poses": copy.deepcopy(self.poses), "masks": masks, "lost": lost,
                  "gripper_open": not closed, "site_xyz": mid.tolist(), "jaw_gap_m": gap,
                  "holding": [] if self._held is None else [self._held[0]],
                  "redetect_count": self.redetect_count, "redetect_error": self.redetect_error,
                  "latency_ms": 1000*(time.perf_counter()-started)}
        self.last_result = result
        return result

    def check(self, result=None, **state):
        from r2p.tasks import check
        result = self.last_result if result is None else result
        if result is None:
            raise RuntimeError("Estimate a frame before checking success")
        valid = {oid: p for oid, p in result["poses"].items() if p.get("seen")}
        return check(self.task, {"objects": valid, "gripper_open": result.get("gripper_open", True),
                                 "site_xyz": result.get("site_xyz", [0, 0, 0]),
                                 "holding": result.get("holding", []), "initial": self.initial, **state})

    def save_setup(self, path):
        if not self._setup:
            raise RuntimeError("No setup to save")
        payload = {"schema": "r2p_perception_setup/1", "camera": self.cam.to_dict(), "task": self.task,
                   "models": {oid: model.to_dict() for oid, model in self.models.items()},
                   "poses": self.poses, "centers": self.centers, "areas": self.areas, "initial": self.initial}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def load_setup(self, path):
        if self._worker is not None:
            raise RuntimeError("Load saved setup into a fresh perceiver")
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != "r2p_perception_setup/1":
            raise ValueError("Unsupported saved setup schema")
        if payload["task"] != self.task or payload["camera"] != self.cam.to_dict():
            raise ValueError("Saved setup task or calibration differs from this perceiver")
        self.models = {oid: Appearance.from_dict(model) for oid, model in payload["models"].items()}
        self.poses, self.centers, self.areas = payload["poses"], payload["centers"], payload["areas"]
        self.initial = payload.get("initial", copy.deepcopy(self.poses))
        self.last_seen = dict.fromkeys(self.objects, self.clock())
        self.replay = self._setup = True
        return self

    def close(self):
        self._closed = True
        if self._worker is not None:
            self._worker.join(timeout=.1)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
