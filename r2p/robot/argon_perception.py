"""Argon cap surveys outside the control loop, with contact-gated rigid FK tracking.

``camera`` is a calibrated CameraModel. Call setup(frame) before starting motion;
the first estimate also seeds synchronously if needed. All subsequent estimates
only do local work. Bind commanded_q to a cheap getter of the last accepted MJCF
target (for example ``lambda: session.last_sent``); it must not perform I/O.
"""
from copy import deepcopy
import threading
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from r2p.robot.io import vector
from r2p.tasks import slot_size


RIM_HEIGHT_M = .017
CAP_PROMPT = (
    "Find every white plastic bottle cap (open side up) on the table. For each, "
    "point to the centre of its TOP rim ellipse as [y, x] normalized 0-1000, "
    "and say whether it sits on the printed 3x3 grid."
)
CAP_SCHEMA = {"type": "object", "required": ["caps"], "properties": {
    "caps": {"type": "array", "items": {"type": "object",
        "required": ["rim_center", "on_grid"], "properties": {
            "rim_center": {"type": "array", "minItems": 2, "maxItems": 2,
                           "items": {"type": "number", "minimum": 0, "maximum": 1000}},
            "on_grid": {"type": "boolean"}}}}}}


class ArgonPerceiver:
    """TaskPerceiver's estimate(frame, measured_q) contract for one supply cap.

    A squeeze target below the measured gripper angle AND a cap between nearby
    pads admits an attachment. This is a contact heuristic, not a force sensor.
    After separation, the last attached pose is retained until a fresh survey.
    Masks are projected cap silhouettes, not colour-segmentation measurements.
    """

    def __init__(self, camera, task, *, ask=None, arm=None, commanded_q=None,
                 refresh_s=4., clock=time.monotonic):
        from r2p.model import Arm
        self.cam, self.task = camera, deepcopy(task)
        pieces = [o for o in self.task["objects"] if o["role"] == "manipulated"]
        if len(pieces) != 1 or pieces[0]["shape"] != "cylinder":
            raise ValueError("ArgonPerceiver requires one manipulated bottle cap (cylinder)")
        self.piece = pieces[0]
        self.piece_id = self.piece["id"]
        self.size = np.asarray(slot_size(self.piece["shape"], self.piece["size_m"]), float)
        near = self.piece.get("xyz", self.piece.get("xy"))
        if near is None or isinstance(near, str):
            raise ValueError("The task must specify the supply cap's approximate xy")
        self.near = np.asarray(near, float)[:2]
        if self.near.shape != (2,) or not np.isfinite(self.near).all():
            raise ValueError("Supply xy must be finite")
        self.refresh_s = float(refresh_s)
        if not np.isfinite(self.refresh_s) or self.refresh_s <= 0:
            raise ValueError("refresh_s must be finite and positive")
        self.arm, self.ask, self.clock = arm or Arm(), ask, clock
        self.commanded_q = commanded_q
        self.poses = self._fixed_poses()
        self.initial = {}
        self.last_result = None
        self.last_record = None
        self.redetect_error = None
        self.redetect_count = 0
        self._setup = False
        self._closed = False
        self._held = None
        self._released = False
        self._generation = 0
        self._last_request = -float("inf")
        self._last_seen = None
        self._worker = None
        self._pending = None
        self._pending_lock = threading.Lock()
        self._fatal_error = None
        self._rotation = np.eye(3)

    def _fixed_poses(self):
        """Use surveyed task geometry verbatim; never sample target randomization."""
        poses = {}
        board = self.task.get("board", {}).get("cells", {})
        for obj in self.task["objects"]:
            if not obj.get("fixed"):
                continue
            presence = self.task.get("randomize", {}).get(obj["id"], {}).get("p_present", obj.get("p_present", 1))
            if presence == 0:
                continue
            height = slot_size(obj["shape"], obj["size_m"])[2]
            cell = self.task.get("target_cell") if obj["id"] == "cell" else obj["id"]
            if cell in board:
                xyz = [*board[cell], height/2]
            elif "xyz" in obj:
                xyz = obj["xyz"]
            elif "xy" in obj and not isinstance(obj["xy"], str):
                xyz = [*obj["xy"], height/2]
            else:
                raise ValueError(f"Fixed target {obj['id']} needs a surveyed task position")
            poses[obj["id"]] = {"xyz": list(xyz), "yaw_deg": float(obj.get("yaw_deg", 0)),
                                "size_m": deepcopy(obj["size_m"]), "seen": True, "source": "measured_task"}
        return poses

    def _validate_frame(self, frame):
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("Expected uint8 BGR camera frame")
        if frame.shape[1::-1] != self.cam.image_size:
            raise ValueError("Frame size differs from camera calibration")

    def _survey(self, frame, near, supply_only):
        from r2p import gemini
        response, record = (self.ask or gemini.call)(
            [gemini.image_part(frame), CAP_PROMPT], role="main", thinking="low",
            tag="robot:cap_perception", schema=CAP_SCHEMA)
        response = gemini.parse_json(response) if isinstance(response, str) else response
        errors = gemini.schema_errors(response, CAP_SCHEMA)
        if errors:
            raise ValueError("Invalid Argon cap observation: " + "; ".join(errors))
        h, w = frame.shape[:2]
        cells = self.task.get("board", {}).get("cells", {})
        edge = (min(xy[1] for xy in cells.values()) - self.task["board"]["cell_m"]/2 - .008
                if cells else float("inf"))
        candidates = []
        for cap in response["caps"]:
            if supply_only and cap["on_grid"]:
                continue
            y, x = cap["rim_center"]
            xy = self.cam.deproject(np.array([x*w/1000, y*h/1000]), h=RIM_HEIGHT_M)[:2]
            if supply_only and not (.09 < xy[0] < .25 and -.17 < xy[1] < edge):
                continue
            candidates.append(xy)
        if not candidates:
            raise ValueError("Argon cannot see a supply cap" if supply_only else "Argon cannot see the released cap")
        xy = min(candidates, key=lambda p: np.linalg.norm(p-near))
        # A refresh must not jump to the next cap/cell if the selected cap is occluded.
        if np.linalg.norm(xy-near) > (.08 if not self._setup else .03):
            raise ValueError("Argon cap observation is too far from the tracked cap")
        pose = {"xyz": [float(xy[0]), float(xy[1]), float(self.size[2]/2)],
                "yaw_deg": 0., "size_m": deepcopy(self.piece["size_m"]), "source": "vision", "seen": True}
        return pose, record

    def setup(self, frame):
        """One blocking Argon survey, intended for the no-motion preflight."""
        if self._closed:
            raise RuntimeError("Perceiver is closed")
        if self._setup:
            raise RuntimeError("Already set up; create a new perceiver for a new run")
        self._validate_frame(frame)
        pose, self.last_record = self._survey(frame, self.near, True)
        self.poses[self.piece_id] = pose
        self.initial = deepcopy(self.poses)
        self._last_request = self._last_seen = self.clock()
        self._setup = True
        return {"poses": deepcopy(self.poses), "masks": {self.piece_id: self._mask(frame)}}

    def set_commanded_q(self, q):
        """Set an accepted absolute target, in MJCF radians, without touching hardware."""
        self.commanded_q = vector(q)

    def _start_refresh(self, frame, now):
        if (self._closed or self._held is not None or now-self._last_request < self.refresh_s
                or (self._worker is not None and self._worker.is_alive())):
            return
        # Copy only when a request is due, always from this tick's latest frame.
        image = frame.copy()
        near = np.asarray(self.poses[self.piece_id]["xyz"][:2], float)
        generation, supply_only = self._generation, not self._released
        self._last_request = now
        self.redetect_count += 1

        def work():
            try:
                pose, record = self._survey(image, near, supply_only)
                pending = (generation, now, pose, record, None)
            except Exception as exc:
                pending = (generation, now, None, None, exc)
            with self._pending_lock:
                if not self._closed:
                    self._pending = pending

        self._worker = threading.Thread(target=work, name="argon-cap-survey", daemon=True)
        self._worker.start()

    def _apply_pending(self):
        from r2p import gemini
        with self._pending_lock:
            pending, self._pending = self._pending, None
        if pending is None:
            return
        generation, captured_at, pose, record, error = pending
        if isinstance(error, (gemini.QuotaError, gemini.CacheMiss, gemini.RequestLimit)):
            # Preserve the central client's no-fallback/cache-only failure semantics.
            self._fatal_error = error
            raise error
        if generation != self._generation or self._held is not None:
            return
        self.redetect_error = None if error is None else f"{type(error).__name__}: {error}"
        if pose is not None:
            self.poses[self.piece_id] = pose
            self._rotation = np.eye(3)
            self._last_seen, self.last_record = captured_at, record

    def _track_contact(self, q, target):
        a, b, mid = self.arm.tips(q)
        grip_R = self.arm.fk(q).xmat[self.arm.m.body("gripper").id].reshape(3, 3).copy()
        gap = float(np.linalg.norm(b-a))
        pose = self.poses[self.piece_id]
        if self._held is not None:
            offset, local_R, grasp_gap = self._held
            opening = target is not None and target[5] > q[5] + np.deg2rad(2)
            if gap > self.size[0]+.004 or (opening and gap > grasp_gap+.002):
                # Freeze the last contact pose, without following the separating jaw
                # or guessing a landing height. A post-release image confirms it.
                self._held = None
                self._released = True
                self._generation += 1
                self._last_request = -float("inf")
                pose["source"] = "released"
            else:
                self._rotation = grip_R @ local_R
                pose.update(xyz=(mid + grip_R @ offset).tolist(), source="vision+fk", seen=True,
                            yaw_deg=float(np.rad2deg(np.arctan2(self._rotation[1, 0], self._rotation[0, 0]))),
                            quat=np.roll(Rotation.from_matrix(self._rotation).as_quat(), 1).tolist())
        elif target is not None and q[5]-target[5] > np.deg2rad(4) and gap > 1e-8:
            delta = np.asarray(pose["xyz"])-mid
            along = float(np.dot(delta, (b-a)/gap))
            lateral = np.linalg.norm(delta-along*(b-a)/gap)
            if (self.size[0]-.006 <= gap <= self.size[0]+.004
                    and abs(along) <= gap/2 and lateral < self.size[2]/2+.006):
                self._held = (grip_R.T @ delta, grip_R.T @ self._rotation, gap)
                self._generation += 1
                pose.update(source="vision+fk", seen=True)
        return mid, gap

    def _mask(self, frame):
        xyz = np.asarray(self.poses[self.piece_id]["xyz"])
        angles = np.linspace(0, 2*np.pi, 32, endpoint=False)
        ring = np.column_stack((self.size[0]/2*np.cos(angles), self.size[1]/2*np.sin(angles), np.zeros(32)))
        points = np.concatenate((ring-[0, 0, self.size[2]/2], ring+[0, 0, self.size[2]/2]))
        pixels = self.cam.project(xyz + points @ self._rotation.T)
        mask = np.zeros(frame.shape[:2], np.uint8)
        if np.isfinite(pixels).all():
            hull = cv2.convexHull(np.clip(pixels, -100000, 100000).astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 1)
        return mask.astype(bool)

    def estimate(self, frame, q):
        started = time.perf_counter()
        if self._closed:
            raise RuntimeError("Perceiver is closed")
        if self._fatal_error is not None:
            raise self._fatal_error
        self._validate_frame(frame)
        q = vector(q)
        if not self._setup:
            self.setup(frame)
        target = self.commanded_q() if callable(self.commanded_q) else self.commanded_q
        target = None if target is None else vector(target)
        mid, gap = self._track_contact(q, target)
        self._apply_pending()
        now = self.clock()
        self._start_refresh(frame, now)
        result = {"poses": deepcopy(self.poses), "masks": {self.piece_id: self._mask(frame)},
                  "holding": [self.piece_id] if self._held is not None else [],
                  "gripper_open": self._held is None and gap > self.size[0]+.004,
                  "site_xyz": mid.tolist(), "jaw_gap_m": gap, "lost": [],
                  "redetect_count": self.redetect_count, "redetect_error": self.redetect_error,
                  "pose_age_s": now-self._last_seen, "latency_ms": 1000*(time.perf_counter()-started)}
        self.last_result = result
        return result

    def close(self):
        # A slow API request must not delay a stop or write into a closed perceiver.
        with self._pending_lock:
            self._closed = True
            self._pending = None
        if self._worker is not None:
            self._worker.join(timeout=.05)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
