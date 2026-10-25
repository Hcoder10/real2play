"""Dual-camera recording and survey guards for the NEXT real campaign.

No connection is made on import. Construct around an explicitly supplied mover;
only that mover owns the actuator write boundary. Tests inject all IO.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import threading
import time
from urllib.request import urlopen

import cv2
import numpy as np

from r2p import gemini
from .active import UnsafeExperiment, http_json
from .loop import atomic_image


def http_capture(url="http://127.0.0.1:8794", *, request=http_json):
    def capture():
        now = time.monotonic()
        listing = request(url.rstrip("/")+"/list")["0"]
        age = float(listing["age_s"])
        if not np.isfinite(age) or not 0 <= age <= 2.5:
            raise UnsafeExperiment("Wrist camera is stale")
        with urlopen(url.rstrip("/")+"/frame/0.jpg", timeout=3) as response:
            image = cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise UnsafeExperiment("Wrist frame cannot be decoded")
        return image, now-age, {"age_s": age, "sequence": listing.get("frames"),
                                "timestamp_source": "camserver_age_estimate", "received_at": time.monotonic()}
    return capture


class DualCameraRecorder:
    """Persist frames as they arrive, including on abort; monotonic clock shared."""
    def __init__(self, destination, scene_capture, wrist_capture, *, hz=5):
        self.path = Path(destination)
        self.capture = {"scene": scene_capture, "wrist": wrist_capture}
        self.rows = {"scene": [], "wrist": []}
        self.errors = []
        self.hz = hz
        self.stop = threading.Event()
        self.workers = []
        self.origin = time.monotonic()
        self.last_received = {}

    def _one(self, name):
        frame, timestamp, metadata = self.capture[name]()
        now = time.monotonic()
        if not np.isfinite(timestamp) or timestamp > now+.05 or now-timestamp > 2.5:
            raise UnsafeExperiment(f"{name} camera timestamp invalid/stale")
        rows = self.rows[name]
        if rows and (timestamp <= rows[-1]["captured_at"] or
                     metadata.get("sequence") is not None and metadata["sequence"] == rows[-1].get("sequence")):
            return
        filename = f"{name}/{len(rows):06d}.jpg"
        atomic_image(self.path/filename, frame)
        row = {**metadata, "file": filename, "captured_at": float(timestamp),
               "t": float(timestamp-self.origin), "received_at": now}
        rows.append(row); self.last_received[name] = now
        with (self.path/f"{name}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row)+"\n")

    def start(self):
        self.path.mkdir(parents=True, exist_ok=False)
        # Both cameras must work before any motion is allowed.
        for name in self.capture:
            self._one(name)
        def poll(name):
            while not self.stop.wait(1/self.hz):
                try:
                    self._one(name)
                except Exception as exc:
                    self.errors.append(f"{name}: {exc}"); self.stop.set()
        self.workers = [threading.Thread(target=poll, args=(name,), daemon=True) for name in self.capture]
        for worker in self.workers:
            worker.start()

    def check(self):
        if self.errors:
            raise UnsafeExperiment("Dual-camera recording failed: "+"; ".join(self.errors))
        if any(time.monotonic()-self.last_received.get(name, -1e30) > 2.5 for name in self.capture):
            raise UnsafeExperiment("Dual-camera stream stalled")

    def close(self):
        self.stop.set()
        for worker in self.workers:
            worker.join(timeout=4)
        gemini.atomic_json(self.path/"cameras.json", {"schema": "r2p_dual_camera/1", "origin_monotonic": self.origin,
            "frames": self.rows, "errors": self.errors, "clock": "shared host monotonic; estimated sensor capture times"})


class SurveyedMover:
    """Re-survey every experiment, guard the sheet, record scene AND wrist.

    survey(frame, q) returns independently observed poses. sheet_geometry(frame)
    returns labelled fixed paper landmarks (at least three XY points); compare
    all of them so a rotation about the centre cannot slip past the guard.
    """
    def __init__(self, mover, *, survey, sheet_geometry, wrist_capture=None):
        self.mover = mover
        self.survey = survey
        self.sheet_geometry = sheet_geometry
        self.wrist_capture = wrist_capture or http_capture()
        self.reference = None

    def __getattr__(self, name):
        return getattr(self.mover, name)

    def _landmarks(self, frame):
        landmarks = self.sheet_geometry(frame)
        if not isinstance(landmarks, dict) or len(landmarks) < 3:
            raise UnsafeExperiment("Sheet guard needs at least three visible labelled landmarks")
        if any(np.asarray(p).shape != (2,) or not np.isfinite(p).all() for p in landmarks.values()):
            raise UnsafeExperiment("Invalid sheet landmarks")
        return {k: list(p) for k, p in landmarks.items()}

    def _guard(self, frame):
        points = self._landmarks(frame)
        if self.reference is None:
            self.reference = points
        if points.keys() != self.reference.keys():
            raise UnsafeExperiment("Sheet landmarks disappeared")
        displacement = max(np.linalg.norm(np.asarray(v)-self.reference[k]) for k, v in points.items())
        if displacement > .004:
            raise UnsafeExperiment(f"Sheet moved {displacement*1000:.1f} mm")
        return float(displacement)

    def prepare(self):
        frame, _, _ = self.mover.capture()
        self._guard(frame)
        state = self.mover.request(self.mover.url+"/state")
        poses = self.survey(frame, self.mover.measured_q(state))
        proposed = deepcopy(self.mover.scene)
        for obj in proposed["objects"]:
            if obj.get("fixed"):
                continue
            pose = poses.get(obj["id"], {})
            xyz = np.asarray(pose.get("xyz", []), float)
            if not pose.get("seen") or pose.get("source") != "vision" or xyz.shape != (3,) or not np.isfinite(xyz).all():
                raise UnsafeExperiment(f"Fresh survey missing {obj['id']}")
            obj["xyz"] = xyz.tolist()
            obj["yaw_deg"] = float(pose.get("yaw_deg", obj.get("yaw_deg", 0)))
        self.mover.scene = proposed

    def execute(self, experiment, destination, *, split="train"):
        self.prepare()
        recorder = DualCameraRecorder(Path(destination).with_name(Path(destination).name+"_cameras"),
                                      self.mover.capture, self.wrist_capture)
        original_request = self.mover.request
        measured = None
        def guarded_request(url, payload=None, **kwargs):
            nonlocal measured
            if payload is not None and not payload.get("hold"):
                recorder.check()
                if measured is None:
                    raise UnsafeExperiment("No measured jaw evidence before motion")
                jaw = self.validator.arm.tips(measured)[2]
                if jaw[2] < self.validator.safety.live_min_jaw_z:
                    raise UnsafeExperiment(f"Live abort: measured jaw {jaw[2]*1000:.1f} mm")
            reply = original_request(url, payload, **kwargs)
            if payload is None and url.endswith("/state"):
                measured = self.mover.measured_q(reply)
            return reply
        observation = None
        try:
            recorder.start()
            self.mover.request = guarded_request
            observation = self.mover.execute(experiment, destination, split=split)
            recorder.check()
            return observation
        finally:
            self.mover.request = original_request
            recorder.close()
            # Also check after a failed/aborted experiment; raw camera evidence is
            # already on disk even if either guard raises.
            displacement = self._guard(self.mover.capture()[0])
            gemini.atomic_json(recorder.path/"sheet_guard.json", {"max_displacement_m": displacement, "ok": True})
            if observation is not None:
                raw_path = Path(destination).with_name(Path(destination).name+"_raw.json")
                raw = json.loads(raw_path.read_text()) if raw_path.exists() else {}
                captures = raw.get("captures", [])
                origin = None
                if captures:
                    first = captures[0]
                    absolute = first["received_at"]-first.get("age_s", 0)
                    origin = absolute-first["t"]
                    sensors = raw.get("sensors", [])
                    if sensors:
                        origin += sensors[0]["t"]
                meta = {"dual_camera_path": str(recorder.path.resolve()), "trace_origin_monotonic_estimate": origin}
                gemini.atomic_json(Path(destination)/"meta.json", meta)
                observation.session.meta.update(meta)

    def wrist_parts(self, observation):
        if observation.split != "train":
            raise ValueError("Held-out wrist images cannot enter MAIN prompts")
        root = Path(observation.session.meta["dual_camera_path"])
        manifest = json.loads((root/"cameras.json").read_text())
        rows = manifest["frames"]["wrist"]
        if not rows:
            return ()
        origin = observation.session.meta.get("trace_origin_monotonic_estimate")
        chosen = []
        # Match close/lift/release events to wrist timestamps. When old traces
        # lack phases, provide explicitly timed coverage without event labels.
        for phase in ("close", "lift", "release"):
            indices = [i for i, p in enumerate(observation.plan.phases) if p == phase]
            if indices and origin is not None:
                t = origin + indices[len(indices)//2]/30
                chosen.append((phase, min(rows, key=lambda r: abs(r["captured_at"]-t))))
        if not chosen:
            chosen = [("timed wrist observation", rows[i]) for i in np.linspace(0, len(rows)-1, 3, dtype=int)]
        parts = ["Training wrist camera: inspect grasp, slip and release; it is not calibrated scene geometry."]
        for label, row in chosen:
            image = cv2.imread(str(root/row["file"]))
            if image is not None:
                parts += [f"{label}, capture monotonic {row['captured_at']:.3f}", gemini.image_part(image)]
        return parts
