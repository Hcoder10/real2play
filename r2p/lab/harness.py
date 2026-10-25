"""Loopback HTTP harness, real daemon adapter, and timestamped experiment capture."""
from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import threading
import time
from types import FunctionType
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np

from r2p import gemini
from r2p.frames import ROOT
from r2p.model import Arm, JointMap
from r2p.perception.camera import CameraModel
from .motion import Motion, Refused, StallDetector, aim_with_jaw_ik, vector
from .media import timeline_video


def http_json(url, body=None, timeout=3):
    req = Request(url, data=json.dumps(body).encode() if body is not None else None,
                  headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as response:
            result = json.load(response)
    except HTTPError as exc:
        try:
            result = json.loads(exc.read())
        except ValueError:
            result = {"error": str(exc)}
    if result.get("ok") is False or any(result.get(k) for k in ("error", "refused", "abort")):
        raise Refused("armd refusal: "+str(result))
    return result


def bind(function, **globals_override):
    """Adapt proven scripts without changing their module globals or source files."""
    result = FunctionType(function.__code__, {**function.__globals__, **globals_override},
                          function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


class RealBackend:
    def __init__(self, url="http://127.0.0.1:8793", scene_url="http://127.0.0.1:8792",
                 wrist_url="http://127.0.0.1:8794", request=http_json):
        from scripts.lead import jaw_move
        from scripts.lead.ttt_armd import park
        self.url, self.request = url, request
        self.camera_urls = {"scene": scene_url, "wrist": wrist_url}
        self.jm = JointMap.load(ROOT/"config/joint_map.json")
        self.arm, self.monitor_arm = Arm(), Arm()
        self.scene_camera = CameraModel.load(ROOT/"config/camera.json")
        self.camera_sizes = {"scene": self.scene_camera.image_size}
        self.motion, self.speed_limit = None, 12
        self._stop = threading.Event()
        self._goto = bind(jaw_move.goto, armd=self.armd)
        self._gripper = bind(jaw_move.gripper, armd=self.armd)
        self._park = bind(park, armd=self.armd, goto=self._park_goto)
        self._resting = False
        self.rest_phase = False
        self.on_command = None
        self.stall = StallDetector()
        st = self.request(self.url+"/state")
        if "owner_deg" not in st or "q_target" not in st:
            raise Refused("Lab requires the repository streaming armd with q_target telemetry; legacy daemon refused")

    def state(self):
        st = self.request(self.url+"/state")
        q, target = vector(st["q"], 6), vector(st["q_target"], 6)
        return {**st, "q": q.tolist(), "q_target": target.tolist(),
                "xyz": vector(st["xyz"]).tolist(), "gripper": float(st["owner_deg"][5]),
                "holding": self.stall.update(q[5], target[5], time.monotonic()), "source": "real"}

    def camera(self, name="scene"):
        if name == "scene":
            return self.scene_camera.resized(self.camera_sizes.get(name, self.scene_camera.image_size))
        # A wrist camera must be calibrated before metric deprojection or look_at.
        path = ROOT/"config/wrist_camera.json"
        if not path.exists():
            raise Refused("Wrist extrinsic/intrinsic calibration is unavailable; raw wrist observation still works")
        data = json.loads(path.read_text())
        if data.get("mount_body") and data.get("mount_R") is not None and data.get("mount_pos") is not None:
            from r2p.frames import BASE_ORIGIN_MJCF
            self.arm.fk(self.state()["q"])
            b = self.arm.m.body(data["mount_body"]).id
            body_R = self.arm.d.xmat[b].reshape(3, 3)
            position = self.arm.d.xpos[b]-BASE_ORIGIN_MJCF+body_R@vector(data["mount_pos"])
            R = np.asarray(data["mount_R"], float) @ body_R.T
            camera = CameraModel(data["K"], data.get("dist", [0]*5), R, -R@position, data["image_size"])
            return camera.resized(self.camera_sizes.get(name, camera.image_size))
        raise Refused("Wrist calibration must declare mount_body, mount_R and mount_pos")

    def capture(self, name="scene"):
        started = time.monotonic()
        listing = self.request(self.camera_urls[name]+"/list")["0"]
        age = float(listing["age_s"])
        if not math.isfinite(age) or not 0 <= age <= 2.5:
            raise Refused(f"{name} camera is stale")
        with urlopen(self.camera_urls[name]+"/frame/0.jpg", timeout=3) as response:
            raw = response.read()
            header_age = response.headers.get("X-Frame-Age")
            if header_age is not None:
                age = float(header_age)
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if frame is None or age > 3:
            raise Refused(f"{name} camera missing/stale")
        self.camera_sizes[name] = frame.shape[1::-1]
        return frame, started-age, {"age_s": age, "sequence": listing.get("frames"),
                                   "timestamp_source": "camserver_age_estimate", "received_monotonic": time.monotonic()}

    def frame(self, name="scene"):
        return self.capture(name)[0]

    def _send_joints(self, target):
        sent = time.monotonic()
        state = self.request(self.url+"/cmd", {"joints": np.asarray(target).tolist(), "stream": True})
        if self.on_command:
            self.on_command(state.get("q_target", np.asarray(target).tolist()), sent)
        return state

    def armd(self, path, body=None):
        from scripts.lead.jaw_move import OWNER_LO, OWNER_HI
        st = self.state()
        if body is None:
            return {"q": st["owner_deg"], "lo": OWNER_LO, "hi": OWNER_HI, "modern": True}
        cmd = body["cmd"]
        if cmd == "joints":
            target = self.jm.to_model(vector(body["owner"], 6))
        elif cmd == "gripper":
            own = np.asarray(st["owner_deg"], float)
            own[5] = body["value"]
            target = self.jm.to_model(own)
        elif cmd == "rest":
            target = vector(st["rest_q"], 6)
            self.rest_phase = True
        else:
            raise Refused("Unsupported daemon command")
        start = vector(st["q_target"], 6)
        speed = min(self.speed_limit, 8 if st["xyz"][2] < .045 else 20)
        if cmd == "rest":
            speed = min(speed, 8)
        n = max(1, math.ceil(np.max(np.abs(target-start))/np.deg2rad(speed)*30))
        for q in np.linspace(start, target, n+1)[1:]:
            if self._stop.is_set():
                raise Refused("Live abort requested")
            begin = time.monotonic()
            if self.motion:
                self.motion.check_joints(q, self.arm)
                self.motion.check_joints(self.state()["q"], self.arm, measured=True)
            st = self._send_joints(q)
            if self._stop.wait(max(0, 1/30-(time.monotonic()-begin))):
                raise Refused("Live abort requested")
        # Keep all sends streaming so state polling cannot be blocked by a long daemon move.
        for _ in range(15):
            st = self.state()
            if np.max(np.abs(np.asarray(st["q_target"])-target)) < 1e-4:
                break
            if self._stop.is_set():
                raise Refused("Live abort requested")
            current = np.asarray(st["q_target"])
            self._send_joints(current+np.clip(target-current, -np.deg2rad(speed)/30, np.deg2rad(speed)/30))
            self._stop.wait(1/30)
        st = self.state()
        if np.max(np.abs(np.asarray(st["q_target"])-target)) > 1e-4:
            raise Refused("Daemon did not reach the guarded final command")
        return {"ok": True, "q": st["owner_deg"], "lo": OWNER_LO, "hi": OWNER_HI}

    def goto(self, xyz, *, speed=12, min_finger=.006):
        self.speed_limit = speed
        iterations = 1 if self.motion and self.motion.phase.endswith("until_contact") else 3
        return self._goto(xyz, min_finger=min_finger, iters=iterations)

    def _park_goto(self, xyz, grip=None, **kw):
        return self._goto(xyz, grip, min_finger=.006, iters=1)

    def gripper(self, value):
        if not math.isfinite(value) or not 0 <= value <= 100:
            raise Refused("Gripper must be 0..100")
        return self._gripper(value)

    def park(self):
        try:
            return self._park()
        finally:
            self.rest_phase = False

    def stop(self):
        self._stop.set()
        # stop is lock-free in the repo daemon; hold follows after the in-flight tick releases its lock.
        try:
            self.request(self.url+"/cmd", {"stop": True})
        finally:
            self.request(self.url+"/cmd", {"hold": True})

    def dwell(self, seconds):
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            if self._stop.wait(min(1/30, max(0, deadline-time.monotonic()))):
                raise Refused("Live abort requested")
            state = self.state()
            if self.motion:
                self.motion.check_joints(state["q"], self.arm, measured=True)
        return {"held_s": seconds, "holding": self.state()["holding"]}

    def look_at(self, xyz):
        # Use the same roll-locked arm geometry and the measured wrist camera mount.
        self.camera("wrist")  # fail closed before attempting any movement
        data = json.loads((ROOT/"config/wrist_camera.json").read_text())
        from r2p.frames import BASE_ORIGIN_MJCF
        def optical_axis(q):
            self.arm.fk(q)
            b = self.arm.m.body(data["mount_body"]).id
            Rb = self.arm.d.xmat[b].reshape(3, 3)
            origin = self.arm.d.xpos[b]-BASE_ORIGIN_MJCF+Rb@np.asarray(data["mount_pos"])
            R = np.asarray(data["mount_R"])@Rb.T
            return origin, R[2]
        return aim_with_jaw_ik(self, vector(xyz), optical_axis)


def annotate(image, camera):
    frame = image.copy()
    cam = camera.resized(frame.shape[1::-1])
    def line(a, b, color, width=1):
        uv = cam.project(np.asarray([a, b], float))
        if np.isfinite(uv).all():
            cv2.line(frame, tuple(np.rint(uv[0]).astype(int)), tuple(np.rint(uv[1]).astype(int)), color, width, cv2.LINE_AA)
    for x in np.arange(.1, .341, .02):
        line([x, -.2, 0], [x, .2, 0], (120, 120, 120))
        uv = cam.project([x, -.18, 0]).round().astype(int)
        cv2.putText(frame, f"{x*100:.0f}", tuple(uv), cv2.FONT_HERSHEY_SIMPLEX, .32, (80, 80, 255), 1)
    for y in np.arange(-.2, .201, .02):
        line([.1, y, 0], [.34, y, 0], (120, 120, 120))
        uv = cam.project([.12, y, 0]).round().astype(int)
        cv2.putText(frame, f"{y*100:.0f}", tuple(uv), cv2.FONT_HERSHEY_SIMPLEX, .32, (80, 255, 80), 1)
    for xyz, label, color in [([.10, 0, 0], "+X", (0, 0, 255)), ([0, .10, 0], "+Y", (0, 255, 0)),
                              ([0, 0, .10], "+Z", (255, 0, 0))]:
        line([0, 0, 0], xyz, color, 2)
        cv2.putText(frame, label, tuple(cam.project(xyz).round().astype(int)), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
    cv2.putText(frame, "SPEC FRAME | grid cm | +X away, +Y left, +Z up", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    return frame


class Recorder:
    def __init__(self, directory, backend, motion):
        self.directory, self.backend, self.motion = Path(directory), backend, motion
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rows, self.frames, self.errors = [], [], []
        self.commands = []
        self.origin = time.monotonic()
        self.sim_origin = float(backend.state().get("t", 0))
        self.stop_event = threading.Event()
        self.threads = []
        self.counts = {"scene": 0, "wrist": 0}
        self.capture_locks = {name: threading.Lock() for name in self.counts}
        self.latest_captures = {}
        self.stream = (self.directory/"states.jsonl").open("w", encoding="utf-8")

    def state(self, state=None):
        st = state or self.backend.state()
        t = float(st.get("t", 0))-self.sim_origin if hasattr(self.backend, "on_tick") else time.monotonic()-self.origin
        if self.rows and t <= self.rows[-1]["t"]:
            return
        row = {"t": t, "wall_time": time.time(), "q": st["q"], "q_target": st["q_target"], "xyz": st["xyz"],
               "phase": self.motion.phase, "holding": st.get("holding", False)}
        self.rows.append(row)
        self.stream.write(json.dumps(row)+"\n")
        self.stream.flush()
        if t > self.motion.safety.max_duration_s:
            raise Refused("Episode exceeded the 150-second motion budget")
        if not hasattr(self.backend, "on_tick"):
            self.motion.check_joints(st["q"], self.backend.monitor_arm, measured=True)

    def capture(self, camera, timestamp=None):
        with self.capture_locks[camera]:
            return self._capture(camera)

    def _capture(self, camera):
        if hasattr(self.backend, "capture"):
            frame, captured, metadata = self.backend.capture(camera)
            t = captured-self.origin
        else:
            frame = self.backend.frame(camera)
            t = float(self.backend.state().get("t", 0))-self.sim_origin
            metadata = {"source": "simulation", "captured_t": t}
        previous = self.latest_captures.get(camera)
        if previous and (t <= previous["t"] or metadata.get("sequence") is not None and
                         metadata["sequence"] == previous.get("sequence")):
            return frame
        n = self.counts[camera]
        self.counts[camera] += 1
        path = Path("frames")/camera/f"{n:06d}.jpg"
        (self.directory/path).parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(self.directory/path), frame):
            raise OSError("Cannot save frame")
        row = {"t": t, "camera": camera, "path": path.as_posix(), **metadata}
        self.frames.append(row)
        self.latest_captures[camera] = row
        return frame

    def verify_after(self, completed):
        """A delayed camera frame is not evidence of a completed action."""
        if hasattr(self.backend, "on_tick"):
            self.capture("scene")
            self.capture("wrist")
            return
        deadline = time.monotonic()+6
        while True:
            for name in ("scene", "wrist"):
                self.capture(name)
            if all(row["t"]+self.origin >= completed for row in self.latest_captures.values()) and len(self.latest_captures) == 2:
                return
            if self.errors or time.monotonic() > deadline:
                raise Refused("No fresh dual-camera evidence after action; stopping for re-observation")
            self.stop_event.wait(.05)

    def tick(self, st):
        self.state(st)
        # Sim traces retain 30 Hz states and a timestamped 10 Hz video for both cameras.
        if len(self.rows) % 3 == 0:
            self.capture("scene")
            self.capture("wrist")

    def command(self, q, timestamp):
        origin = self.sim_origin if hasattr(self.backend, "on_tick") else self.origin
        self.commands.append({"t": timestamp-origin, "q_target": list(q),
                              "timestamp_source": "sim_clock" if hasattr(self.backend, "on_tick") else "request_sent"})

    def start(self):
        self.backend.on_command = self.command
        self.state()
        if hasattr(self.backend, "on_tick"):
            self.capture("scene")
            self.capture("wrist")
            self.backend.on_tick = self.tick
            return
        def poll(fn, hz):
            while not self.stop_event.is_set():
                started = time.monotonic()
                try:
                    fn()
                except Exception as exc:
                    self.errors.append(str(exc))
                    self.stop_event.set()
                    try:
                        self.backend.stop()
                    except Exception as stop_error:
                        self.errors.append(str(stop_error))
                self.stop_event.wait(max(0, 1/hz-(time.monotonic()-started)))
        for fn, hz in [(self.state, 30), (lambda: self.capture("scene"), 30), (lambda: self.capture("wrist"), 30)]:
            thread = threading.Thread(target=poll, args=(fn, hz), daemon=True)
            thread.start()
            self.threads.append(thread)
        # State logging is already running while initial camera fetches complete.
        self.capture("scene")
        self.capture("wrist")

    def finish(self):
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=4)
        if hasattr(self.backend, "on_tick"):
            self.backend.on_tick = None
        self.backend.on_command = None
        self.stream.close()
        # Store the motor target active at each sensor timestamp as a right-continuous signal.
        if self.commands and hasattr(self.backend, "on_tick"):
            index = 0
            for row in self.rows:
                while index+1 < len(self.commands) and self.commands[index+1]["t"] <= row["t"]+1e-9:
                    index += 1
                if self.commands[index]["t"] <= row["t"]+1e-9:
                    row["q_target"] = self.commands[index]["q_target"]
            with (self.directory/"states.jsonl").open("w", encoding="utf-8") as stream:
                for row in self.rows:
                    stream.write(json.dumps(row)+"\n")
        gemini.atomic_json(self.directory/"commands.json", self.commands)
        self.frames.sort(key=lambda row: (row["t"], row["camera"]))
        gemini.atomic_json(self.directory/"frames.json", self.frames)
        for camera in ("scene", "wrist"):
            frames = [r for r in self.frames if r["camera"] == camera]
            if frames:
                timeline_video([(r["t"], self.directory/r["path"]) for r in frames], self.directory/(camera+".mp4"),
                               start=self.rows[0]["t"] if self.rows else frames[0]["t"],
                               end=self.rows[-1]["t"] if self.rows else frames[-1]["t"])
        gaps = np.diff([r["t"] for r in self.rows])
        return {"states": len(self.rows), "frames": self.counts,
                "final_frames": {camera: row["path"] for camera, row in self.latest_captures.items()},
                "state_hz": float(1/np.mean(gaps)) if len(gaps) else None,
                "max_state_gap_s": float(max(gaps)) if len(gaps) else None, "errors": self.errors}


class Harness:
    def __init__(self, run, task, backend, *, approve=None):
        self.run, self.task, self.backend = Path(run).resolve(), task, backend
        self.run.mkdir(parents=True, exist_ok=True)
        self.motion = Motion(backend, task)
        self.lock = threading.Lock()
        self.approve = approve
        self.last_episode, self.status = None, "ready"
        self._sheet_moved = False

    def obs(self):
        state = self.backend.state()
        paths = {}
        destination = self.run/"obs"
        destination.mkdir(exist_ok=True)
        for camera in ("scene", "wrist"):
            image = self.backend.frame(camera)
            if camera == "scene":
                image = annotate(image, self.backend.camera("scene"))
            path = destination/(camera+".jpg")
            cv2.imwrite(str(path), image)
            paths[camera+"_cam_path"] = str(path)
        return {"ee_pos": state["xyz"], "q": state["q"], "q_target": state["q_target"],
                "gripper": state["gripper"], "holding": state.get("holding", False),
                "status": self.status, "workspace": self.motion.safety.bounds(), "last_episode": self.last_episode,
                **paths}

    def sheet_guard(self):
        board = self.task.get("board")
        if not board or hasattr(self.backend, "on_tick"):
            return {"checked": False, "reason": "no printed board or simulated bench"}
        if self._sheet_moved:
            raise Refused("Sheet moved: restart only after a fresh survey")
        cells = np.asarray(list(board.get("cells", {}).values()))
        xy = np.asarray(self.backend.state()["xyz"][:2])
        if not len(cells) or np.min(np.linalg.norm(cells-xy, axis=1)) < .09:
            return {"checked": False, "reason": "arm near board; outline measurement would be occluded"}
        from scripts.lead.measure_grid import measure
        readings = []
        for _ in range(3):
            try:
                measured, size, _ = measure(self.backend.frame("scene"), self.backend.camera("scene"))
                if abs(size-board["cell_m"]) < .0012:
                    readings.append(np.linalg.norm(np.asarray(measured["r1c1"])-board["center_xy"]))
            except SystemExit:
                pass
        if not readings:
            raise Refused("Sheet guard: board not measurable while arm is away")
        moved = float(np.median(readings))
        if moved > .004:
            self._sheet_moved = True
            raise Refused(f"Sheet moved {moved*1000:.1f} mm; experiment stopped")
        return {"checked": True, "moved_mm": moved*1000}

    def plan(self, primitives, *, split="train"):
        if not self.lock.acquire(blocking=False):
            raise Refused("Another plan is running")
        recorder = None
        result = {"status": "refused", "results": []}
        episodes = self.run/"episodes"
        episodes.mkdir(exist_ok=True)
        number = 1+max([int(p.name) for p in episodes.iterdir() if p.is_dir() and p.name.isdigit()] or [0])
        directory = episodes/str(number)
        directory.mkdir()
        (directory/"states.jsonl").touch()
        gemini.atomic_json(directory/"frames.json", [])
        gemini.atomic_json(directory/"commands.json", [])
        result.update(episode=str(number), directory=str(directory), split=split)
        gemini.atomic_json(directory/"primitives.json", primitives)
        try:
            if not isinstance(primitives, list) or not 1 <= len(primitives) <= 100:
                raise Refused("Plan must contain 1..100 primitives")
            if split not in ("train", "heldout"):
                raise Refused("split must be train or heldout")
            if self.approve and not self.approve(primitives):
                raise Refused("Owner declined this real_plan")
            self.sheet_guard()
            if hasattr(self.backend, "_stop"):
                self.backend._stop.clear()
            self.status = "recording"
            recorder = Recorder(directory, self.backend, self.motion)
            recorder.start()
            for primitive in primitives:
                if recorder.errors:
                    raise Refused("Recording/live safety failed: "+"; ".join(recorder.errors))
                value = self.motion.execute(primitive)
                completed = time.monotonic()
                # Re-observe after EVERY action; these images are supplied on the next model turn.
                if hasattr(self.backend, "on_tick"):
                    self.backend.advance(self.backend.target, self.backend.d.time+1/30)
                    recorder.state()
                else:
                    time.sleep(.05)
                recorder.verify_after(completed)
                result["results"].append({"primitive": primitive, "result": value,
                                          "observed_state": self.backend.state()})
            result["sheet_guard"] = self.sheet_guard()
            result["status"] = "ok"
        except Exception as exc:
            result["error"] = gemini._safe(str(exc))
            if recorder is not None:
                try:
                    self.backend.stop()
                except Exception as stop_error:
                    result["stop_error"] = gemini._safe(str(stop_error))
        finally:
            if recorder:
                try:
                    result["recording"] = recorder.finish()
                    if recorder.errors:
                        result["status"] = "refused"
                        result["error"] = "; ".join(recorder.errors)
                    hz = result["recording"]["state_hz"]
                    if not hasattr(self.backend, "on_tick") and (hz is None or hz < 25 or result["recording"]["max_state_gap_s"] > .15):
                        # Keep the underlying failure: a plan that stopped early also reads as low telemetry.
                        result.update(status="refused", cause=result.get("error"),
                                      error="Telemetry did not sustain 30 Hz (minimum accepted 25 Hz)")
                except Exception as exc:
                    result.update(status="refused", error="Recording finalization failed: "+gemini._safe(str(exc)))
            self.last_episode = result
            self.status = result["status"]
            try:
                gemini.atomic_json(directory/"result.json", result)
            finally:
                self.lock.release()
        return result


def make_server(harness, address=("127.0.0.1", 8081)):
    if address[0] not in ("127.0.0.1", "localhost"):
        raise ValueError("Lab binds loopback only")
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, code, value):
            data = json.dumps(value, default=gemini._json_default, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != "/obs":
                return self.reply(404, {"error": "not found"})
            try:
                self.reply(200, harness.obs())
            except Exception as exc:
                self.reply(503, {"error": gemini._safe(str(exc))})

        def do_POST(self):
            if self.path != "/plan":
                return self.reply(404, {"error": "not found"})
            if self.headers.get("Origin"):
                return self.reply(403, {"error": "Browser-origin robot commands are disabled"})
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 65536:
                    raise ValueError("Invalid request size")
                body = json.loads(self.rfile.read(size))
                self.reply(200, harness.plan(body["primitives"], split=body.get("split", "train")))
            except Exception as exc:
                self.reply(400, {"error": gemini._safe(str(exc))})
    return ThreadingHTTPServer(address, Handler)
