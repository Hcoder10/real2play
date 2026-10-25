"""One bus owner, with guarded model-radian commands over loopback HTTP."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import numpy as np

from .config import load_config, overlay
from .guard import Guard
from .io import vector
from .kinematics import solve_ik
from .record import json_default


class ArmController:
    def __init__(self, robot, *, arm=None, joint_map=None, cfg=None, realtime=True):
        from r2p.model import Arm, JointMap
        self.robot = robot
        self.arm = arm or Arm()
        self.joint_map = joint_map or getattr(robot, "joint_map", None) or JointMap()
        self.cfg = overlay(load_config(), cfg or {})
        self.target = self.joint_map.to_model(robot.read())
        # Boot must be at a supported rest pose, as priming drops torque once.
        self.rest_q = vector(self.cfg.get("rest_q", self.target))
        guard_cfg = {**self.cfg["guard"], "speed_deg_s": min(20, self.cfg["guard"]["speed_deg_s"])}
        self.guard = Guard(self.arm, guard_cfg, start_q=self.target)
        self.lock = threading.RLock()
        self.stop_requested = threading.Event()
        self.realtime = realtime
        self.torque_enabled = True
        self.last_write = time.monotonic() - 1/30

    def state(self):
        with self.lock:
            owner = self.robot.read()
            q = self.joint_map.to_model(owner)
            return {"q": q.tolist(), "owner_deg": owner.tolist(), "q_target": self.target.tolist(),
                    "xyz": self.arm.tips(q)[2].tolist(), "loads": self.robot.loads(),
                    "at_rest": bool(np.max(np.abs(q - self.rest_q)) <= np.deg2rad(2.01)),
                    "rest_q": self.rest_q.tolist(), "torque_enabled": self.torque_enabled,
                    "joint_map_verified": self.joint_map.verified}

    def _hold(self):
        self.robot.hold()
        self.target = self.joint_map.to_model(self.robot.read())

    def _check(self):
        q = self.joint_map.to_model(self.robot.read())
        loads = self.robot.loads()
        if loads and max(map(abs, loads.values())) > self.cfg["load_limit"]:
            raise RuntimeError("load abort")
        if np.max(np.abs(q[:5] - self.target[:5])) > np.deg2rad(self.cfg["guard"]["lag_deg"]):
            raise RuntimeError("lag abort")
        if self.stop_requested.is_set():
            raise RuntimeError("operator stop")
        return q

    def move(self, target, *, stream=False, retreat=False):
        wanted = vector(target)
        with self.lock:
            if not self.torque_enabled:
                raise RuntimeError("torque is off; restart only at supported rest")
            self.stop_requested.clear()
            if np.any(wanted < self.guard.limits[:, 0]) or np.any(wanted > self.guard.limits[:, 1]):
                raise ValueError("target outside joint limits with 2 degree margin")
            if not self.guard.path_clear(self.target, wanted):
                raise ValueError("path clearance refusal")
            try:
                for _ in range(30 * 60):
                    started = time.perf_counter()
                    measured = self._check()
                    # Stream clients cannot exceed 20 deg/s by sending faster.
                    period = min(1/30, max(1e-6, time.monotonic() - self.last_write)) if self.realtime else 1/30
                    self.guard.hz = 1/period if stream else 30
                    result = self.guard.project(self.target, wanted, measured=measured, retreat=retreat)
                    if not result.accepted:
                        raise ValueError(f"guard refusal: {result.reason}")
                    self.robot.write(self.joint_map.to_owner(result.target))
                    self.target = result.target
                    self.last_write = time.monotonic()
                    if hasattr(self.robot, "step"):
                        self.robot.step(1/30)
                    if stream or np.max(np.abs(self.target - wanted)) < 1e-8:
                        return self.state()
                    if self.realtime:
                        time.sleep(max(0, 1/30 - (time.perf_counter() - started)))
                raise TimeoutError("move timed out")
            except BaseException:
                self._hold()
                raise

    def command(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("command must be a JSON object")
        names = [k for k in ("joints", "xyz", "gripper", "rest", "release", "hold", "stop") if k in payload]
        if len(names) != 1:
            raise ValueError("specify exactly one of joints, xyz, gripper, rest, release, hold, stop")
        kind = names[0]
        if kind == "stop":
            self.stop_requested.set()
            return {"stopping": True}
        with self.lock:
            if kind == "hold":
                self._hold()
                return self.state()
            if kind == "release":
                state = self.state()
                if not state["at_rest"] or np.max(np.abs(self.target - self.rest_q)) > np.deg2rad(2.01):
                    raise ValueError("release allowed only at the supported rest pose")
                self.robot.torque_off()
                self.torque_enabled = False
                return self.state()
            if kind == "joints":
                target = vector(payload[kind])
            elif kind == "xyz":
                target = solve_ik(self.arm, payload[kind], self.target)
            elif kind == "gripper":
                target = self.target.copy()
                target[5] = float(payload[kind])
            else:
                target = np.clip(self.rest_q, self.guard.limits[:, 0], self.guard.limits[:, 1])
            return self.move(target, stream=bool(payload.get("stream", False)), retreat=kind == "rest")


def make_server(controller, address=("127.0.0.1", 8793)):
    if address[0] not in ("127.0.0.1", "localhost"):
        raise ValueError("armd binds loopback only")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, code, result):
            body = json.dumps(result, default=json_default).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/state":
                return self.reply(404, {"error": "not found"})
            try:
                self.reply(200, controller.state())
            except Exception as exc:
                self.reply(503, {"error": str(exc)})

        def do_POST(self):
            if self.path != "/cmd":
                return self.reply(404, {"error": "not found"})
            # Reject browser cross-origin writes to the local actuator endpoint.
            if self.headers.get("Origin"):
                return self.reply(403, {"error": "browser-origin commands are disabled"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 65536:
                    raise ValueError("invalid command size")
                result = controller.command(json.loads(self.rfile.read(size)))
                self.reply(200, result)
            except (ValueError, TypeError, KeyError) as exc:
                self.reply(400, {"error": str(exc)})
            except Exception as exc:
                self.reply(409, {"error": str(exc)})

    return ThreadingHTTPServer(address, Handler)


class DaemonRobot:
    """Owner-space robot interface without opening or reconnecting the bus."""
    def __init__(self, url="http://127.0.0.1:8793", joint_map=None):
        from .config import load_joint_map
        self.url = url.rstrip("/")
        self.joint_map = joint_map or load_joint_map()

    def request(self, command=None):
        request = Request(self.url + ("/state" if command is None else "/cmd"),
                          data=None if command is None else json.dumps(command, default=json_default).encode(),
                          headers={"Content-Type": "application/json"})
        timeout = 2 if command is None or command.get("stream") else 65
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            try:
                detail = json.loads(exc.read()).get("error", str(exc))
            except (ValueError, AttributeError):
                detail = str(exc)
            raise RuntimeError(f"armd refused: {detail}") from exc

    def read(self):
        return vector(self.request()["owner_deg"])

    def write(self, owner):
        result = self.request({"joints": self.joint_map.to_model(vector(owner)), "stream": True})
        return self.joint_map.to_owner(vector(result["q_target"]))

    def loads(self):
        return self.request()["loads"]

    def hold(self):
        self.request({"hold": True})

    def torque_off(self):
        self.request({"release": True})

    def close(self, torque_off=False):
        if torque_off:
            self.torque_off()
