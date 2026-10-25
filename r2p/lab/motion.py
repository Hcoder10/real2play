"""Shared Cartesian primitives. All coordinates are footprint-frame metres.

The backend must expose state/goto/gripper/park/stop. Safety is checked here and
again on every interpolated joint command in the backend, never delegated to an LLM.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import math
import threading

import numpy as np


class Refused(ValueError):
    pass


class StallDetector:
    """Sustained closure error with a stationary encoder, not transient servo lag."""
    def __init__(self):
        self.samples = deque(maxlen=12)
        self.lock = threading.Lock()

    def update(self, measured, commanded, t):
        with self.lock:
            if commanded >= .1:
                self.samples.clear()
                return False
            if not self.samples or t-self.samples[-1][0] >= .02:
                self.samples.append((t, measured, commanded))
            recent = [s for s in self.samples if t-s[0] <= .35]
            return bool(len(recent) >= 4 and recent[-1][0]-recent[0][0] >= .15 and
                        max(s[1] for s in recent)-min(s[1] for s in recent) < .02 and
                        all(s[1]-s[2] > .08 for s in recent))


def aim_with_jaw_ik(backend, goal, optical_axis, initial=None):
    """Optimize a Cartesian command using the SAME IK that will execute it.

    A joint-space camera aim followed by an independent fingers-down IK solve
    would silently discard the solved camera orientation.
    """
    from scipy.optimize import least_squares
    from scripts.lead.jaw_move import plan, OWNER_LO, OWNER_HI
    from r2p.sim.twin import load_physics
    state = backend.state()
    owner = backend.jm.to_owner(state["q"])
    start = np.asarray(state["xyz"])
    bounds = (np.array([.10, -.22, .045]), np.array([.315, .22, .22]))
    seed = np.clip(start if initial is None else initial, bounds[0]+1e-5, bounds[1]-1e-5)
    def residual(p):
        _, error, q = plan(np.asarray(p), state["gripper"], owner, backend.arm, backend.jm,
                           load_physics(), OWNER_LO, OWNER_HI)
        origin, direction = optical_axis(q)
        ray = np.asarray(goal)-origin
        ray /= max(np.linalg.norm(ray), 1e-9)
        return np.r_[direction-ray, error*100, .03*(p-start)]
    fit = least_squares(residual, seed, bounds=bounds, max_nfev=35, diff_step=.001)
    r = residual(fit.x)
    if np.linalg.norm(r[:3]) > .15 or r[3] > .4:
        raise Refused("Point cannot be viewed using calibrated roll-locked jaw IK")
    return fit.x


def vector(value, n=3):
    a = np.asarray(value, float)
    if a.shape != (n,) or not np.isfinite(a).all():
        raise Refused(f"Expected {n} finite numbers")
    return a


@dataclass
class Safety:
    reach: tuple = (.10, .32)
    max_z: float = .25
    min_finger: float = .006
    min_jaw: float = .007
    contact_finger: float = .002
    travel_z: float = .045
    protected_z: float = .035
    speed_cap: float = 20.0  # degrees / second, consistent with armd
    near_speed_cap: float = 8.0
    max_duration_s: float = 150

    def bounds(self):
        return {"frame": "spec", "units": "metres", "radial_m": list(self.reach),
                "z_m": [self.min_jaw, self.max_z], "finger_min_m": self.min_finger,
                "speed_deg_s": self.speed_cap, "near_table_speed_deg_s": self.near_speed_cap}


class Motion:
    def __init__(self, backend, task=None, safety=None):
        self.backend, self.task = backend, task or {}
        self.safety = safety or Safety()
        self.contact_xy = None
        self.phase = "idle"
        self.protected = []
        board = self.task.get("board", {})
        for xy in board.get("cells", {}).values():
            self.protected.append((np.asarray(xy, float), np.full(2, board.get("cell_m", .045)/2+.024)))
        for obj in self.task.get("objects", []):
            if obj.get("role") in ("target", "container") or obj.get("fixed"):
                xy = obj.get("xy", obj.get("xyz", [])[:2])
                if len(xy) == 2:
                    size = obj.get("size_m", [.04, .04])
                    self.protected.append((np.asarray(xy, float), np.full(2, max(size[:2])/2+.024)))
        for region in self.task.get("protected_regions", []):
            self.protected.append((vector(region["xy"], 2), vector(region["half_size"], 2)))
        backend.motion = self

    def contact_allowed(self, xyz):
        return self.contact_xy is not None and np.linalg.norm(np.asarray(xyz)[:2]-self.contact_xy) <= .004

    def check_point(self, xyz, *, measured=False):
        p = vector(xyz)
        s = self.safety
        rest_transit = bool(getattr(self.backend, "rest_phase", False))
        if (not rest_transit and not s.reach[0] <= np.linalg.norm(p[:2]) <= s.reach[1]) or not .002 <= p[2] <= s.max_z:
            raise Refused("Outside reach band 0.10..0.32 m or height limits")
        contact = self.contact_allowed(p)
        if p[2] < s.min_jaw and not contact:
            raise Refused("Live abort: measured jaw below 7 mm" if measured else "Jaw below 7 mm outside grasp/place")
        if p[2] < s.protected_z and not contact:
            for center, half in self.protected:
                if np.all(np.abs(p[:2]-center) <= half):
                    raise Refused("Protected region: only a vertical grasp/place descent at its own xy is permitted")

    def check_joints(self, q, arm, *, measured=False):
        p = arm.tips(q)[2]
        self.check_point(p, measured=measured)
        geo = arm.geometry(q)
        minimum = self.safety.contact_finger if self.contact_allowed(p) else self.safety.min_finger
        if geo["finger"] < minimum or geo["gripper"] < .004 or geo["link"] < .005:
            raise Refused(f"Mesh clearance refusal: {geo}")

    def speed(self, value):
        value = float(value)
        if not math.isfinite(value) or not 0 < value <= self.safety.speed_cap:
            raise Refused("Speed must be in (0, 20] degrees/second")
        return value

    def move(self, xyz, speed=12):
        p = vector(xyz)
        self.check_point(p)
        state = self.backend.state()
        if state.get("at_rest") and not self.safety.reach[0] <= np.linalg.norm(state["xyz"][:2]) <= self.safety.reach[1]:
            # Only the daemon's known rest pose can enter through this staging
            # route. Finger, paper, speed and joint-path guards remain enforced.
            self.backend.rest_phase = True
            try:
                self.backend.goto([.17, 0, .05], speed=8, min_finger=self.safety.min_finger)
            finally:
                self.backend.rest_phase = False
        start = vector(self.backend.state()["xyz"])
        # Near the table, ALL moves are vertical. Descent exceptions cannot authorize a lateral sweep.
        if min(start[2], p[2]) < self.safety.protected_z and np.linalg.norm(start[:2]-p[:2]) > .002:
            raise Refused("Raise vertically above 35 mm before lateral travel")
        speed = self.speed(speed)
        if min(start[2], p[2]) < self.safety.travel_z:
            speed = min(speed, self.safety.near_speed_cap)
        # Small Cartesian waypoints prevent joint interpolation from dragging fingers sideways.
        n = max(1, math.ceil(np.linalg.norm(p-start)/.003))
        result = None
        for target in np.linspace(start, p, n+1)[1:]:
            self.check_point(target)
            result = self.backend.goto(target, speed=speed,
                                       min_finger=self.safety.contact_finger if self.contact_allowed(target)
                                       else self.safety.min_finger)
        return result

    def _approach(self, xyz, speed):
        current = vector(self.backend.state()["xyz"])
        hover = max(self.safety.travel_z, xyz[2]+.045)
        self.move([*current[:2], max(current[2], hover)], speed)
        self.move([*xyz[:2], hover], speed)
        return hover

    def _contact(self, xyz, speed, place=False):
        # Stop descent at the first sustained encoder discrepancy; no Cartesian force is ever injected.
        start = self.backend.state()["xyz"][2]
        contact = False
        last = start
        for z in np.arange(start-.002, xyz[2]-.001, -.002):
            last = max(z, xyz[2])
            self.move([*xyz[:2], last], speed)
            st = self.backend.state()
            lag = np.rad2deg(np.abs(vector(st["q"], 6)-vector(st["q_target"], 6)))
            if np.max(lag[1:4]) > 4:
                contact = True
                break
        self.move([*xyz[:2], min(start, last+.004)], speed)
        if place and contact:
            self.backend.gripper(20)
        return {"contact": contact, "encoder_lag_deg": lag.tolist() if start > xyz[2]+.002 else None,
                "relieved_m": .004}

    def execute(self, primitive):
        if not isinstance(primitive, dict):
            raise Refused("Primitive must be an object")
        name = primitive.get("name", primitive.get("tool"))
        args = primitive.get("args", {k: v for k, v in primitive.items() if k not in ("name", "tool")})
        if not isinstance(args, dict):
            raise Refused("Primitive args must be an object")
        self.phase = str(name)
        speed = self.speed(args.get("speed", 12))
        self.backend.speed_limit = speed
        if name == "deproject":
            camera = self.backend.camera(args.get("camera", "scene"))
            height = float(args.get("z_plane", 0))
            if not math.isfinite(height) or not 0 <= height <= self.safety.max_z:
                raise Refused("Deprojection plane must be finite and within 0..0.25 m")
            xyz = camera.deproject(vector([args["u"], args["v"]], 2), h=height)
            return {"x": float(xyz[0]), "y": float(xyz[1]), "frame": "spec"}
        if name in ("goto", "move_by", "look_at", "grasp", "place", "place_until_contact", "move_until_contact"):
            if name == "move_by":
                xyz = vector(self.backend.state()["xyz"]) + vector([args[k] for k in ("dx", "dy", "dz")])
            else:
                xyz = vector([args[k] for k in ("x", "y", "z")])
            if not self.safety.reach[0] <= np.linalg.norm(xyz[:2]) <= self.safety.reach[1] or not .002 <= xyz[2] <= self.safety.max_z:
                raise Refused("Outside reach band or height limits")
            if name in ("goto", "move_by"):
                return self.move(xyz, speed)
            if name == "look_at":
                # Backend solves the calibrated optical-axis objective, roll remains locked.
                return self.move(self.backend.look_at(xyz), speed)
            if name == "grasp":
                self.backend.gripper(float(args.get("open", 20)))
            hover = self._approach(xyz, speed)
            self.contact_xy = xyz[:2].copy() if name in ("grasp", "place", "place_until_contact") else None
            try:
                if name.endswith("until_contact"):
                    result = self._contact(xyz, speed, name == "place_until_contact")
                else:
                    self.move(xyz, speed)
                    result = self.backend.gripper(float(args.get("close", 0)) if name == "grasp" else 20)
                    if name == "grasp":
                        if hasattr(self.backend, "dwell"):
                            self.backend.dwell(.4)
                        st = self.backend.state()
                        result = {**(result or {}), "holding": bool(st.get("holding", False)),
                                  "verification": "encoder stall", "verified": bool(st.get("holding", False)),
                                  "status": "grasp_verified" if st.get("holding") else "no_gripper_stall"}
                self.move([*xyz[:2], hover], speed)
                return result
            finally:
                self.contact_xy = None
        if name in ("open_gripper", "close_gripper"):
            return self.backend.gripper(float(args.get("value", 20 if name == "open_gripper" else 0)))
        if name == "home":
            state = self.backend.state()
            if state.get("at_rest"):
                return {"at_rest": True}
            current = state["xyz"]
            self.move([*current[:2], max(.05, current[2])], speed)
            return self.backend.park()
        if name == "wait":
            seconds = float(args.get("seconds", .5))
            if not math.isfinite(seconds) or not 0 < seconds <= 10:
                raise Refused("Wait must be 0..10 seconds")
            return self.backend.dwell(seconds)
        if name == "done":
            return {"done": True}
        raise Refused(f"Unknown primitive: {name}")
