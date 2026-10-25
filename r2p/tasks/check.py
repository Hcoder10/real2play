"""Shared geometric success semantics for the reference twin, perception and Luau port."""
from __future__ import annotations

import copy
import math
import numpy as np

from .scene import pose_of, slot_size, yaw_symmetry


def _refs(goal):
    refs = [goal[key] for key in ("a", "b") if key in goal]
    refs.extend(goal.get("objs", []))
    if isinstance(goal.get("xy"), str):
        refs.append(goal["xy"])
    return refs


def check(task, state) -> dict:
    """Evaluate strict tolerances in metres/degrees; absent required evidence fails.

    Flat gripper_open/site_xyz (INTERFACES) and nested gripper.open/site_xyz/holding
    (task_spec) are accepted. Explicit holding is authoritative; the legacy geometric
    grasp predicate also requires closure AND lift, never just proximity to the jaws.
    Velocities, when supplied, must be below 0.5 m/s. No velocity is inferred here.
    """
    objects = state.get("objects", {})
    definitions = {obj["id"]: obj for obj in task["objects"]}
    grip = dict(state.get("gripper") or {})
    if "gripper_open" in state:
        grip["open"] = state["gripper_open"]
    for key in ("site_xyz", "holding"):
        if key in state:
            grip[key] = state[key]
    has_gripper = "gripper_open" in state or state.get("gripper") is not None

    def geom(oid):
        definition, pose = definitions[oid], objects[oid]
        xyz, yaw = pose_of(pose)
        size = pose.get("size_m", definition["size_m"]) if isinstance(pose, dict) else definition["size_m"]
        dims = np.asarray(slot_size(definition["shape"], size, definition.get("axis", "z")))
        if dims.shape != (3,) or not np.isfinite(dims).all() or np.any(dims <= 0):
            raise ValueError("invalid object dimensions")
        return xyz, yaw, dims

    def over(a, b, margin):
        pa, _, _ = geom(a)
        pb, yaw, dims = geom(b)
        delta = pa[:2] - pb[:2]
        shape = definitions[b]["shape"]
        if shape in ("bowl", "cylinder", "sphere"):
            return np.linalg.norm(delta) < dims[0] / 2 - margin
        angle = math.radians(yaw)
        x = math.cos(angle) * delta[0] + math.sin(angle) * delta[1]
        y = -math.sin(angle) * delta[0] + math.cos(angle) * delta[1]
        return abs(x) < dims[0] / 2 - margin and abs(y) < dims[1] / 2 - margin

    def on_top(a, b):
        pa, _, sa = geom(a)
        pb, _, sb = geom(b)
        return over(a, b, 0) and abs(pa[2] - sa[2]/2 - pb[2] - sb[2]/2) < 0.012

    def evaluate(goal):
        pred = goal["pred"]
        if pred == "gripper_open":
            return grip.get("open") is True
        if pred == "stacked":
            chain = goal["objs"]
            return len(chain) >= 2 and all(on_top(top, base) for base, top in zip(chain, chain[1:]))
        a = goal["a"]
        pa, yaw, sa = geom(a)
        if pred == "inside":
            pb, _, sb = geom(goal["b"])
            return over(a, goal["b"], 0.004) and pa[2] < pb[2] + sb[2]/2 + sa[2]/2
        if pred == "on_top":
            return on_top(a, goal["b"])
        if pred == "at":
            target = geom(goal["xy"])[0][:2] if isinstance(goal["xy"], str) else np.asarray(goal["xy"])
            return np.linalg.norm(pa[:2] - target) < goal["tol_m"]
        if pred == "near":
            return np.linalg.norm(pa[:2] - geom(goal["b"])[0][:2]) < goal["d_m"]
        if pred == "lifted":
            return pa[2] - sa[2]/2 > goal["h_m"]
        if pred == "grasped":
            if "holding" in grip:
                return a in grip["holding"]
            if grip.get("open") is not False or "site_xyz" not in grip:
                return False
            site = np.asarray(grip["site_xyz"], dtype=float)
            return site.shape == (3,) and np.isfinite(site).all() and np.linalg.norm(site - pa) < max(sa)/2 + 0.02 and pa[2] - sa[2]/2 > 0.005
        if pred == "upright":
            pose = objects[a]
            if isinstance(pose, dict):
                quat = pose.get("quat")
            else:
                quat = pose[3:] if len(pose) == 7 else None
            if quat is None:
                return True
            quat = np.asarray(quat, dtype=float)
            norm = np.linalg.norm(quat)
            if quat.shape != (4,) or not np.isfinite(quat).all() or norm == 0:
                return False
            w, x, y, z = quat / norm
            tilt = math.degrees(math.acos(float(np.clip(1 - 2*(x*x + y*y), -1, 1))))
            return tilt < goal["tol_deg"]
        if pred == "yaw":
            definition = dict(definitions[a], size_m=objects[a].get("size_m", definitions[a]["size_m"])) if isinstance(objects[a], dict) else definitions[a]
            k = yaw_symmetry(definition)
            if k == 0:
                return True
            period = 360 / k
            error = (yaw - goal["deg"] + period/2) % period - period/2
            return abs(error) < goal["tol_deg"]
        if pred == "not_moved":
            initial = state.get("initial", {})
            if a not in initial:
                return False
            return np.linalg.norm(pa[:2] - pose_of(initial[a])[0][:2]) < goal["tol_m"]
        return False

    results = []
    for goal in task["goal"]:
        result = {"pred": goal["pred"], "ok": False}
        refs = _refs(goal)
        missing = [oid for oid in refs if oid not in objects or objects[oid] is None]
        if missing:
            result.update(ok=bool(goal.get("optional", False)), absent=missing)
        else:
            try:
                result["ok"] = bool(evaluate(goal))
            except (ValueError, TypeError, KeyError, IndexError, OverflowError):
                result["reason"] = "invalid or missing predicate evidence"
        results.append(result)
    if has_gripper and not any(g["pred"] in ("grasped", "gripper_open") for g in task["goal"]):
        results.append({"pred": "gripper_open", "ok": grip.get("open") is True})
    moving = []
    for oid in sorted({oid for goal in task["goal"] for oid in _refs(goal)}):
        pose = objects.get(oid)
        if not isinstance(pose, dict) or "v" not in pose:
            continue
        try:
            velocity = np.asarray(pose["v"], dtype=float)
            speed = float(np.linalg.norm(velocity))
            if not np.isfinite(speed) or speed >= 0.5:
                moving.append(oid)
        except (TypeError, ValueError):
            moving.append(oid)
    results.append({"pred": "settled", "ok": not moving})
    return {"ok": all(r["ok"] for r in results), "per_predicate": results}


def check_window(task, states, hz=30, *, hold_s=0.5) -> dict:
    """Require every predicate on every tick in the final ceil(hold_s * hz) ticks.

    Short histories fail. A history's first object poses provide not_moved's initial
    evidence unless the caller supplies an explicit initial mapping on a tick.
    """
    if not math.isfinite(hz) or hz <= 0 or not math.isfinite(hold_s) or hold_s <= 0:
        raise ValueError("hz and hold_s must be finite and positive")
    states = list(states)
    count = math.ceil(hold_s * hz)
    if not states:
        return {"ok": False, "per_predicate": [{"pred": "window", "ok": False}], "ticks": 0, "required_ticks": count}
    initial = states[0].get("initial", states[0].get("objects", {}))
    checks = [check(task, dict(state, initial=state.get("initial", initial))) for state in states[-count:]]
    # Aggregate each explicit predicate by position (several may have the same name).
    results = []
    for index, goal in enumerate(task["goal"]):
        entries = [result["per_predicate"][index] for result in checks]
        results.append({"pred": goal["pred"], "ok": all(entry["ok"] for entry in entries)})
    for name in ("gripper_open", "settled"):
        entries = [entry for result in checks for entry in result["per_predicate"][len(task["goal"]):] if entry["pred"] == name]
        if entries:
            results.append({"pred": name, "ok": all(entry["ok"] for entry in entries)})
    enough = len(states) >= count
    if not enough:
        results.append({"pred": "window", "ok": False})
    return {"ok": enough and all(r["ok"] for r in results), "per_predicate": results,
            "ticks": min(len(states), count), "required_ticks": count}
