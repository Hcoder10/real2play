"""Geometry and deterministic scene sampling, entirely in the footprint frame."""
from __future__ import annotations

import copy
import math
import numpy as np

from .spec import SHAPES, validate


def slot_size(shape: str, size_m, axis: str = "z") -> list[float]:
    s = list(map(float, size_m))
    if shape not in SHAPES or len(s) != SHAPES[shape] or any(not math.isfinite(v) or v <= 0 for v in s):
        raise ValueError("size_m must contain the shape's positive finite dimensions")
    if shape in ("box", "plate"):
        return s
    if shape == "sphere":
        return [s[0]] * 3
    if shape == "stick":
        return [s[0], s[1], s[1]]
    if shape == "bowl":
        return [s[0], s[0], s[1]]
    if shape == "cylinder":
        dims = [s[0]] * 3
        dims[("x", "y", "z").index(axis)] = s[1]
        return dims
    raise ValueError(f"Unknown shape: {shape}")


def yaw_symmetry(obj) -> int:
    shape = obj["shape"]
    if shape in ("sphere", "bowl") or (shape == "cylinder" and obj.get("axis", "z") == "z"):
        return 0
    if shape in ("stick", "cylinder"):
        return 2
    sx, sy, _ = slot_size(shape, obj["size_m"], obj.get("axis", "z"))
    return 4 if math.isclose(sx, sy, rel_tol=0, abs_tol=1e-9) else 2


def pose_of(p):
    """Return (xyz ndarray, yaw degrees) from a pose dict or xyz/[xyz,wxyz] array."""
    if isinstance(p, dict):
        xyz = np.asarray(p["xyz"], dtype=float)
        if "yaw_deg" in p:
            yaw = float(p["yaw_deg"])
        elif "quat" in p:
            q = np.asarray(p["quat"], dtype=float)
            if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) == 0:
                raise ValueError("quat must be a nonzero finite [w,x,y,z]")
            w, x, y, z = q / np.linalg.norm(q)
            yaw = math.degrees(math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z)))
        else:
            yaw = 0.0
    else:
        values = np.asarray(p, dtype=float)
        if values.shape == (7,):
            return pose_of({"xyz": values[:3], "quat": values[3:]})
        xyz, yaw = values, 0.0
    if xyz.shape != (3,) or not np.isfinite(xyz).all() or not math.isfinite(yaw):
        raise ValueError("pose requires finite xyz and yaw")
    return xyz, yaw


def poses_from_scene(scene) -> dict:
    fields = ("xyz", "yaw_deg", "quat", "size_m")
    return {o["id"]: {key: copy.deepcopy(o[key]) for key in fields if key in o} for o in scene["objects"]}


def encode_slots(task, poses) -> np.ndarray:
    k = task["limits"]["max_objects"]
    if k < len(task["objects"]):
        raise ValueError("max_objects is smaller than the object count")
    out = np.zeros((k, 9), dtype=np.float32)
    for index, obj in enumerate(task["objects"]):
        p = poses.get(obj["id"])
        if p is None:
            continue
        xyz, yaw = pose_of(p)
        # A stick has no front/back: this canonical interval is part of policy I/O.
        if obj["shape"] == "stick":
            yaw = (yaw + 90) % 180 - 90
        size = p.get("size_m", obj["size_m"]) if isinstance(p, dict) else obj["size_m"]
        angle = math.radians(yaw)
        out[index] = [*xyz, math.sin(angle), math.cos(angle),
                      *slot_size(obj["shape"], size, obj.get("axis", "z")), 1]
    return out


def encode_obs(task, q, poses, command=None) -> np.ndarray:
    from .commands import command_onehot
    joints = np.asarray(q, dtype=np.float32)
    if joints.shape != (6,) or not np.isfinite(joints).all():
        raise ValueError("q must contain six finite joint angles")
    parts = [joints, encode_slots(task, poses).ravel()]
    if command is not None:
        parts.append(command_onehot(task, command))
    return np.concatenate(parts)


def _draw(rng, value):
    return float(rng.uniform(*value)) if isinstance(value, (list, tuple)) else float(value)


def _radius(obj):
    dims = slot_size(obj["shape"], obj["size_m"], obj.get("axis", "z"))
    if obj["shape"] in ("sphere", "bowl") or (obj["shape"] == "cylinder" and obj.get("axis", "z") == "z"):
        return dims[0] / 2 + (obj.get("wall_m", 0) if obj["shape"] == "bowl" else 0)
    return math.hypot(*dims[:2]) / 2


def sample_scene(task, rng, tries=500) -> dict:
    """Sample with the supplied numpy Generator; raise rather than return overlapping scenes.

    Discrete xy_choice slots reserve different spots even for non-colliding target markers.
    Other non-colliding markers and intentional stacks are exempt from footprint separation.
    Failed placement restarts the entire scene, so an unlucky early draw cannot trap later objects.
    """
    errors = validate(task)
    if errors:
        raise ValueError("Invalid task: " + "; ".join(errors))
    rnd = task.get("randomize", {})
    sep = rnd.get("min_sep_m", 0.03)
    table = copy.deepcopy(task.get("table", {"size_m": [0.8, 0.6], "center_xy": [0.2, 0.0],
                                           "color": "#b08a5a", "friction": 0.6}))
    # Presence is drawn once per episode, independent of geometric rejection.
    active = [o for o in task["objects"] if rng.random() < rnd.get(o["id"], {}).get("p_present", 1)]
    for _ in range(tries):
        placed = {}
        pending = list(active)
        failed = False
        while pending:
            ready = next((o for o in pending if "on" not in o or o["on"] in placed), None)
            if ready is None:
                raise ValueError("Present object has an absent or cyclic support")
            pending.remove(ready)
            obj = copy.deepcopy(ready)
            rule = rnd.get(obj["id"], {})
            size = [_draw(rng, v) for v in rule.get("size_m", obj["size_m"])]
            scale = _draw(rng, rule.get("size_scale", 1))
            obj["size_m"] = [v * scale for v in size]
            for key, default in (("mass_kg", 0.03), ("friction", 0.6)):
                obj[key] = _draw(rng, rule.get(key, obj.get(key, default)))
            if "xy_choice" in rule:
                x, y = rule["xy_choice"][int(rng.integers(len(rule["xy_choice"])))]
            elif "r" in rule:
                radius = _draw(rng, rule["r"])
                az = math.radians(_draw(rng, rule["az_deg"]))
                x, y = radius * math.cos(az), radius * math.sin(az)
            elif "x" in rule:
                x, y = _draw(rng, rule["x"]), _draw(rng, rule["y"])
            elif "on" in obj:
                x, y = placed[obj["on"]]["xyz"][:2]
            else:
                x, y = obj.get("xy", obj.get("xyz"))[:2]
            yaw = _draw(rng, rule.get("yaw_deg", obj.get("yaw_deg", 0)))
            if "yaw_rel_deg" in rule:
                yaw = math.degrees(math.atan2(y, x)) + _draw(rng, rule["yaw_rel_deg"])
            height = slot_size(obj["shape"], obj["size_m"], obj.get("axis", "z"))[2]
            z = obj.get("xyz", [0, 0, height / 2])[2]
            if "on" in obj:
                base = placed[obj["on"]]
                z = base["xyz"][2] + slot_size(base["shape"], base["size_m"], base.get("axis", "z"))[2] / 2 + height / 2
            obj.update(xyz=[float(x), float(y), float(z)], yaw_deg=float((yaw + 180) % 360 - 180),
                       fixed=obj.get("fixed", obj["role"] == "fixture"), color=obj.get("color", "gray"))
            if obj["shape"] == "bowl":
                obj["container"] = True
                obj.setdefault("wall_m", 0.005)
            obj.pop("xy", None)
            ancestors = set()
            ancestor = obj.get("on")
            while ancestor in placed:
                ancestors.add(ancestor)
                ancestor = placed[ancestor].get("on")
            for other in placed.values():
                distance = math.dist(obj["xyz"][:2], other["xyz"][:2])
                both_discrete = "xy_choice" in rule and "xy_choice" in rnd.get(other["id"], {})
                if both_discrete:
                    collision = distance < 1e-9
                elif not obj.get("collide", True) or not other.get("collide", True) or other["id"] in ancestors:
                    collision = False
                else:
                    collision = distance < _radius(obj) + _radius(other) + sep
                if collision:
                    failed = True
                    break
            # Match the tabletop convention while allowing explicit custom tables.
            center = table.get("center_xy", [0.2, 0])
            half = table.get("size_m", [0.8, 0.6])
            if any(abs(obj["xyz"][axis] - center[axis]) + _radius(obj) > half[axis] / 2 for axis in (0, 1)):
                failed = True
            if failed:
                break
            placed[obj["id"]] = obj
        if not failed:
            return {"version": 0, "task": task["name"], "robot": copy.deepcopy(task.get("robot", {
                "type": "so101", "joint_convention": "mjcf_rad"})), "table": table,
                "objects": [placed[o["id"]] for o in active], "cameras": copy.deepcopy(task.get("cameras", []))}
    raise ValueError(f"Cannot sample a non-overlapping scene for {task['name']} after {tries} attempts")
