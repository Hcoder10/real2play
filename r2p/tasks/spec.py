"""Validation of the closed argon_task/1 vocabulary; no engine dependencies."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

SCHEMA = "argon_task/1"
SHAPES = {"box": 3, "cylinder": 2, "sphere": 1, "bowl": 2, "plate": 3, "stick": 2}
ROLES = ("manipulated", "container", "target", "obstacle", "fixture")
GRASPS = ("top", "side", "along")
PREDICATES = {
    "inside": ("a", "b"), "on_top": ("a", "b"), "at": ("a", "xy", "tol_m"),
    "near": ("a", "b", "d_m"), "lifted": ("a", "h_m"), "grasped": ("a",),
    "upright": ("a", "tol_deg"), "yaw": ("a", "deg", "tol_deg"),
    "not_moved": ("a", "tol_m"), "stacked": ("objs",), "gripper_open": (),
}
# Tuple order is a public part of the command encoding contract.
SKILLS = {
    "pick": ("obj",), "place_in": ("obj", "target"), "place_on": ("obj", "target"),
    "place_at": ("obj", "xy"), "push_to": ("obj", "xy"),
    "release": (), "home": (), "wait": ("s",),
}
RANDOMIZE_KEYS = {
    "r", "az_deg", "x", "y", "yaw_deg", "yaw_rel_deg", "size_scale", "size_m",
    "mass_kg", "friction", "xy_choice", "p_present",
}
TASK_ROOT = Path(__file__).resolve().parents[2] / "tasks"


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def vector(value, n):
    return isinstance(value, (list, tuple)) and len(value) == n and all(number(x) for x in value)


def validate(task) -> list[str]:
    """Return all detected errors, including for malformed JSON-shaped input."""
    errors = []
    if not isinstance(task, dict):
        return ["task must be an object"]
    if task.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA}")
    if not isinstance(task.get("name"), str) or not task["name"].strip():
        errors.append("name must be a nonempty string")
    objects = task.get("objects")
    if not isinstance(objects, list) or not objects:
        errors.append("objects must be a nonempty list")
        objects = []
    ids = set()
    by_id = {}
    for index, obj in enumerate(objects):
        label = f"objects[{index}]"
        if not isinstance(obj, dict):
            errors.append(f"{label} must be an object")
            continue
        oid = obj.get("id")
        if not isinstance(oid, str) or not oid or any(c in oid for c in "(),[] "):
            errors.append(f"{label}.id must be a nonempty command-safe string")
        elif oid in ids:
            errors.append(f"duplicate object id: {oid}")
        else:
            ids.add(oid)
            by_id[oid] = obj
        shape = obj.get("shape")
        n = SHAPES.get(shape) if isinstance(shape, str) else None
        if n is None:
            errors.append(f"{label}.shape is not in the closed vocabulary")
        elif not vector(obj.get("size_m"), n) or any(x <= 0 for x in obj["size_m"]):
            errors.append(f"{label}.size_m needs {n} positive finite dimensions")
        if obj.get("role") not in ROLES:
            errors.append(f"{label}.role is not in the closed vocabulary")
        if "grasp" in obj and obj["grasp"] not in GRASPS:
            errors.append(f"{label}.grasp must be top, side or along")
        if "axis" in obj and obj["axis"] not in ("x", "y", "z"):
            errors.append(f"{label}.axis must be x, y or z")
        for field in ("fixed", "collide"):
            if field in obj and not isinstance(obj[field], bool):
                errors.append(f"{label}.{field} must be boolean")
        for field in ("mass_kg", "friction", "wall_m"):
            if field in obj and (not number(obj[field]) or obj[field] < 0):
                errors.append(f"{label}.{field} must be finite and nonnegative")
        for field, length in (("xyz", 3), ("xy", 2)):
            if field in obj and not vector(obj[field], length):
                errors.append(f"{label}.{field} must have {length} finite coordinates")
        if "yaw_deg" in obj and not number(obj["yaw_deg"]):
            errors.append(f"{label}.yaw_deg must be finite")

    def ref(value, label):
        if not isinstance(value, str) or value not in ids:
            errors.append(f"{label} references an unknown object: {value!r}")

    def xy(value, label):
        if isinstance(value, str):
            ref(value, label)
        elif not vector(value, 2):
            errors.append(f"{label} must be [x,y] or an object id")

    for field, key, vocabulary in (("goal", "pred", PREDICATES), ("skeleton", "skill", SKILLS)):
        entries = task.get(field)
        if not isinstance(entries, list) or not entries:
            errors.append(f"{field} must be a nonempty list")
            continue
        for index, entry in enumerate(entries):
            label = f"{field}[{index}]"
            if not isinstance(entry, dict):
                errors.append(f"{label} must be an object")
                continue
            kind = entry.get(key)
            if not isinstance(kind, str) or kind not in vocabulary:
                errors.append(f"{label}.{key} is not in the closed vocabulary")
                continue
            for required in vocabulary[kind]:
                if required not in entry:
                    errors.append(f"{label} needs {required}")
            for name in ("a", "b", "obj", "target"):
                if name in entry:
                    ref(entry[name], f"{label}.{name}")
            if "xy" in entry:
                xy(entry["xy"], f"{label}.xy")
            if "objs" in entry:
                chain = entry["objs"]
                if not isinstance(chain, list) or len(chain) < 2:
                    errors.append(f"{label}.objs must list at least two objects bottom-first")
                else:
                    for oid in chain:
                        ref(oid, f"{label}.objs")
                    if len({repr(oid) for oid in chain}) != len(chain):
                        errors.append(f"{label}.objs must not repeat objects")
            for name in ("tol_m", "d_m", "tol_deg", "h_m", "s", "deg"):
                if name in entry:
                    value = entry[name]
                    if not number(value) or (name != "deg" and value < 0) or (name in ("tol_m", "d_m", "tol_deg") and value == 0):
                        errors.append(f"{label}.{name} has an invalid numeric value")
            if "optional" in entry and not isinstance(entry["optional"], bool):
                errors.append(f"{label}.optional must be boolean")
            if "approach" in entry and entry["approach"] not in ("top", "side"):
                errors.append(f"{label}.approach must be top or side")

    limits = task.get("limits")
    if not isinstance(limits, dict):
        errors.append("limits must be an object")
    else:
        capacity = limits.get("max_objects")
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < len(objects):
            errors.append("limits.max_objects must be an integer >= len(objects)")
        if not number(limits.get("episode_s")) or limits["episode_s"] <= 0:
            errors.append("limits.episode_s must be positive")
    rnd = task.get("randomize", {})
    if not isinstance(rnd, dict):
        errors.append("randomize must be an object")
        rnd = {}
    if "min_sep_m" in rnd and (not number(rnd["min_sep_m"]) or rnd["min_sep_m"] < 0):
        errors.append("randomize.min_sep_m must be nonnegative")

    def interval(value, label, lower=None, positive=False):
        vals = value if isinstance(value, (list, tuple)) else [value, value]
        if not vector(vals, 2) or vals[0] > vals[1]:
            errors.append(f"{label} must be finite or an ordered [low,high] range")
        elif (lower is not None and vals[0] < lower) or (positive and vals[0] <= 0):
            errors.append(f"{label} has an invalid lower bound")

    if "board_pose" in rnd:
        pose = rnd["board_pose"]
        if not isinstance(pose, dict) or not isinstance(task.get("board"), dict) or set(pose) - {"dx", "dy", "yaw_deg"}:
            errors.append("randomize.board_pose needs a board and only dx, dy, yaw_deg")
        else:
            for key, value in pose.items():
                interval(value, f"randomize.board_pose.{key}")
    for oid, entry in rnd.items():
        if oid in ("min_sep_m", "board_pose"):
            continue
        ref(oid, "randomize")
        if not isinstance(entry, dict):
            errors.append(f"randomize.{oid} must be an object")
            continue
        for key, value in entry.items():
            label = f"randomize.{oid}.{key}"
            if key not in RANDOMIZE_KEYS:
                errors.append(f"{label} is not in the closed vocabulary")
            elif key == "p_present":
                if not number(value) or not 0 <= value <= 1:
                    errors.append(f"{label} must be in [0,1]")
            elif key == "xy_choice":
                if not isinstance(value, list) or not value or not all(vector(p, 2) for p in value):
                    errors.append(f"{label} must be a nonempty list of [x,y]")
            elif key == "size_m":
                shape = by_id.get(oid, {}).get("shape")
                n = SHAPES.get(shape) if isinstance(shape, str) else None
                if not isinstance(value, list) or len(value) != n:
                    errors.append(f"{label} must match the shape dimensions")
                else:
                    for dim in value:
                        interval(dim, label, positive=True)
            else:
                interval(value, label, lower=0 if key in ("r", "mass_kg", "friction") else None,
                         positive=key == "size_scale")
        for pair in (("r", "az_deg"), ("x", "y")):
            if (pair[0] in entry) != (pair[1] in entry):
                errors.append(f"randomize.{oid} requires {pair[0]} and {pair[1]} together")
        if sum(key in entry for key in ("r", "x", "xy_choice")) > 1:
            errors.append(f"randomize.{oid} has conflicting position distributions")
        if "yaw_deg" in entry and "yaw_rel_deg" in entry:
            errors.append(f"randomize.{oid} has conflicting yaw distributions")
    for oid, obj in by_id.items():
        entry = rnd.get(oid, {})
        entry = entry if isinstance(entry, dict) else {}
        if entry.get("p_present") != 0 and not any(key in obj for key in ("xy", "xyz", "on")) and not any(key in entry for key in ("r", "x", "xy_choice")):
            errors.append(f"{oid} needs xy/xyz/on or a randomized position")
        if "on" in obj:
            ref(obj["on"], f"{oid}.on")
            visited = {oid}
            parent = obj["on"]
            while isinstance(parent, str) and parent in by_id:
                if parent in visited:
                    errors.append(f"{oid}.on has a support cycle")
                    break
                visited.add(parent)
                parent = by_id[parent].get("on")
    return errors


def load_task(name_or_path) -> dict:
    """Load by registered name, directory, JSON path or dict; validate and return a copy."""
    if isinstance(name_or_path, dict):
        task = copy.deepcopy(name_or_path)
    else:
        path = Path(name_or_path)
        if not path.exists():
            path = TASK_ROOT / path
        if path.is_dir():
            path = path / "task.json"
        task = json.loads(path.read_text(encoding="utf-8-sig"))
    errors = validate(task)
    if errors:
        raise ValueError("Invalid task:\n" + "\n".join(errors))
    return task
