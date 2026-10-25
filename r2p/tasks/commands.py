"""Stable command conditioning: skill-major, then manipulated object-slot index.

For K slots there are 8 * (K+1) coordinates. The final slot in each skill block
means no object (release/home/wait). Target slots and numeric arguments remain
in the task/poses; changing a target never changes the command vector's shape.
"""
from __future__ import annotations

import json
import re
import numpy as np

from .spec import SKILLS


def _arg(value):
    return value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))


def commands(task) -> list[str]:
    result = []
    for step in task["skeleton"]:
        skill = step["skill"]
        args = [_arg(step[key]) for key in SKILLS[skill]]
        result.append(skill + ("(" + ",".join(args) + ")" if args else ""))
    return result


def command_vocab(task) -> list[str]:
    """Export labels are independent of object names, skeleton order and target choice."""
    k = task["limits"]["max_objects"]
    return [f"{skill}(slot:{i})" if i < k else f"{skill}(none)" for skill in SKILLS for i in range(k + 1)]


def _parse(command):
    match = re.fullmatch(r"([a-z_]+)(?:\((.*)\))?", command.strip())
    if match is None or match[1] not in SKILLS:
        raise ValueError(f"Unknown command: {command!r}")
    args, depth, start = [], 0, 0
    body = match[2] or ""
    for i, char in enumerate(body):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        elif char == "," and depth == 0:
            args.append(body[start:i].strip())
            start = i + 1
    if body:
        args.append(body[start:].strip())
    if depth != 0 or len(args) != len(SKILLS[match[1]]):
        raise ValueError(f"Wrong arguments for command: {command!r}")
    return match[1], dict(zip(SKILLS[match[1]], args))


def command_onehot(task, cmd) -> np.ndarray:
    if isinstance(cmd, dict):
        cmd = commands({"skeleton": [cmd]})[0]
    if not isinstance(cmd, str):
        raise ValueError("command must be a skill string or skeleton entry")
    skill, args = _parse(cmd)
    ids = [obj["id"] for obj in task["objects"]]
    for field in ("obj", "target"):
        if field in args and args[field] not in ids:
            raise ValueError(f"Unknown object in command: {args[field]}")
    if "xy" in args and args["xy"] not in ids:
        try:
            xy = np.asarray(json.loads(args["xy"]), dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError("command xy must be an object id or [x,y]") from exc
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("command xy must be a finite [x,y]")
    if "s" in args:
        try:
            seconds = float(args["s"])
        except ValueError as exc:
            raise ValueError("wait requires finite nonnegative seconds") from exc
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError("wait requires finite nonnegative seconds")
    k = task["limits"]["max_objects"]
    slot = ids.index(args["obj"]) if "obj" in args else k
    out = np.zeros(len(SKILLS) * (k + 1), dtype=np.float32)
    out[list(SKILLS).index(skill) * (k + 1) + slot] = 1
    return out
