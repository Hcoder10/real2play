"""Shared language, state, normalization and chunk contracts (no model imports)."""
from __future__ import annotations
import re
import numpy as np

CAMERAS = ("observation.images.scene", "observation.images.wrist")
STATE = "observation.state"
HORIZON, HZ = 15, 30


def instruction(command, target_cell=None):
    """The same deterministic Argon-command translation in training and runtime."""
    if isinstance(command, dict):
        skill = command["skill"]
        args = [command[k] for k in ("obj", "target", "xy") if k in command]
        command = skill + ("(" + ",".join(map(str, args)) + ")" if args else "")
    value = str(command).strip()
    if value in ("release", "release()"):
        return "Release the piece."
    if value in ("home", "home()"):
        return "Return the arm to its starting pose."
    m = re.fullmatch(r"pick\((piece)\)", value)
    if m:
        return f"Pick up the {m[1]}."
    m = re.fullmatch(r"place_at\((piece),(cell|r[0-2]c[0-2])\)", value.replace(" ", ""))
    if m:
        cell = target_cell if m[2] == "cell" else m[2]
        if not re.fullmatch(r"r[0-2]c[0-2]", cell or ""):
            raise ValueError("place_at(piece,cell) requires the actual target_cell")
        return f"Place the {m[1]} on cell {cell}."
    raise ValueError(f"Unsupported Argon VLA command: {value}")


def target_cell(scene):
    objects = {o["id"]: o for o in scene["objects"]}
    target = np.asarray(objects["cell"]["xyz"][:2])
    cells = {k: o for k, o in objects.items() if re.fullmatch(r"r[0-2]c[0-2]", k)}
    if not cells:
        raise ValueError("scene has no named board cells")
    return min(cells, key=lambda k: np.linalg.norm(np.asarray(cells[k]["xyz"][:2])-target))


def commanded_state(commanded, measured, age_s):
    commanded, measured = np.asarray(commanded, float), np.asarray(measured, float)
    if commanded.shape != (6,) or measured.shape != (6,):
        raise ValueError("six MJCF joint angles required")
    if not np.isfinite(commanded).all() or not np.isfinite(measured).all():
        raise ValueError("joint observations must be finite")
    reset = bool(np.max(np.abs(commanded-measured)) > np.deg2rad(4))
    state = np.r_[measured if reset else commanded, age_s].astype(np.float32)
    if not np.isfinite(state).all() or age_s < 0:
        raise ValueError("non-finite state or negative command age")
    return state, reset


def action_chunk(ticks, index):
    """Pad at the command boundary; never use executed/DART targets as labels."""
    command = ticks[index]["command"]
    rows = []
    for i in range(index, min(index+HORIZON, len(ticks))):
        if ticks[i]["command"] != command:
            break
        rows.append(ticks[i]["q_target"])
    rows.extend([rows[-1]]*(HORIZON-len(rows)))
    return np.asarray(rows, np.float32)


class Normalizer:
    def __init__(self, stats):
        self.stats = stats

    @classmethod
    def fit(cls, state, action):
        result = {}
        for key, x in ((STATE, state), ("action", action.reshape(-1, 6))):
            x = np.asarray(x, np.float64)
            result[key] = {"mean": x.mean(0).tolist(), "std": x.std(0).tolist(),
                           "min": x.min(0).tolist(), "max": x.max(0).tolist(), "count": len(x)}
        return cls(result)

    def normalize(self, key, x, clamp_constants=False):
        cfg = self.stats[key]
        mean, std = np.asarray(cfg["mean"]), np.asarray(cfg["std"])
        x = np.asarray(x).copy()
        if clamp_constants:
            mask = (std < 1e-6) & (np.abs(x-mean) > 5*np.maximum(std, 1e-3))
            x = np.where(mask, mean, x)
        return ((x-mean)/np.maximum(std, 1e-3)).astype(np.float32)

    def action(self, x):
        cfg = self.stats["action"]
        return np.asarray(x)*np.maximum(cfg["std"], 1e-3)+cfg["mean"]
