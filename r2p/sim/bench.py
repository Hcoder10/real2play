"""Opt-in real-bench training convention. Never rewrite an existing export."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from r2p.frames import HOME_Q
from r2p.model import Arm
from r2p.robot.limits import OwnerLimits


def make_profile(state, joint_map, *, margin_deg=2):
    bounds = OwnerLimits.from_state(state).model_bounds(joint_map, Arm().range, margin_deg)
    # Match the roll actually observed during the open-jaw visual calibration.
    roll = joint_map.to_model([0, 0, 0, 0, 81.9, 0])[4]
    opening = joint_map.to_model([0, 0, 0, 0, 0, 20])[5]
    squeeze = joint_map.to_model([0, 0, 0, 0, 0, 6])[5]
    start = np.clip(HOME_Q, bounds[:, 0], bounds[:, 1])
    start[3] = bounds[3, 1] - np.deg2rad(4)  # leave room for feedforward
    start[4], start[5] = roll, opening
    return {"schema": "r2p_bench_policy/1", "joint_map": joint_map.name,
            "joint_map_parameters": {"sign": joint_map.sign.tolist(), "offset": joint_map.offset.tolist(),
                                     "closed_rad": joint_map.gc, "open_rad": joint_map.go},
            "owner_state": deepcopy(state), "margin_deg": margin_deg,
            "model_limits": bounds.tolist(), "start_q": start.tolist(), "roll_rad": float(roll),
            "gripper_open_rad": float(opening), "gripper_close_rad": float(squeeze),
            "gripper_open_threshold_rad": float((opening+squeeze)/2),
            "travel_height_m": .05, "joint_step_rad": float(np.deg2rad(4)/30),
            "notes": "Round upright caps only; locked calibrated roll. Requires fresh traces and training; not a policy certification."}


def load_profile(value):
    profile = deepcopy(value) if isinstance(value, dict) else json.loads(Path(value).read_text(encoding="utf-8"))
    if profile.get("schema") != "r2p_bench_policy/1":
        raise ValueError("expected r2p_bench_policy/1 profile")
    bounds, start = np.asarray(profile["model_limits"], float), np.asarray(profile["start_q"], float)
    if bounds.shape != (6, 2) or start.shape != (6,) or not np.isfinite(bounds).all() or not np.isfinite(start).all():
        raise ValueError("invalid bench bounds/start_q")
    if np.any(bounds[:, 0] >= bounds[:, 1]) or np.any(start < bounds[:, 0]) or np.any(start > bounds[:, 1]):
        raise ValueError("bench start_q outside limits")
    if not bounds[4, 0] <= profile["roll_rad"] <= bounds[4, 1]:
        raise ValueError("bench wrist roll outside limits")
    return profile
