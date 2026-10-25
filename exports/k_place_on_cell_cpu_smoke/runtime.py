"""Standalone NumPy inference; this file is copied into every policy export.

No torch, MuJoCo or r2p imports. Joint targets are absolute MJCF radians.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np

SKILLS = ("pick", "place_in", "place_on", "place_at", "push_to", "release", "home", "wait")


def _rotation(axis, angle):
    axis = np.asarray(axis, float)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle)[..., None, None]*skew + (1-np.cos(angle))[..., None, None]*(skew@skew)


def fk_features(q, fk):
    """Exact official-model jaw midpoint + gripper finger axis, in the spec frame."""
    q = np.asarray(q, dtype=float)
    shape = q.shape[:-1]
    positions = [np.zeros(shape+(3,))]
    rotations = [np.broadcast_to(np.eye(3), shape+(3, 3))]
    for body in fk["bodies"]:
        parent = body["parent"]
        R0 = rotations[parent] @ np.asarray(body["rotation"])
        p0 = positions[parent]+np.einsum("...ij,j->...i", rotations[parent], body["pos"])
        j = body["joint"]
        if j >= 0:
            R = R0 @ _rotation(body["axis"], q[..., j])
            anchor = np.asarray(body["anchor"])
            p0 = p0 + np.einsum("...ij,j->...i", R0-R, anchor)
        else:
            R = R0
        positions.append(p0)
        rotations.append(R)
    tips = []
    for tip in fk["tips"]:
        b = tip["body"]
        tips.append(positions[b]+np.einsum("...ij,j->...i", rotations[b], tip["local"]))
    mid = (tips[0]+tips[1])/2-np.asarray(fk["origin"])
    axis = -rotations[fk["tips"][0]["body"]][..., :, 2]
    return np.concatenate([mid, axis], axis=-1)


def command_encoding(config, command):
    match = re.fullmatch(r"([a-z_]+)(?:\((.*)\))?", command.strip())
    if match is None or match[1] not in SKILLS:
        raise ValueError(f"Unknown policy command {command!r}")
    skill = match[1]
    ids = config["object_ids"]
    k = config["K"]
    if skill in SKILLS[:5]:
        first = (match[2] or "").split(",", 1)[0]
        if first not in ids:
            raise ValueError(f"Unknown manipulated object {first!r}")
        slot = ids.index(first)
    else:
        slot = k
    out = np.zeros(len(SKILLS)*(k+1))
    out[SKILLS.index(skill)*(k+1)+slot] = 1
    return out


def expand_features(raw, config):
    raw = np.asarray(raw, float)
    k = config["K"]
    slots = raw[..., 6:6+9*k].reshape(raw.shape[:-1]+(k, 9))
    yaw = np.arctan2(slots[..., 3], slots[..., 4])
    bearing = np.arctan2(slots[..., 1], slots[..., 0])
    sym = np.asarray(config["yaw_sym"])
    present = slots[..., 8]*(sym != 0)
    yaw_features = np.stack([np.sin(yaw*sym), np.cos(yaw*sym),
                             np.sin((yaw-bearing)*sym), np.cos((yaw-bearing)*sym)], axis=-1)
    yaw_features *= present[..., None]
    return np.concatenate([raw, fk_features(raw[..., :6], config["fk"]),
                           yaw_features.reshape(raw.shape[:-1]+(4*k,))], axis=-1)


def observation(obs, config):
    q, slots = np.asarray(obs["q"], float), np.asarray(obs["slots"], float)
    if q.shape != (6,) or slots.shape != (config["K"], 9):
        raise ValueError(f"Expected q(6) and slots({config['K']},9)")
    if not np.isfinite(q).all() or not np.isfinite(slots).all():
        raise ValueError("Observation must contain finite values")
    command = obs.get("command", config["commands"][0])
    raw = np.concatenate([q, slots.ravel(), command_encoding(config, command)])
    return expand_features(raw, config)


def clamp_constants(features, obs_const):
    x = np.array(features, dtype=float, copy=True)
    for item in obs_const:
        index, mean = item["index"], item["mean"]
        threshold = item.get("threshold", 5*max(item["std"], 1e-3))
        x[..., index] = np.where(np.abs(x[..., index]-mean) > threshold, mean, x[..., index])
    return x


def gelu(x):
    return .5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+.044715*x*x*x)))


class Policy:
    def __init__(self, config, norm, weights):
        self.config = config
        self.norm = {k: np.asarray(v, np.float32) for k, v in norm.items()}
        self.weights = {k: np.asarray(v) for k, v in weights.items()}
        self.start_q = np.asarray(config["start_q"], float)
        self.horizon = config["horizon"]
        self._members = int(config.get("ensemble_size", 1))

    @classmethod
    def load(cls, directory):
        path = Path(directory)
        config = json.loads((path/"config.json").read_text(encoding="utf-8"))
        if config.get("action") != "absolute_q" or config.get("horizon") != 15:
            raise ValueError("Only horizon-15 absolute_q exports are supported")
        norm = json.loads((path/"norm.json").read_text(encoding="utf-8"))
        with np.load(path/"weights.npz", allow_pickle=False) as archive:
            weights = dict(archive)
        return cls(config, norm, weights)

    def features(self, obs):
        return clamp_constants(observation(obs, self.config), self.config["obs_const"])

    def _linear(self, x, prefix):
        return x@self.weights[prefix+".weight"].T+self.weights[prefix+".bias"]

    def predict_features(self, features, *, ensemble=False):
        x = ((clamp_constants(features, self.config["obs_const"])-self.norm["x_mean"])/self.norm["x_std"]).astype(np.float32)
        outputs = []
        for member in range(self._members):
            prefix = f"member{member}."
            h = gelu(self._linear(x, prefix+"input"))
            for block in range(self.config["blocks"]):
                h = gelu(h+self._linear(gelu(self._linear(h, prefix+f"blocks.{block}.first")),
                                        prefix+f"blocks.{block}.second"))
            y = self._linear(h, prefix+"output")*self.norm["y_std"]+self.norm["y_mean"]
            y = y.reshape(y.shape[:-1]+(15, 6))
            limits = np.asarray(self.config["joint_limits"])
            outputs.append(np.clip(y, limits[:, 0], limits[:, 1]))
        return np.stack(outputs) if ensemble else np.mean(outputs, axis=0)

    def act(self, obs):
        return self.predict_features(self.features(obs))

    def act_ensemble(self, obs):
        return self.predict_features(self.features(obs), ensemble=True)
