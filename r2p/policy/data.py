"""Trace windows and observation-only DART/pose augmentations.

Labels always come from q_target. A chunk stops at a command boundary and pads
with that command's final target. Gripper transitions are never chatter-smoothed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy

import numpy as np

from r2p.frames import HOME_Q, HORIZON
from r2p.tasks import command_onehot, encode_slots
from r2p.traces import read_trace
from .runtime import expand_features


@dataclass
class Augmentation:
    joint_offset_std: float = np.deg2rad(1.5)
    position_std: float = .0015
    stale_probability: float = .25
    stale_ticks: int = 6
    aim_radial_m: float = .010
    aim_lateral_m: float = .004
    recovery_weight: float = 4.0
    smooth_ticks: int = 3


class TraceDataset:
    def __init__(self, task, directory, config, augmentation=None):
        self.task, self.config = task, config
        self.aug = augmentation or Augmentation()
        files = sorted(Path(directory).rglob("*.jsonl"))
        if not files:
            raise ValueError(f"No .jsonl traces in {directory}")
        raw, labels, weights, previous, pick, starts = [], [], [], [], [], []
        self.provenance = []
        self.physics = None
        for path in files:
            meta, ticks = read_trace(path)
            if meta["task_name"] != task["name"]:
                continue
            if not ticks:
                continue
            from r2p.sim.twin import load_physics
            physics = meta.get("physics_config") or load_physics(meta.get("physics", meta.get("rig") or "so101_bench"))
            if self.physics is None:
                self.physics = deepcopy(physics)
            elif self.physics != physics:
                raise ValueError("Trace physics versions disagree; collect one reference version per export")
            self.provenance.append({"path": str(path), "seed": meta.get("seed"), "success": meta["success"],
                                    "source": meta["source"], "ticks": len(ticks)})
            starts.append(meta.get("start_q", ticks[0]["q"]))
            targets = np.asarray([t["q_target"] for t in ticks], float)
            smoothed = targets.copy()
            radius = self.aug.smooth_ticks//2
            for i, tick in enumerate(ticks):
                indices = [j for j in range(max(0, i-radius), min(len(ticks), i+radius+1))
                           if ticks[j]["command"] == tick["command"] and ticks[j]["phase"] == tick["phase"]]
                smoothed[i, :5] = targets[indices, :5].mean(axis=0)
            base = len(raw)
            for i, tick in enumerate(ticks):
                command = tick.get("command")
                if not command:
                    raise ValueError(f"Unsegmented trace {path}: every tick needs a command")
                q = np.asarray(tick.get("q_commanded", targets[i-1] if i else starts[-1]), float).copy()
                measured = np.asarray(tick["q"], float)
                if np.max(np.abs(q-measured)) > np.deg2rad(4):
                    q = measured.copy()
                raw.append(np.r_[q, encode_slots(task, tick["poses"]).ravel(), command_onehot(task, command)])
                end = i
                while end+1 < min(len(ticks), i+HORIZON) and ticks[end+1]["command"] == command:
                    end += 1
                indices = np.minimum(np.arange(i, i+HORIZON), end)
                labels.append(smoothed[indices].ravel())
                recovery = "recovery" in (tick.get("phase") or "")
                weights.append((self.aug.recovery_weight if recovery else 1.)*(1. if meta["success"] else .25))
                previous.append(base+max(0, i-self.aug.stale_ticks))
                pick.append(command.startswith("pick("))
        if not raw:
            raise ValueError(f"No nonempty traces for task {task['name']!r}")
        self.raw = np.asarray(raw, np.float32)
        self.labels = np.asarray(labels, np.float32)
        self.weights = np.asarray(weights, float)
        self.weights /= self.weights.sum()
        self.previous = np.asarray(previous)
        self.pick = np.asarray(pick)
        starts = np.asarray(starts, float)
        if np.max(np.abs(starts-starts[0])) > .01:
            raise ValueError("Trace start poses disagree; use one declared start_q per export")
        self.start_q = starts[0]
        self.features = expand_features(self.raw, config).astype(np.float32)

    def sample(self, batch_size, rng, augment=True):
        if getattr(self, "_cdf", None) is None:  # rng.choice(p=...) rebuilds an O(N) CDF on every call
            self._cdf = np.cumsum(self.weights)
            self._cdf /= self._cdf[-1]
        index = np.minimum(np.searchsorted(self._cdf, rng.random(batch_size), side="right"), len(self._cdf) - 1)
        x = self.raw[index].copy()
        k = self.config["K"]
        slots = x[:, 6:6+9*k].reshape(-1, k, 9)
        if augment:
            a = self.aug
            x[:, :5] += rng.normal(0, a.joint_offset_std, (len(x), 5))
            stale = rng.random(len(x)) < a.stale_probability
            old = self.raw[self.previous[index], 6:6+9*k].reshape(-1, k, 9)
            for j, obj in enumerate(self.task["objects"]):
                if obj["role"] == "manipulated":
                    slots[stale, j, :5] = old[stale, j, :5]
                    active = self.pick[index] & (slots[:, j, 8] > .5)
                    radial = slots[:, j, :2].copy()
                    radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-6)
                    lateral = np.stack([-radial[:, 1], radial[:, 0]], axis=-1)
                    shift = radial*rng.uniform(0, a.aim_radial_m, (len(x), 1))
                    shift += lateral*rng.normal(0, a.aim_lateral_m, (len(x), 1))
                    slots[active, j, :2] += shift[active]
            slots[:, :, :3] += rng.normal(0, a.position_std, slots[:, :, :3].shape)*slots[:, :, 8:9]
        return expand_features(x, self.config).astype(np.float32), self.labels[index]

    def normalization(self):
        mean = self.features.mean(axis=0, dtype=np.float64)
        std = self.features.std(axis=0, dtype=np.float64)
        constants = [{"index": int(i), "mean": float(mean[i]), "std": float(std[i]),
                      "threshold": float(5*max(std[i], 1e-3))} for i in np.flatnonzero(std < 1e-6)]
        return {"x_mean": mean.tolist(), "x_std": np.maximum(std, .01).tolist(),
                "y_mean": self.labels.mean(axis=0, dtype=np.float64).tolist(),
                "y_std": np.maximum(self.labels.std(axis=0, dtype=np.float64), .05).tolist()}, constants
