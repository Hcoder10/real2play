"""Closed-loop evaluation from declared start_q, on identical seeded scenes.

python -m r2p.policy.eval --export NAME --n 100 --noise clean
Noise combines persistent joint calibration error, pose error, stale observations
and one extra action-latency tick. The clean twin retains its calibrated latency.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from r2p.frames import HZ, ROOT
from r2p.model import Arm
from r2p.tasks import check, check_window, commands, encode_slots, sample_scene
from r2p.sim.twin import TaskTwin
from . import load

NOISE = {"clean": 0., "real": 1., "realx0.5": .5}


class CommandTracker:
    """State-based subtask completion; no expert actions or hidden waypoint labels."""
    def __init__(self, task, initial, *, pick_lift_m=.04):
        self.task = task
        self.initial = initial
        self.vocab = commands(task)
        self.index = 0
        self.age = 0
        self.stable = 0
        self.pick_lift_m = float(pick_lift_m)

    @property
    def command(self):
        return self.vocab[min(self.index, len(self.vocab)-1)]

    @property
    def done(self):
        return self.index >= len(self.vocab)

    def update(self, state, q, start_q):
        if self.done:
            return
        self.age += 1
        step = self.task["skeleton"][self.index]
        skill = step["skill"]
        obj = step.get("obj")
        poses = state["objects"]
        complete = False
        if skill == "pick" and obj in poses and obj in self.initial:
            complete = not state["gripper_open"] and poses[obj]["xyz"][2]-self.initial[obj]["xyz"][2] > self.pick_lift_m
        elif skill in ("place_in", "place_on", "place_at", "push_to"):
            goals = [g for g in self.task["goal"] if g.get("a") == obj]
            if not goals:
                pred = {"place_in": "inside", "place_on": "on_top", "place_at": "at", "push_to": "at"}[skill]
                goal = {"pred": pred, "a": obj}
                goal.update({"b": step["target"]} if "target" in step else {"xy": step["xy"], "tol_m": .012})
                goals = [goal]
            complete = state["gripper_open"] and check(dict(self.task, goal=goals), state)["ok"]
        elif skill == "release":
            complete = state["gripper_open"]
        elif skill == "home":
            complete = np.max(np.abs(np.asarray(q)[:5]-np.asarray(start_q)[:5])) < .12
        elif skill == "wait":
            complete = self.age >= max(1, round(step["s"]*HZ))
        self.stable = self.stable+1 if complete else 0
        if self.stable >= 4:
            self.index += 1
            self.age, self.stable = 0, 0


def rollout(policy, scene, *, seed=0, noise="clean", max_ticks=None, on_step=None):
    if noise not in NOISE:
        raise ValueError(f"noise must be one of {tuple(NOISE)}")
    scale = NOISE[noise]
    task = policy.config["task_spec"]
    if callable(getattr(policy, "reset", None)):
        policy.reset()
    rng = np.random.default_rng(seed)
    arm = Arm()
    states, chunks = deque(maxlen=15), deque(maxlen=15)
    slot_history = deque(maxlen=8)
    start_q = np.asarray(policy.config["start_q"], float)
    # Offset is a calibration error: perceived q = physical q + offset; the same
    # mistaken map converts a model-space target back to physical joint space.
    offset = np.r_[rng.normal(0, np.deg2rad(1.5)*scale, 5), 0]
    pose_bias = rng.normal(0, .002*scale, (policy.config["K"], 3))
    pending = deque([start_q.copy()] if scale and rng.random() < scale else [])
    budget = max_ticks if max_ticks is not None else int(task["limits"]["episode_s"]*HZ)
    if budget < 1:
        raise ValueError("max_ticks must be positive")
    with TaskTwin(scene, physics=policy.config.get("physics", "so101_bench"), q0=start_q) as twin:
        initial = twin.poses()
        tracker = CommandTracker(task, initial, pick_lift_m=policy.config.get("pick_lift_m", .04))
        commanded = start_q.copy()
        last_sent = start_q.copy()
        divergences = 0
        for tick in range(budget):
            measured = twin.q()+offset
            if np.max(np.abs(commanded-measured)) > np.deg2rad(4):
                commanded = measured.copy()
                chunks.clear()
                divergences += 1
            truth = twin.poses()
            slots = encode_slots(task, truth)
            slot_history.append(slots.copy())
            if scale and rng.random() < .3*scale and len(slot_history) > 1:
                old = slot_history[int(rng.integers(len(slot_history)))]
                for j, obj in enumerate(task["objects"]):
                    if obj["role"] == "manipulated":
                        slots[j, :5] = old[j, :5]
            slots[:, :3] += (pose_bias+rng.normal(0, .0015*scale, pose_bias.shape))*slots[:, 8:9]
            q_input = commanded.copy()
            chunk = np.asarray(policy.act({"q": q_input, "slots": slots, "command": tracker.command,
                                           "tick": tick}))
            if chunk.shape != (15, 6) or not np.isfinite(chunk).all():
                return {"success": False, "reason": "invalid_action", "ticks": tick, "resets": divergences}
            chunks.append((tick, chunk))
            predictions = np.stack([prediction[tick-time_] for time_, prediction in chunks])
            weights = np.exp(-.1*np.arange(len(predictions)))  # oldest predictions weighted most
            target = np.average(predictions, axis=0, weights=weights)
            # Match the runner's bounded absolute-command execution.
            # Observation resets must not reset the actuator's rate-limit origin.
            # The real Session likewise projects from last_sent, not q_obs.
            target = last_sent+np.clip(target-last_sent, -.10, .10)
            target = np.clip(target, *arm.range.T)
            last_sent = target.copy()
            pending.append(target.copy())
            executed = pending.popleft()
            twin.step(executed-offset)
            commanded = executed.copy()
            threshold = policy.config.get("gripper_open_threshold_rad")
            opened = (not twin.grip_closed() if threshold is None else
                      bool(twin.q()[5] > threshold and twin.commanded_q[5] > threshold))
            state = {"objects": twin.poses(), "initial": initial, "gripper_open": opened,
                     "site_xyz": arm.tips(twin.q())[2].tolist()}
            if on_step is not None:
                on_step({"tick": tick, "command": tracker.command, "q": measured.copy(),
                         "q_input": q_input, "target": target.copy(),
                         "chunk": chunk.copy(), "state": state, "resets": divergences})
            states.append(state)
            old_command = tracker.command
            tracker.update(state, twin.q(), start_q)
            if old_command != tracker.command:
                chunks.clear()
            result = check_window(task, list(states), hz=HZ)
            if result["ok"]:
                return {"success": True, "reason": "goal_reached", "ticks": tick+1, "resets": divergences,
                        "commands_completed": tracker.index, "check": result}
        return {"success": False, "reason": "timeout", "ticks": budget, "resets": divergences,
                "commands_completed": tracker.index, "check": check_window(task, list(states), hz=HZ)}


def evaluate(export, n=100, noise="clean", *, seed=10000, max_ticks=None, record=True):
    if n < 1:
        raise ValueError("n must be positive")
    if noise not in NOISE:
        raise ValueError(f"noise must be one of {tuple(NOISE)}")
    directory = Path(export)
    if not directory.is_dir():
        directory = ROOT/"exports"/str(export)
    policy = load(directory)
    outcomes, failures = [], Counter()
    started = time.perf_counter()
    for i in range(n):
        scene = sample_scene(policy.config["task_spec"], np.random.default_rng(seed+i))
        result = rollout(policy, scene, seed=seed+i, noise=noise, max_ticks=max_ticks)
        outcomes.append(result)
        if not result["success"]:
            failures[result["reason"]] += 1
    report = {"n": n, "noise": noise, "seed": seed, "scene_seeds": list(range(seed, seed+n)),
              "success": sum(r["success"] for r in outcomes)/n, "failures": dict(failures),
              "max_ticks": max_ticks, "seconds": time.perf_counter()-started, "episodes": outcomes,
              "weights_sha256": hashlib.sha256((directory/"weights.npz").read_bytes()).hexdigest(),
              "physics_sha256": policy.config.get("physics_sha256"),
              "noise_parameters": {"joint_offset_std_deg": 1.5*NOISE[noise], "pose_bias_std_m": .002*NOISE[noise],
                                   "pose_std_m": .0015*NOISE[noise], "stale_probability": .3*NOISE[noise],
                                   "additional_latency_probability": NOISE[noise]}}
    if record:
        config_path = directory/"config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config.setdefault("eval", {})[noise] = report
        config["eval"]["n"] = n
        temp = config_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(config, indent=2), encoding="utf-8")
        temp.replace(config_path)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export", required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--noise", choices=NOISE, default="clean")
    p.add_argument("--seed", type=int, default=10000)
    a = p.parse_args()
    print(json.dumps(evaluate(a.export, a.n, a.noise, seed=a.seed), indent=2))


if __name__ == "__main__":
    main()
