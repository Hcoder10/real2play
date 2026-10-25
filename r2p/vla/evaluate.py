"""Batched physical closed-loop VLA evaluation using the shared task checker."""
from __future__ import annotations
import argparse
from collections import deque
import json
from pathlib import Path
import time
import numpy as np
from r2p.frames import HZ
from r2p.tasks import sample_scene, check_window
from r2p.sim.twin import TaskTwin
from r2p.policy.eval import CommandTracker
from .contracts import commanded_state, instruction, target_cell
from .render import Cameras


class Episode:
    def __init__(self, config, seed, noise, replan):
        self.config, self.seed, self.noise, self.replan = config, seed, noise, replan
        self.rng = np.random.default_rng(seed)
        self.scene = sample_scene(config["task_spec"], np.random.default_rng(seed))
        self.start = np.asarray(config["start_q"])
        self.twin = TaskTwin(self.scene, physics=config["physics"], q0=self.start)
        self.cameras = Cameras(self.scene, config["physics"], config["camera"], config["mounts"])
        self.initial = self.twin.poses()
        self.tracker = CommandTracker(config["task_spec"], self.initial)
        self.target_cell = target_cell(self.scene)
        self.last_sent = self.start.copy()
        self.offset = np.r_[self.rng.normal(0, np.deg2rad(1.5), 5), 0] if noise == "real" else np.zeros(6)
        self.pending = deque([self.start.copy()] if noise == "real" else [])
        self.chunks, self.states = deque(), deque(maxlen=15)
        self.visual_history = deque([(self.twin.q(), self.initial)], maxlen=8)
        self.command_start, self.resets = 0, 0
        self.done = False
        self.result = None
        self.ticks = 0
        self.last_observed_tick = -1

    def observe(self, tick):
        q = self.twin.q()+self.offset
        state, reset = commanded_state(self.last_sent, q, (tick-self.command_start)/HZ)
        self.resets += int(reset)
        visual_q, visual_poses = self.twin.q(), self.twin.poses()
        if self.noise == "real" and self.rng.random() < .3 and len(self.visual_history) > 1:
            visual_q, visual_poses = self.visual_history[int(self.rng.integers(len(self.visual_history)))]
        images = self.cameras.render(visual_q, visual_poses)
        if self.noise == "real":
            import cv2
            gain = self.rng.uniform(.9, 1.1)
            images = np.clip(images.astype(np.float32)*gain+self.rng.normal(0, 2, images.shape), 0, 255).astype(np.uint8)
            if self.rng.random() < .2:
                images = np.stack([cv2.GaussianBlur(x, (3, 3), .6) for x in images])
        self.last_observed_tick = tick
        return state, images, instruction(self.tracker.command, self.target_cell)

    def step(self, tick, max_ticks):
        self.chunks = deque((start, chunk) for start, chunk in self.chunks if tick-start < 15)
        predictions = np.stack([chunk[tick-start] for start, chunk in self.chunks])
        weights = np.exp(-.1*np.arange(len(predictions)))
        target = np.average(predictions, axis=0, weights=weights)
        target = self.last_sent+np.clip(target-self.last_sent, -.10, .10)
        target = np.clip(target, *self.twin.range.T)
        self.last_sent = target.copy()
        self.pending.append(target)
        executed = self.pending.popleft()
        self.twin.step(executed-self.offset)
        self.visual_history.append((self.twin.q(), self.twin.poses()))
        state = {"objects": self.twin.poses(), "initial": self.initial,
                 "gripper_open": not self.twin.grip_closed()}
        self.states.append(state)
        old = self.tracker.command
        self.tracker.update(state, self.twin.q(), self.start)
        if self.tracker.command != old:
            self.command_start = tick+1
            self.chunks.clear()
        result = check_window(self.config["task_spec"], list(self.states), hz=HZ)
        self.ticks = tick+1
        if result["ok"] or self.ticks >= max_ticks:
            self.done = True
            self.result = {"seed": self.seed, "success": bool(result["ok"]), "ticks": self.ticks,
                           "reason": "goal_reached" if result["ok"] else "timeout", "commands_completed": self.tracker.index,
                           "observation_resets": self.resets, "check": result}

    def close(self):
        self.cameras.close()
        self.twin.close()


def evaluate(policy, n=30, noise="clean", seed=10000, batch_size=6, replan=15, max_ticks=1200, gpu_duty=.60):
    if noise not in ("clean", "real") or n < 1 or not 1 <= replan <= 15 or batch_size < 1:
        raise ValueError("invalid evaluation settings")
    train = json.loads((policy.directory/"dataset_manifest.json").read_text())
    training_seeds = {r["seed"] for r in train["episodes"]}
    if training_seeds & set(range(seed, seed+n)):
        raise ValueError("evaluation seeds overlap training episodes")
    import torch
    torch.manual_seed(seed)
    started = time.monotonic()
    outcomes, latency = [], []
    for first in range(0, n, batch_size):
        episodes = []
        try:
            episodes = [Episode(policy.config, seed+i, noise, replan) for i in range(first, min(first+batch_size, n))]
            for tick in range(max_ticks):
                active = [e for e in episodes if not e.done]
                if not active:
                    break
                replanning = [e for e in active if tick % replan == 0 or not e.chunks]
                if replanning:
                    rows = [e.observe(tick) for e in replanning]
                    start = time.monotonic()
                    chunks = policy.predict(np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows]), [r[2] for r in rows])
                    elapsed = time.monotonic()-start
                    latency.append({"batch": len(rows), "seconds": elapsed})
                    if policy.device.type == "cuda":
                        time.sleep(elapsed*(1/gpu_duty-1))
                    for ep, chunk in zip(replanning, chunks):
                        ep.chunks.append((tick, chunk))
                for ep in active:
                    ep.step(tick, max_ticks)
            for ep in episodes:
                outcomes.append(ep.result)
                print(json.dumps(ep.result), flush=True)
        finally:
            for ep in episodes:
                ep.close()
    report = {"n": n, "noise": noise, "scene_seeds": list(range(seed, seed+n)),
              "successes": sum(r["success"] for r in outcomes), "success": sum(r["success"] for r in outcomes)/n,
              "episodes": outcomes, "seconds": time.monotonic()-started, "max_ticks": max_ticks,
              "replan_every": replan, "inference_seconds_mean": float(np.mean([r["seconds"] for r in latency])),
              "inference_calls": len(latency), "weights_sha256": policy.config["weights_sha256"],
              "sampling_seed": seed,
              "scheduler": "simulation-only state-based CommandTracker; no live Gemini calls",
              "noise_parameters": {} if noise == "clean" else {"joint_offset_std_deg": 1.5, "pixel_noise_std": 2,
                  "brightness": [.9, 1.1], "stale_probability": .3, "stale_max_ticks": 7,
                  "blur_probability": .2, "additional_latency_ticks": 1}}
    path = policy.directory/"config.json"
    cfg = json.loads(path.read_text())
    cfg.setdefault("eval", {})[noise] = report
    path.write_text(json.dumps(cfg, indent=2))
    (policy.directory/f"eval_{noise}.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export", required=True)
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--noise", choices=("clean", "real", "both"), default="both")
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--memory-fraction", type=float)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--replan", type=int, default=15)
    p.add_argument("--max-ticks", type=int, default=1200)
    a = p.parse_args()
    from .runtime import Policy
    policy = Policy.load(a.export, a.device, a.memory_fraction)
    for noise in ("clean", "real") if a.noise == "both" else (a.noise,):
        result = evaluate(policy, a.n, noise, a.seed, a.batch_size, a.replan, a.max_ticks)
        print(json.dumps({k: result[k] for k in ("n", "noise", "successes", "success", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
