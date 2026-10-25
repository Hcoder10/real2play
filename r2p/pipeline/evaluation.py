"""Paired physical-twin evaluation with a saved, reproducible scene/noise protocol.

No training process or accelerator is used here. Both policies start from their
declared start_q and receive commanded q, with the same reset rule as the bench.
"""

import copy
from collections import deque

from .storage import digest


def make_plan(task, focus, n, seed):
    import numpy as np
    from r2p.tasks import sample_scene

    rng = np.random.default_rng(seed)
    scenes = [copy.deepcopy(focus)] + [sample_scene(task, rng) for _ in range(n - 1)]
    return {"schema": "r2p_eval_plan/1", "seed": seed, "task": task["name"],
            "noise": ["clean", "real"], "protocol": "commanded-q-ensemble30hz-v1",
            "real_noise": {"pose_std_m": 0.0015, "pose_bias_std_m": 0.002, "joint_offset_std_deg": 1.5,
                           "stale_ticks": 3, "stale_probability": 0.3, "latency_ticks": 1},
            "episodes": [{"id": f"scene_{i:04d}", "scene": scene, "noise_seed": int(rng.integers(2**31))}
                         for i, scene in enumerate(scenes)]}


def rollout(policy, task, scene, noise, seed, noise_spec):
    import numpy as np
    from r2p.model import Arm
    from r2p.sim import TaskTwin
    from r2p.tasks import check_window, encode_slots
    from r2p.policy.eval import CommandTracker

    config = policy.config
    if config.get("task") != task["name"] or config.get("object_ids") != [o["id"] for o in task["objects"]]:
        raise ValueError("Paired policies must declare this task and identical object-slot ordering")
    start_q = np.asarray(config["start_q"], dtype=float)
    twin = TaskTwin(copy.deepcopy(scene), q0=start_q, physics=noise_spec.get("physics", "so101_bench"))
    arm = Arm()
    commanded = start_q.copy()
    total = max(15, int(task.get("limits", {}).get("episode_s", 30) * 30))
    rng = np.random.default_rng(seed)
    initial = twin.poses()
    tracker = CommandTracker(task, initial)
    offsets = np.r_[rng.normal(0, np.deg2rad(noise_spec["joint_offset_std_deg"]), 5), 0]
    bias = rng.normal(0, noise_spec["pose_bias_std_m"], (config["K"], 3))
    disturbances = rng.normal(0, noise_spec["pose_std_m"], (total, config["K"], 3))
    stale = rng.random(total) < noise_spec["stale_probability"]
    states, chunks = deque(maxlen=15), deque(maxlen=15)
    history = deque(maxlen=noise_spec["stale_ticks"] + 1)
    pending = deque([start_q.copy() for _ in range(noise_spec["latency_ticks"])]) if noise != "clean" else deque()
    if noise == "clean":
        offsets *= 0
    try:
        for tick in range(total):
            measured = np.asarray(twin.q()) + offsets
            if np.any(np.abs(commanded - measured) > np.deg2rad(4)):
                commanded = measured.copy()
            slots = encode_slots(task, twin.poses())
            history.append(slots.copy())
            if noise != "clean":
                if stale[tick]:
                    slots = history[0].copy()
                slots[:, :3] += (bias + disturbances[tick]) * slots[:, 8:9]
            chunk = np.asarray(policy.act({"q": commanded.copy(), "slots": slots, "command": tracker.command}))
            if chunk.shape != (15, 6) or not np.isfinite(chunk).all():
                raise ValueError("Policy must emit finite absolute joint targets with shape (15,6)")
            chunks.append((tick, chunk))
            predictions = np.stack([prediction[tick-issued] for issued, prediction in chunks])
            target = np.average(predictions, axis=0, weights=np.exp(-.1*np.arange(len(predictions))))
            target = np.clip(commanded + np.clip(target-commanded, -.10, .10), *arm.range.T)
            pending.append(target.copy())
            commanded = pending.popleft()
            twin.step(commanded - offsets)
            current = twin.poses()
            states.append({"objects": current, "gripper_open": not twin.grip_closed(),
                           "site_xyz": arm.tips(twin.q())[2].tolist(), "initial": initial})
            previous = tracker.command
            tracker.update(states[-1], twin.q(), start_q)
            if previous != tracker.command:
                chunks.clear()
            if check_window(task, list(states), hz=30)["ok"]:
                return True
        return False
    finally:
        if callable(getattr(twin, "close", None)):
            twin.close()


def evaluate_pair(challenger, incumbent, task, plan, *, evaluator=None, loader=None):
    if loader is None:
        from r2p.policy import load
        loader = load
    evaluate = evaluator or rollout
    reports = []
    for export in (challenger, incumbent):
        results = {}
        for noise in plan["noise"]:
            rows = []
            for episode in plan["episodes"]:
                # Reload for each episode so policy state cannot carry between trials.
                policy = loader(export)
                noise_spec = copy.deepcopy(plan["real_noise"])
                if "physics" in plan:
                    noise_spec["physics"] = copy.deepcopy(plan["physics"])
                success = evaluate(policy, copy.deepcopy(task), copy.deepcopy(episode["scene"]), noise,
                                   episode["noise_seed"], noise_spec)
                rows.append({"scene_id": episode["id"], "noise_seed": episode["noise_seed"], "success": bool(success)})
            results[noise] = rows
        reports.append({"schema": "r2p_paired_eval/1", "policy": str(export), "plan_hash": digest(plan),
                        "evidence": "simulation", "results": results})
    return tuple(reports)
