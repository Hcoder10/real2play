"""Opt-in real-bench training convention. Never rewrite an existing export."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

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


def contact_profile(value):
    """Declare the expert's contact control without modifying the captured file.

    A force-limited hold on a wide cap can be above the old half-stroke opening
    threshold. Require nearly the full opening before reporting release, and
    carry this convention through trace metadata into policy exports.
    """
    profile = load_profile(value)
    closed, opened = profile["gripper_close_rad"], profile["gripper_open_rad"]
    profile["gripper_open_threshold_rad"] = max(profile["gripper_open_threshold_rad"], closed+.9*(opened-closed))
    profile["expert_control"] = {"gripper_hold_torque_nm": .1, "grasp_tip_height_m": .002}
    return profile


def _evaluate_episode(args):
    from .expert import Expert

    task, profile, seed, dart = args
    result = Expert(task, deployment=profile, dart=dart).run(seed=seed)
    targets = np.asarray([row["q_target"] for row in result["ticks"]])
    executed = np.asarray([row["q_exec"] for row in result["ticks"]])
    bounds = np.asarray(profile["model_limits"])
    errors = []
    for goal in task["goal"]:
        if goal["pred"] == "at":
            xy = goal["xy"]
            xy = result["final"][xy]["xyz"][:2] if isinstance(xy, str) else xy
            errors.append(float(np.linalg.norm(np.asarray(result["final"][goal["a"]]["xyz"][:2])-xy)*1000))
    return {"seed": seed, "success": result["meta"]["success"],
            "fail_reason": result["meta"]["fail_reason"], "placement_mm": errors,
            "ticks": len(targets),
            "max_target_step_rad": float(np.abs(np.diff(np.vstack([profile["start_q"], targets]), axis=0)).max()),
            "max_executed_step_rad": float(np.abs(np.diff(np.vstack([profile["start_q"], executed]), axis=0)).max()),
            "max_limit_violation_rad": float(max(0, (bounds[:, 0]-targets).max(), (targets-bounds[:, 1]).max(),
                                                  (bounds[:, 0]-executed).max(), (executed-bounds[:, 1]).max())),
            "max_roll_error_rad": float(max(np.abs(targets[:, 4]-profile["roll_rad"]).max(),
                                              np.abs(executed[:, 4]-profile["roll_rad"]).max()))}


def evaluate_profile(task, profile, *, n=64, seed=0, workers=2, dart=0.0):
    """Repeatable physical-contact acceptance, with bounded CPU parallelism."""
    from r2p.tasks import load_task
    from .twin import load_physics

    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(workers, int) or not 1 <= workers <= 8:
        raise ValueError("workers must be an integer between 1 and 8")
    task = deepcopy(task) if isinstance(task, dict) else load_task(task)
    profile = contact_profile(profile)
    start = time.perf_counter()
    jobs = [(task, profile, seed+i, dart) for i in range(n)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        episodes = list(pool.map(_evaluate_episode, jobs))
    successes = sum(row["success"] for row in episodes)
    errors = [error for row in episodes for error in row["placement_mm"]]
    physics = load_physics()
    digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return {"task": task["name"], "n": n, "seed": seed, "workers": workers, "dart_std_rad": dart,
            "successes": successes, "success": successes/n,
            "failures": dict(Counter(row["fail_reason"] for row in episodes if not row["success"])),
            "median_placement_mm": float(np.median(errors)) if errors else None,
            "seconds": time.perf_counter()-start, "episodes": episodes,
            "task_spec": task, "deployment": profile, "physics_config": physics,
            "task_sha256": digest(task), "profile_sha256": digest(profile), "physics_sha256": digest(physics),
            "code_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                            for name in ("bench.py", "expert.py", "ik.py", "twin.py")}}


def main():
    parser = argparse.ArgumentParser(description="Evaluate the physical expert with calibrated bench constraints")
    parser.add_argument("--task", default="place_on_cell")
    parser.add_argument("--bench-profile", required=True)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dart", type=float, default=0)
    parser.add_argument("--require-success", type=float, default=.7)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_profile(args.task, args.bench_profile, n=args.n, seed=args.seed,
                              workers=args.workers, dart=args.dart)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(f"{report['successes']}/{report['n']} successes ({report['success']:.1%}); "
          f"failures={report['failures']}; report={args.out}")
    return 0 if report["success"] >= args.require_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
