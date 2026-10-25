"""Read-only VLA export acceptance and calibrated real-bench start audit."""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path
import numpy as np
from r2p.frames import JOINTS
from r2p.model import Arm
from r2p.robot.config import load_config, load_joint_map, sag_gain
from r2p.robot.gravity import GravityComp
from r2p.robot.rehearsal import validate_profile
from r2p.sim.bench import load_profile


def check(export, profile_path, config_path=None):
    directory = Path(export)
    cfg = json.loads((directory/"config.json").read_text())
    bench = load_config(config_path)
    joint_map = load_joint_map(bench)
    profile = load_profile(profile_path)
    limits = validate_profile(profile, joint_map)
    arm = Arm()
    start = np.asarray(cfg["start_q"])
    servo_start = GravityComp(arm, sag_gain(bench))(start)
    bounds = limits.model_bounds(joint_map, arm.range, bench["guard"]["margin_deg"])
    bad = (servo_start < bounds[:, 0]) | (servo_start > bounds[:, 1])
    evals = cfg.get("eval", {})
    passed = all(evals.get(mode, {}).get("n", 0) >= 30 and evals[mode].get("success", 0) >= .8 for mode in ("clean", "real"))
    return {"motion": False, "task": cfg["task"], "action": cfg.get("action"),
            "start_q": start.tolist(), "compensated_start_owner": joint_map.to_owner(servo_start).tolist(),
            "start_valid": not bool(bad.any()), "start_limit_joints": [j for j, b in zip(JOINTS, bad) if b],
            "twin_gate_passed": passed, "deployment_qualified": bool(passed and not bad.any() and cfg.get("deployment_qualified")),
            "eval": {mode: {k: evals.get(mode, {}).get(k) for k in ("n", "successes", "success")} for mode in ("clean", "real")}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export", required=True)
    p.add_argument("--bench-profile", default="scripts/robot/K_BENCH_PROFILE.json")
    p.add_argument("--config")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    result = check(a.export, a.bench_profile, a.config)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["deployment_qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
