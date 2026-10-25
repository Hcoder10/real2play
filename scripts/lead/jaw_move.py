"""Move the real arm's jaw midpoint to a spec-frame xyz with OUR model (JawIK at the actual gripper opening, gravity
sag compensated), sent to armd as owner-degree joint targets (armd still checks limits and table clearance).

  python scripts/lead/jaw_move.py X Y Z [--grip 35]       grip = owner gripper value 0-100 held during the move
  python scripts/lead/jaw_move.py --grip-only 5           just set the gripper
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from r2p.frames import HOME_Q  # noqa: E402
from r2p.model import Arm, JointMap  # noqa: E402
from r2p.sim.ik import JawIK  # noqa: E402

ARMD = "http://127.0.0.1:8793"
ROLL_OWNER = 81.9


def armd(path, body=None):
    req = urllib.request.Request(ARMD + path, json.dumps(body).encode() if body else None,
                                 {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


def plan(xyz, grip, q_owner, arm, jm, physics, lo=None, hi=None):
    if lo is not None:  # armd's calibrated owner limits, as model-space bounds (position wins over fingers-down)
        a, b = jm.to_model(np.asarray(lo, float)), jm.to_model(np.asarray(hi, float))
        arm.range[:5] = np.sort(np.c_[a, b], axis=1)[:5] + [[.01, -.01]]
    q0 = jm.to_model(np.asarray(q_owner, float))
    # Wrist roll stays where the visual calibration observed it (owner ~82 deg); other rolls are unverified.
    roll = jm.to_model(np.r_[0, 0, 0, 0, ROLL_OWNER, 0])[4]
    arm.range[4] = [roll - 1e-3, roll + 1e-3]
    q0[4] = roll
    g = jm.to_model(np.r_[q_owner[:5], grip])[5]
    best = None
    for seed in (q0, np.r_[HOME_Q[:4], roll, g]):  # the folded rest pose is a poor seed: also start from home
        q = JawIK(arm).solve(xyz, q=seed, grip=g)
        err = np.linalg.norm(arm.tips(q)[2] - xyz)
        if best is None or err < best[1]:
            best = (q, err)
    q, err = best
    qc = JawIK(arm).compensate(q, physics)
    return jm.to_owner(qc), err, q


class JawMoveError(RuntimeError):
    pass


def goto(xyz, grip=None, *, min_finger=None, iters=3):
    """Jaw midpoint to spec xyz (our IK, roll locked, sag compensated, re-aimed by the encoder-measured miss)."""
    st = armd("/state")
    arm, jm = Arm(), JointMap.load(ROOT / "config" / "joint_map.json")
    physics = json.loads((ROOT / "config" / "physics" / "so101_bench.json").read_text())
    grip = st["q"][5] if grip is None else grip
    goal = np.asarray(xyz, float)
    aim = goal.copy()
    for _ in range(iters):
        own, err, q = plan(aim, grip, st["q"], arm, jm, physics, st["lo"], st["hi"])
        if err > 0.004:
            raise JawMoveError(f"IK error {err * 1000:.1f} mm at {goal.round(3).tolist()}")
        own[5] = grip
        g = arm.geometry(jm.to_model(own))  # our own table clearance with OUR (camera-fitted) map
        if min_finger is not None and min(g["finger"], g["gripper"] - 0.01) < min_finger:
            raise JawMoveError(f"our clearance check: finger {g['finger'] * 1000:.1f} mm < {min_finger * 1000:.1f} mm")
        body = {"cmd": "joints", "owner": [round(float(v), 2) for v in own]}
        if min_finger is not None:
            body["min_finger"] = min_finger
        r = armd("/cmd", body)
        st = armd("/state")
        mid = arm.tips(jm.to_model(np.array(st["q"])))[2]
        if not r.get("ok"):
            raise JawMoveError(f"armd: {r.get('refused') or r.get('abort') or r}")
        if np.linalg.norm(mid - goal) < 0.002:
            break
        aim = aim + np.clip(goal - mid, -0.015, 0.015)
    return {"target": goal.tolist(), "fk_measured_mid": mid.round(4).tolist(),
            "miss_mm": round(float(np.linalg.norm(mid - goal) * 1000), 1)}


def gripper(value):
    return armd("/cmd", {"cmd": "gripper", "value": value})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz", type=float, nargs="*")
    ap.add_argument("--grip", type=float)
    ap.add_argument("--grip-only", type=float)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--min-finger", type=float, help="allow fingers this close to the table (m), checked by us AND armd")
    a = ap.parse_args()
    if a.grip_only is not None:
        print(gripper(a.grip_only))
        sys.exit()
    try:
        print(json.dumps(goto(a.xyz, a.grip, min_finger=a.min_finger, iters=a.iters)))
    except JawMoveError as e:
        sys.exit(str(e))
