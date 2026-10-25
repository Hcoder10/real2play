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
    q = JawIK(arm).solve(xyz, q=q0, grip=g)
    err = np.linalg.norm(arm.tips(q)[2] - xyz)
    qc = JawIK(arm).compensate(q, physics)
    return jm.to_owner(qc), err, q


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz", type=float, nargs="*")
    ap.add_argument("--grip", type=float)
    ap.add_argument("--grip-only", type=float)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--min-finger", type=float, help="allow fingers this close to the table (m), checked by us AND armd")
    a = ap.parse_args()
    st = armd("/state")
    if a.grip_only is not None:
        print(armd("/cmd", {"cmd": "gripper", "value": a.grip_only}))
        sys.exit()
    arm, jm = Arm(), JointMap.load(ROOT / "config" / "joint_map.json")
    physics = json.loads((ROOT / "config" / "physics" / "so101_bench.json").read_text())
    grip = st["q"][5] if a.grip is None else a.grip
    goal = np.array(a.xyz)
    aim = goal.copy()
    for it in range(a.iters):  # re-aim by the encoder-measured miss (servo tracking under load)
        own, err, q = plan(aim, grip, st["q"], arm, jm, physics, st["lo"], st["hi"])
        if err > 0.004:
            sys.exit(f"IK error {err * 1000:.1f} mm: refusing")
        own[5] = grip
        g = arm.geometry(jm.to_model(own))  # our own table clearance with OUR (camera-fitted) map
        if a.min_finger is not None and min(g["finger"], g["gripper"] - 0.01) < a.min_finger:
            sys.exit(f"our clearance check: finger {g['finger'] * 1000:.1f} mm < {a.min_finger * 1000:.1f} mm")
        body = {"cmd": "joints", "owner": [round(float(v), 2) for v in own]}
        if a.min_finger is not None:
            body["min_finger"] = a.min_finger
        r = armd("/cmd", body)
        st = armd("/state")
        mid = arm.tips(jm.to_model(np.array(st["q"])))[2]
        if not r.get("ok") or np.linalg.norm(mid - goal) < 0.002:
            break
        aim = aim + np.clip(goal - mid, -0.015, 0.015)
    print(json.dumps({"ok": r.get("ok"), "refused": r.get("refused") or r.get("abort"), "target": a.xyz,
                      "fk_measured_mid": mid.round(4).tolist(),
                      "miss_mm": round(float(np.linalg.norm(mid - a.xyz) * 1000), 1)}))
