"""Touch fit logger: read the limp follower's joints at 20 Hz (READ-ONLY, never writes a register) for N seconds while
the owner holds the FIXED jaw's tip on known points; segment the stationary holds and save them.

  python scripts/lead/touch_log.py --seconds 90 --out runs/bench/touch.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, r"C:\Users\sarta\argon-hack\deploy")
from so101d.bench import load_cfg  # noqa: E402
from so101d.robot import ReadOnlyBus  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=90)
ap.add_argument("--out", default="runs/bench/touch.json")
a = ap.parse_args()
c = load_cfg()["robot"]
bus = ReadOnlyBus(c["port"], c["id"], c["calibration_dir"])
assert not bus.torque_on, "torque is on: release the arm first"
rows, t0 = [], time.time()
print("logging; hold the fixed-jaw tip on each point ~3 s", flush=True)
while time.time() - t0 < a.seconds:
    rows.append([time.time() - t0, *bus.read().tolist()])
    time.sleep(0.05)
bus.close()
r = np.array(rows)
# stationary holds: windows of >= 1.5 s where all arm joints stay within 0.6 deg
holds, i = [], 0
while i < len(r):
    j = i
    while j + 1 < len(r) and np.all(np.ptp(r[i:j + 2, 1:6], axis=0) < 0.6):
        j += 1
    if r[j, 0] - r[i, 0] >= 1.5:
        holds.append({"t0": round(r[i, 0], 2), "t1": round(r[j, 0], 2), "owner_deg": r[i:j + 1, 1:].mean(0).round(3).tolist()})
        i = j + 1
    else:
        i += 1
Path(a.out).write_text(json.dumps({"holds": holds, "n": len(r)}, indent=1))
print(f"{len(holds)} holds -> {a.out}")
for h in holds:
    print(h["t0"], h["t1"], h["owner_deg"])
