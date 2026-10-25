"""Hands-on real2sim (lead, Oct 25 ~15:15): run a real motion with the proven jaw_move/ttt_armd path while recording
30 Hz joint states (q, q_target) + both cameras, then replay the exact commanded targets in the MuJoCo twin and report
real-vs-sim residuals (joint tracking, jaw path, sag). Tune config/physics/*.json from what it shows.

  python scripts/lead/handson.py pick B r1c1      record a real pick-and-place (cap B -> cell r1c1) + replay
  python scripts/lead/handson.py hover            record hovers over lane and board + replay
  python scripts/lead/handson.py replay <dir> [--physics so101_bench]   replay an existing recording
"""
import argparse
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "lead"))
import ttt_armd  # noqa: E402
from jaw_move import goto, gripper  # noqa: E402
from r2p.model import Arm  # noqa: E402
from r2p.tasks import load_task  # noqa: E402

OUT = ROOT / "runs" / "handson"


def get(url, timeout=2):
    return urllib.request.urlopen(url, timeout=timeout).read()


class Recorder:
    def __init__(self, folder):
        self.folder, self.stop, self.rows = folder, threading.Event(), []
        (folder / "scene").mkdir(parents=True, exist_ok=True)
        (folder / "wrist").mkdir(parents=True, exist_ok=True)
        self.t0 = time.monotonic()

    def states(self):
        while not self.stop.is_set():
            b = time.monotonic()
            try:
                st = json.loads(get("http://127.0.0.1:8793/state"))
                self.rows.append({"t": b - self.t0, "q": st["q"], "q_target": st["q_target"], "xyz": st["xyz"]})
            except Exception:  # noqa: BLE001
                pass
            time.sleep(max(0, 1 / 30 - (time.monotonic() - b)))

    def cams(self):
        n = 0
        while not self.stop.is_set():
            for name, port in (("scene", 8792), ("wrist", 8794)):
                try:
                    (self.folder / name / f"{n:04d}_{time.monotonic() - self.t0:07.2f}.jpg").write_bytes(
                        get(f"http://127.0.0.1:{port}/frame/0.jpg"))
                except Exception:  # noqa: BLE001
                    pass
            n += 1
            time.sleep(0.25)

    def __enter__(self):
        self.threads = [threading.Thread(target=self.states, daemon=True), threading.Thread(target=self.cams, daemon=True)]
        for t in self.threads:
            t.start()
        return self

    def __exit__(self, *exc):
        time.sleep(1.0)
        self.stop.set()
        for t in self.threads:
            t.join(timeout=3)
        (self.folder / "states.json").write_text(json.dumps(self.rows))


def replay(folder, physics="so101_bench", scene=None):
    from r2p.sim.twin import TaskTwin
    rows = json.loads((Path(folder) / "states.json").read_text())
    if len(rows) < 10:
        raise SystemExit("too few states")
    scene = scene or {"objects": [], "table": {"friction": 0.6}}
    arm = Arm()
    ts = np.array([r["t"] for r in rows])
    targets = np.array([r["q_target"] for r in rows])
    real_q = np.array([r["q"] for r in rows])
    twin = TaskTwin(scene, physics=physics, q0=real_q[0])
    sim_q = []
    t = ts[0]
    for i in range(len(rows)):
        while t <= ts[i]:
            twin.step(targets[i])
            t += 1 / 30
        sim_q.append(twin.q())
    twin.close()
    sim_q = np.array(sim_q)
    jaw_real = np.array([arm.tips(q)[2] for q in real_q])
    jaw_sim = np.array([arm.tips(q)[2] for q in sim_q])
    jaw_cmd = np.array([arm.tips(q)[2] for q in targets])
    err = np.rad2deg(np.abs(sim_q - real_q))[:, :5]
    out = {"n": len(rows), "duration_s": float(ts[-1] - ts[0]),
           "joint_rms_deg_per_joint": np.sqrt((err ** 2).mean(0)).round(2).tolist(),
           "joint_rms_deg": float(np.sqrt((err ** 2).mean())),
           "jaw_path_rms_mm": float(np.sqrt(((jaw_sim - jaw_real) ** 2).sum(1).mean()) * 1000),
           "real_sag_mm_mean": float((jaw_cmd[:, 2] - jaw_real[:, 2]).mean() * 1000),
           "sim_sag_mm_mean": float((jaw_cmd[:, 2] - jaw_sim[:, 2]).mean() * 1000),
           "real_lag_deg_p90": np.percentile(np.rad2deg(np.abs(targets - real_q))[:, :5], 90, axis=0).round(2).tolist(),
           "sim_lag_deg_p90": np.percentile(np.rad2deg(np.abs(targets - sim_q))[:, :5], 90, axis=0).round(2).tolist()}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("pick", "hover", "replay"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--physics", default="so101_bench")
    a = ap.parse_args()
    if a.what == "replay":
        print(json.dumps(replay(a.args[0], a.physics), indent=1))
        sys.exit()
    cells = load_task("place_on_cell")["board"]["cells"]
    folder = OUT / time.strftime("%H%M%S_") / a.what
    folder.mkdir(parents=True, exist_ok=True)
    with Recorder(folder):
        if a.what == "hover":
            caps = ttt_armd.locate_caps(ttt_armd.frame(), ttt_armd.CameraModel.load(ROOT / "config" / "camera.json"))
            c = cells["r1c1"]
            for p in ([caps[0][0], caps[0][1], 0.05], [c[0], c[1], 0.05], [c[0], c[1], 0.03], [c[0], c[1], 0.05]):
                goto(p, 20, min_finger=0.002)
                time.sleep(1.5)
        else:
            which, cell = (a.args + ["A", "r1c1"])[:2]
            caps = ttt_armd.locate_caps(ttt_armd.frame(), ttt_armd.CameraModel.load(ROOT / "config" / "camera.json"))
            cap = caps[0] if which == "A" or len(caps) < 2 else caps[1]
            print("cap", cap, "->", cell, cells[cell], flush=True)
            (folder / "plan.json").write_text(json.dumps({"cap": cap, "cell": cell, "cell_xy": cells[cell]}))
            print(ttt_armd.place(cap, cells[cell], cell), flush=True)
    print("recorded", folder, flush=True)
    print(json.dumps(replay(folder, a.physics), indent=1))
