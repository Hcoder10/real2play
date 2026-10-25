"""Argon calibrates the arm by sight: the arm visits safe poses (armd, torque on, fingers >= 10 cm up), the calibrated
scene camera photographs each, Gemini (MAIN role) points at the jaw tips, and the joint-map offsets are fitted so the
FK jaw tips reproject onto Argon's points. Held-out poses validate the fit.

  python scripts/lead/visual_calib.py collect   (moves the arm; owner present)
  python scripts/lead/visual_calib.py point     (Gemini pointing on the saved frames)
  python scripts/lead/visual_calib.py fit       (offsets + held-out error)
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "runs" / "bench" / "vcal"
ARMD, CAM = "http://127.0.0.1:8793", "http://127.0.0.1:8792"
OLD_MAP = Path(r"C:\Users\sarta\argon-hack\deploy\config\joint_map.json")


def post(d, timeout=120):
    r = urllib.request.Request(ARMD + "/cmd", json.dumps(d).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=timeout).read())


def fresh_frame(after_t):
    """A camera frame captured after time after_t (the phone stream lags ~1-2 s)."""
    for _ in range(160):
        info = json.loads(urllib.request.urlopen(CAM + "/list", timeout=3).read()).get("0", {})
        if info and time.time() - info["age_s"] > after_t:
            buf = urllib.request.urlopen(CAM + "/frame/0.jpg", timeout=3).read()
            return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        time.sleep(0.25)
    raise RuntimeError("no fresh camera frame")


def collect():
    OUT.mkdir(parents=True, exist_ok=True)
    poses = []
    for z in (0.11, 0.17):
        for x in (0.15, 0.21, 0.27):
            for y in (-0.08, 0.0, 0.08):
                poses.append([x, y, z])
    rows = []
    for k, p in enumerate(poses):
        r = post({"cmd": "xyz", "xyz": p})
        if not r.get("ok"):
            print(k, p, "skipped:", r.get("refused") or r.get("abort"), flush=True)
            continue
        time.sleep(1.0)
        t_settled = time.time()
        img = fresh_frame(t_settled + 0.3)
        st = json.loads(urllib.request.urlopen(ARMD + "/state", timeout=5).read())
        cv2.imwrite(str(OUT / f"p{k:02d}.jpg"), img)
        rows.append({"k": k, "cmd_xyz": p, "owner_deg": st["q"]})
        print(k, p, "q", st["q"], flush=True)
    post({"cmd": "rest"})
    (OUT / "poses.json").write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} poses saved; arm at rest (torque still on)")


def point():
    from r2p import gemini
    rows = json.loads((OUT / "poses.json").read_text())
    schema = {"type": "object", "properties": {
        "fixed_tip": {"type": "array", "items": {"type": "integer"}},
        "moving_tip": {"type": "array", "items": {"type": "integer"}},
        "visible": {"type": "boolean"}}, "required": ["fixed_tip", "moving_tip", "visible"]}
    prompt = ("This photo shows a small robot arm (SO-101) with a two-finger gripper pointing down over a table. "
              "Point to the very TIP (lowest end) of each of the two gripper fingers. The FIXED finger is part of the "
              "gripper body; the MOVING finger is the one hinged to it. Return points as [y, x] normalized to 0-1000. "
              "visible=false if the fingertips are hidden or out of frame.")
    for r in rows:
        img = cv2.imread(str(OUT / f"p{r['k']:02d}.jpg"))
        txt, rec = gemini.call([gemini.image_part(img), prompt], role="main", tag="vcal_tips", schema=schema)
        a = json.loads(txt)
        h, w = img.shape[:2]
        r["tips_px"] = {k: [a[k][1] / 1000 * w, a[k][0] / 1000 * h] for k in ("fixed_tip", "moving_tip")}
        r["visible"], r["model"] = a["visible"], rec.get("served_by")
        print(r["k"], r["visible"], {k: np.round(v).tolist() for k, v in r["tips_px"].items()}, r["model"], flush=True)
    (OUT / "poses.json").write_text(json.dumps(rows, indent=1))


def fit():
    from scipy.optimize import least_squares
    from r2p.model import Arm, JointMap
    from r2p.perception.camera import CameraModel
    arm, cam = Arm(), CameraModel.load(ROOT / "config" / "camera.json")
    rows = [r for r in json.loads((OUT / "poses.json").read_text()) if r.get("visible")]
    base = json.loads(OLD_MAP.read_text())["candidates"]["via_corpus_audit"]
    j = [base["joints"][n] for n in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")]
    off0 = np.array([x.get("offset_deg", 0) for x in j], float)
    gc, go = base["gripper"]["closed_rad"], base["gripper"]["open_rad"]
    test = rows[::4]
    train = [r for r in rows if r not in test]

    def residual(off, rs, mid_only=False):
        jm = JointMap(offset=np.asarray(off), gc=gc, go=go)
        e = []
        for r in rs:
            a, b, m = arm.tips(jm.to_model(np.array(r["owner_deg"])))
            uv = cam.project(np.array([a, b, m]))
            pa, pb = np.array(r["tips_px"]["fixed_tip"]), np.array(r["tips_px"]["moving_tip"])
            if mid_only:
                e += list(uv[2] - (pa + pb) / 2)
            else:  # finger identity can be ambiguous to the model: take the better assignment
                e1 = np.r_[uv[0] - pa, uv[1] - pb]; e2 = np.r_[uv[0] - pb, uv[1] - pa]
                e += list(e1 if e1 @ e1 <= e2 @ e2 else e2)
        return np.array(e)

    def rms(off, rs):
        e = residual(off, rs, mid_only=True).reshape(-1, 2)
        return float(np.sqrt((e ** 2).sum(1).mean()))

    best = None
    for droll in (0, 90, -90, 180):
        x0 = off0 + np.r_[0, 0, 0, 0, droll]
        s = least_squares(lambda o: np.r_[residual(o, train), 0.05 * (o - x0)[:4]], x0, loss="huber", f_scale=15)
        if best is None or s.cost < best.cost:
            best = s
    off = best.x
    print("start offsets", off0.round(1), "-> fitted", off.round(1))
    print(f"train jaw-mid reprojection: {rms(off0, train):.1f} px -> {rms(off, train):.1f} px (n={len(train)})")
    print(f"HELD-OUT  jaw-mid reprojection: {rms(off0, test):.1f} px -> {rms(off, test):.1f} px (n={len(test)})")
    jm = JointMap(offset=off, gc=gc, go=go)
    for r in test:
        _, _, m0 = arm.tips(JointMap(offset=off0, gc=gc, go=go).to_model(np.array(r["owner_deg"])))
        _, _, m1 = arm.tips(jm.to_model(np.array(r["owner_deg"])))
        print("  pose", r["k"], "cmd", r["cmd_xyz"], "old FK", m0.round(3), "fitted FK", m1.round(3))
    (OUT / "fit.json").write_text(json.dumps({"offset_deg": off.tolist(), "start": off0.tolist(),
                                              "train_px": rms(off, train), "heldout_px": rms(off, test),
                                              "heldout_px_before": rms(off0, test)}, indent=1))


if __name__ == "__main__":
    {"collect": collect, "point": point, "fit": fit}[sys.argv[1]]()
