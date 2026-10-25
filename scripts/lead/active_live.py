"""Real-arm active twin campaign with gemini-4-argon as the cap sensor (white caps on white paper defeat colour
segmentation): Argon points at the cap's rim centre, the calibrated camera deprojects it at the rim height, and a
projected disk is the cap's mask. Before every experiment Argon re-surveys the cap, so pushes are planned from where it
actually is (a 4 mm hand reset after each push is not realistic). Experiment choice, patches and the keep-rule are agent
I's engine (r2p/real2sim/active.py) unchanged.

  python scripts/lead/active_live.py --budget 6 --run bench-active-01      (owner present; repo daemon running)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "lead"))
from r2p import gemini  # noqa: E402
from r2p.perception.camera import CameraModel  # noqa: E402
from r2p.real2sim.active import ArmdMover, run_campaign  # noqa: E402
from r2p.sim.twin import load_physics  # noqa: E402
from ttt_armd import CAP_SCHEMA  # noqa: E402

RIM_H, CAP_D = 0.017, 0.033


class ArgonCapPerceiver:
    """estimate(frame, q) -> {"poses": {piece: {xyz, source: vision, seen}}, "masks": {piece: disk}} for every
    `every`-th frame (an Argon call each); cached for `hold_s` so the mover's reset check sees the fresh survey."""

    def __init__(self, cam, piece_id, near_xy, every=10, hold_s=1.5):
        self.cam, self.piece, self.near = cam, piece_id, np.asarray(near_xy, float)
        self.every, self.hold_s, self.n, self.last, self.calls = every, hold_s, 0, None, 0

    def read(self, frame):
        h, w = frame.shape[:2]
        txt, rec = gemini.call([gemini.image_part(frame),
            "Find every white plastic bottle cap (open side up) on the table. For each, point to the centre of its "
            "TOP rim ellipse as [y, x] normalized 0-1000, and say whether it sits on the printed 3x3 grid."],
            role="main", tag="active:cap_pose", schema=CAP_SCHEMA)
        self.calls += 1
        caps = []
        for c in json.loads(txt)["caps"]:
            uv = [c["rim_center"][1] / 1000 * w, c["rim_center"][0] / 1000 * h]
            caps.append(self.cam.deproject(np.array([uv]), h=RIM_H)[0][:2])
        if not caps:
            return None
        xy = min(caps, key=lambda p: np.linalg.norm(p - self.near))
        if np.linalg.norm(xy - self.near) > 0.08:      # not the experiment's cap (or a misread)
            return None
        self.near = xy
        return np.r_[xy, RIM_H / 2]

    def disk(self, xyz, shape):
        t = np.linspace(0, 2 * np.pi, 48)
        ring = np.c_[xyz[0] + CAP_D / 2 * np.cos(t), xyz[1] + CAP_D / 2 * np.sin(t), np.zeros(48)]
        top = ring + [0, 0, RIM_H]
        pts = self.cam.project(np.r_[ring, top]).astype(np.int32)
        mask = np.zeros(shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(pts), 1)
        return mask.astype(bool)

    def estimate(self, frame, q=None):
        now = time.monotonic()
        if self.last is not None and now - self.last[0] < self.hold_s:
            xyz = self.last[1]
        else:
            self.n += 1
            if (self.n - 1) % self.every:
                return {"poses": {}, "masks": {}}
            xyz = self.read(frame)
            self.last = (now, xyz)
        if xyz is None:
            return {"poses": {}, "masks": {}}
        return {"poses": {self.piece: {"xyz": xyz.tolist(), "yaw_deg": 0.0, "source": "vision", "seen": True}},
                "masks": {self.piece: self.disk(xyz, frame.shape)}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=6)
    ap.add_argument("--run", default="bench-active-01")
    ap.add_argument("--scene", default=str(ROOT / "runs" / "bench" / "active_scene.json"))
    ap.add_argument("--request-budget", type=int, default=30)
    ap.add_argument("--max-nfev", type=int, default=9)
    a = ap.parse_args()
    cam = CameraModel.load(ROOT / "config" / "camera.json")
    scene = json.loads(Path(a.scene).read_text())
    piece = next(o for o in scene["objects"] if not o.get("fixed"))
    perceiver = ArgonCapPerceiver(cam, piece["id"], piece["xyz"][:2])
    mover = ArmdMover(scene, cam, perceiver=perceiver, reset_timeout_s=30)
    execute = mover.execute

    def resurvey_then_execute(experiment, destination, **kw):
        frame, _, _ = mover.capture()
        xyz = perceiver.read(frame)
        if xyz is None:
            raise RuntimeError("Argon cannot see the experiment cap; put it back in the lane")
        perceiver.last = (time.monotonic(), xyz)
        for obj in mover.scene["objects"]:
            if obj["id"] == piece["id"]:
                obj["xyz"] = xyz.tolist()
        print(f"[argon] cap at {np.round(xyz, 4).tolist()} before {experiment.kind}", flush=True)
        return execute(experiment, destination, **kw)

    from measure_grid import measure

    from r2p.tasks import load_task
    board = load_task("place_on_cell")["board"]

    def grid_centre(frame):
        """r1c1 from the camera, or None when the reading is implausible (the arm in view corrupts the outer-square
        detection: a cell size off the measured one means the contour is not the printed grid)."""
        try:
            cells, cell_m, _ = measure(frame, cam)
        except SystemExit:
            return None
        return np.asarray(cells["r1c1"]) if abs(cell_m - board["cell_m"]) < 0.0012 else None

    # Reference = the sheet as measured with the arm parked (task.json), not a frame with the arm in view.
    reference = np.asarray(board["center_xy"])
    print(f"[camera] sheet reference r1c1 at {np.round(reference, 4).tolist()} (parked measurement)", flush=True)

    def guarded_execute(experiment, destination, **kw):
        result = resurvey_then_execute(experiment, destination, **kw)
        readings = []
        for _ in range(3):
            c = grid_centre(mover.capture()[0])
            if c is not None:
                readings.append(np.linalg.norm(c - reference))
            time.sleep(0.3)
        if not readings:
            print(f"[camera] sheet not measurable after {experiment.kind} (arm in view); continuing", flush=True)
            return result
        moved = float(np.median(readings))
        print(f"[camera] sheet moved {moved * 1000:.1f} mm after {experiment.kind}", flush=True)
        if moved > 0.004:
            raise RuntimeError(f"The paper moved {moved * 1000:.1f} mm during {experiment.kind}: stopping the campaign")
        return result

    mover.execute = guarded_execute
    out = ROOT / "runs" / "active_twin" / a.run
    client = gemini.Gemini(max_requests=a.request_budget, retries=1)
    result = run_campaign(mover, budget=a.budget, out_dir=out, physics=load_physics(), client=client,
                          max_nfev=a.max_nfev)
    print(json.dumps({"run": str(out), "before": result["before"]["error"], "after": result["heldout"]["error"],
                      "parameters": result["parameters"], "candidate": result["candidate_physics"],
                      "argon_cap_reads": perceiver.calls, "activated": False}, indent=2, default=str))
