"""Measure the printed tic-tac-toe grid with the calibrated scene camera and write its cells into
tasks/place_on_cell/task.json (cell centres, cell size, board centre, randomize choices). Saves a check image.

  python scripts/lead/measure_grid.py [--write]
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from r2p.perception.camera import CameraModel  # noqa: E402

PPM, X0, X1, Y0, Y1 = 2000, 0.06, 0.30, -0.15, 0.15          # top-down window (spec frame), 0.5 mm / px


def topdown(img, cam):
    W, H = int((Y1 - Y0) * PPM), int((X1 - X0) * PPM)
    src = cam.project(np.array([[X0, Y0, 0], [X0, Y1, 0], [X1, Y1, 0], [X1, Y0, 0]], float)).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, np.float32([[0, 0], [W, 0], [W, H], [0, H]]))
    return cv2.warpPerspective(img, M, (W, H), borderValue=(255, 255, 255))


def measure(img, cam):
    """Outer square of the printed grid (largest square contour of long dark strokes) -> its 4 corners in metres;
    cells = the square divided into thirds (bilinear, so a slightly rotated sheet is fine)."""
    top = topdown(img, cam)
    g = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
    solid = cv2.morphologyEx((g < 110).astype(np.uint8), cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    solid = cv2.dilate(solid, np.ones((15, 15), np.uint8))                 # cubes / weights, with margin
    dark = cv2.adaptiveThreshold(g, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 30) & (1 - solid)
    lines_img = cv2.dilate(dark, np.ones((7, 7), np.uint8))          # join the thin, slightly tilted strokes
    cs, _ = cv2.findContours(lines_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cs:
        (cxp, cyp), (w, h), ang = cv2.minAreaRect(c)
        if min(w, h) < 120 or max(w, h) / min(w, h) > 1.25:
            continue
        if best is None or w * h > best[1][0] * best[1][1]:
            best = ((cxp, cyp), (w, h), ang)
    if best is None:
        raise SystemExit("grid square not found")
    box = cv2.boxPoints(best)                                   # 4 corners, top-down px (u = y axis, v = x axis)
    pts = np.c_[X0 + box[:, 1] / PPM, Y0 + box[:, 0] / PPM]     # (x, y) metres
    o = np.argsort(pts[:, 0])                                   # near-robot pair (small x) first
    near, far = pts[o[:2]], pts[o[2:]]
    near, far = near[np.argsort(near[:, 1])], far[np.argsort(far[:, 1])]
    p00, p01, p10, p11 = near[0], near[1], far[0], far[1]       # (row, col) corners: row along x, col along y
    cells = {}
    for i in range(3):
        for j in range(3):
            a, b = (i + 0.5) / 3, (j + 0.5) / 3
            c = (1 - a) * ((1 - b) * p00 + b * p01) + a * ((1 - b) * p10 + b * p11)
            cells[f"r{i}c{j}"] = [round(float(c[0]), 4), round(float(c[1]), 4)]
    side = np.mean([np.linalg.norm(p01 - p00), np.linalg.norm(p11 - p10), np.linalg.norm(p10 - p00), np.linalg.norm(p11 - p01)])
    cv2.drawContours(top, [box.astype(int)], 0, (0, 200, 0), 2)
    return cells, side / 3, top


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    cam = CameraModel.load(ROOT / "config" / "camera.json")
    buf = urllib.request.urlopen("http://127.0.0.1:8792/frame/0.jpg", timeout=5).read()
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    cells, cell_m, top = measure(img, cam)
    print("cell size %.1f mm, centre %s" % (cell_m * 1000, cells["r1c1"]))
    ok = 0.025 < cell_m < 0.05
    for k, (x, y) in cells.items():
        u, v = int((y - Y0) * PPM), int((x - X0) * PPM)
        cv2.circle(top, (u, v), 6, (0, 0, 255), -1)
        cv2.putText(top, k, (u + 6, v - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.imwrite(str(ROOT / "runs" / "bench" / "grid_measured.jpg"), top)
    if a.write and ok:
        p = ROOT / "tasks" / "place_on_cell" / "task.json"
        t = json.loads(p.read_text())
        old = {tuple(v): k for k, v in t["board"]["cells"].items()}
        cell_m = round(float(cell_m), 4)
        old_c = t["board"]["center_xy"]
        t["board"].update(center_xy=cells["r1c1"], cell_m=cell_m, cells=cells)
        for v in t["randomize"].values():   # a supply lane laid out beside the sheet moves with it
            if isinstance(v, dict) and v.get("follow_board") and "x" in v:
                v["x"] = [round(a + cells["r1c1"][0] - old_c[0], 4) for a in v["x"]]
                v["y"] = [round(a + cells["r1c1"][1] - old_c[1], 4) for a in v["y"]]
        for o in t["objects"]:
            if o["id"] in cells:
                o["xy"] = cells[o["id"]]
            if o.get("shape") == "plate":
                o["size_m"][:2] = [cell_m, cell_m]
        for v in t["randomize"].values():
            if isinstance(v, dict) and "xy_choice" in v:
                v["xy_choice"] = [cells[old[tuple(xy)]] for xy in v["xy_choice"]]
        p.write_text(json.dumps(t, indent=1))
        print("wrote", p, "cell", cell_m, "centre", cells["r1c1"])
