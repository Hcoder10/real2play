"""Live tic-tac-toe on the real bench through the arm daemon: Argon (gemini-4-argon) reads the phone frame and picks
the move, the arm takes the next bottle cap from the supply line and puts it on that cell (our calibrated IK via
scripts/lead/jaw_move.py, armd checks every path), then parks at rest. The human plays coins and presses Enter.

  python scripts/lead/ttt_armd.py                      (owner present; armd running)
  python scripts/lead/ttt_armd.py --place r1c1 --cap 0 one placement, no Argon
Status for the dashboard: out/ttt/status.json.
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "lead"))
sys.path.insert(0, str(ROOT / "scripts" / "argon"))
from r2p import ttt  # noqa: E402
from r2p.perception.camera import CameraModel  # noqa: E402
from r2p.tasks import load_task  # noqa: E402
from jaw_move import goto, gripper, armd  # noqa: E402
from tictactoe import publish  # noqa: E402

# Cap supply line on the robot's right, located by gemini-4-argon after the Oct 25 12:4x reset (fallback only: the
# live loop asks Argon every turn; refresh with --find-caps if the caps move).
SUPPLY = [[0.1748, -0.1245], [0.1274, -0.1272], [0.2205, -0.1248]]
OPEN, SQUEEZE = 20, 6                 # gripper owner values: ~51 mm jaw gap, squeeze a ~33 mm cap
GRASP_Z, PLACE_Z, SAFE_Z = 0.009, 0.012, 0.05
MIN_FINGER = 0.002
VID = ROOT / "runs" / "bench" / "video"


def mark(label):
    try:
        with open(Path((VID / "current.txt").read_text().strip()) / "marks.txt", "a") as f:
            f.write(f"{time.time():.3f} {label}\n")
    except OSError:
        pass


def frame():
    t0 = time.time()
    while True:
        info = json.loads(urllib.request.urlopen("http://127.0.0.1:8792/list", timeout=3).read())["0"]
        if time.time() - info["age_s"] > t0 + 0.4:
            break
        time.sleep(0.2)
    buf = urllib.request.urlopen("http://127.0.0.1:8792/frame/0.jpg", timeout=5).read()
    return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)


CAP_SCHEMA = {"type": "object", "properties": {"caps": {"type": "array", "items": {"type": "object", "properties": {
    "rim_center": {"type": "array", "items": {"type": "integer"}}, "on_grid": {"type": "boolean"}},
    "required": ["rim_center", "on_grid"]}}}, "required": ["caps"]}


def locate_caps(img, cam):
    """Argon finds the white bottle caps; returns supply caps (off the grid, robot's right) as table xy, nearest first."""
    from r2p import gemini
    h, w = img.shape[:2]
    txt, rec = gemini.call([gemini.image_part(img),
        "Find every white plastic bottle cap (open side up) on the table. For each, point to the centre of its TOP rim "
        "ellipse as [y, x] normalized 0-1000, and say whether it sits on the printed 3x3 grid."],
        role="main", tag="ttt:caps", schema=CAP_SCHEMA)
    out = []
    for c in json.loads(txt)["caps"]:
        if c["on_grid"]:
            continue
        uv = [c["rim_center"][1] / 1000 * w, c["rim_center"][0] / 1000 * h]
        xy = cam.deproject(np.array([uv]), h=0.017)[0][:2]
        if 0.09 < xy[0] < 0.23 and -0.15 < xy[1] < -0.06:        # the supply zone on the robot's right
            out.append(xy.round(4).tolist())
    return sorted(out, key=lambda p: (p[0] < 0.11, np.hypot(*p)))   # caps near the base tilt the fingers into the table


def safe_z(xy, grip):
    """Highest travel height <= SAFE_Z our IK reaches cleanly here (near cells hit the wrist limit when high)."""
    from jaw_move import plan
    from r2p.model import Arm, JointMap
    st, arm = armd("/state"), Arm()
    jm = JointMap.load(ROOT / "config" / "joint_map.json")
    physics = json.loads((ROOT / "config" / "physics" / "so101_bench.json").read_text())
    for z in (SAFE_Z, 0.045, 0.04, 0.035, 0.03):
        if plan(np.array([*xy, z]), grip, st["q"], arm, jm, physics, st["lo"], st["hi"])[1] < 0.003:
            return z
    raise RuntimeError(f"no safe travel height over {xy}")


def park():
    """Rest is a straight joint-space move that does not check the path: from a low pose it sweeps the fingers across
    the table and drags the paper (Oct 25 12:3x). Always climb to a high pose over the board first."""
    st = armd("/state")
    mid = None
    try:
        from r2p.model import Arm, JointMap
        mid = Arm().tips(JointMap.load(ROOT / "config" / "joint_map.json").to_model(np.array(st["q"])))[2]
    except Exception:  # noqa: BLE001
        pass
    if mid is not None and mid[2] < SAFE_Z - 0.005:
        goto([*mid[:2], safe_z(mid[:2], st["q"][5])], st["q"][5], min_finger=MIN_FINGER, iters=1)
    goto([0.17, 0.0, SAFE_Z], OPEN, min_finger=MIN_FINGER, iters=1)
    return armd("/cmd", {"cmd": "rest"})


def place(cap_xy, cell_xy, label):
    mark(f"pick {label} from {cap_xy}")
    gripper(OPEN)
    za, zb = safe_z(cap_xy, OPEN), safe_z(cell_xy, SQUEEZE)
    goto([*cap_xy, za], OPEN, min_finger=MIN_FINGER)
    goto([*cap_xy, GRASP_Z], OPEN, min_finger=MIN_FINGER)
    r = gripper(SQUEEZE)
    held = r.get("q", [0] * 6)[5]
    goto([*cap_xy, za], SQUEEZE, min_finger=MIN_FINGER, iters=1)
    mark(f"carry {label}")
    goto([*cell_xy, zb], SQUEEZE, min_finger=MIN_FINGER)
    goto([*cell_xy, PLACE_Z], SQUEEZE, min_finger=MIN_FINGER)
    gripper(OPEN)
    goto([*cell_xy, zb], OPEN, min_finger=MIN_FINGER, iters=1)
    park()
    mark(f"placed {label}")
    return {"gripper_stall": held, "grasped": held > SQUEEZE + 1}   # closing on nothing stalls at ~SQUEEZE


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--place", choices=ttt.CELL_IDS)
    ap.add_argument("--cap", type=int, default=-1, help="index into SUPPLY; -1 = let Argon find the nearest cap")
    ap.add_argument("--find-caps", action="store_true")
    ap.add_argument("--park", action="store_true", help="climb clear of the table, then rest")
    ap.add_argument("--status", default=str(ROOT / "out" / "ttt" / "status.json"))
    a = ap.parse_args()
    cells = load_task("place_on_cell")["board"]["cells"]
    cam = CameraModel.load(ROOT / "config" / "camera.json")
    if a.park:
        print(park())
        sys.exit()
    if a.find_caps:
        print(locate_caps(frame(), cam))
        sys.exit()
    if a.place:
        print(place(SUPPLY[a.cap] if a.cap >= 0 else locate_caps(frame(), cam)[0], cells[a.place], a.place))
        sys.exit()
    board, history = [""] * 9, []
    publish(a.status, board, None, status="waiting_for_human", physical_placement=True)
    while not ttt.terminal(board):
        if input("Place your coin (X), then Enter (q to stop): ").strip() == "q":
            break
        img = frame()
        t0 = time.time()
        decision = ttt.decide(img, cam, {"board": board}, "O", "white plastic bottle caps, open side up",
                              "flat metallic coins (pennies or nickels)")
        decision["latency_s"] = round(time.time() - t0, 1)
        board = decision["board"]
        print(json.dumps({k: decision.get(k) for k in ("board", "move", "why", "model", "latency_s")}))
        publish(a.status, board, decision, status="planning", physical_placement=True)
        if decision["move"] is None:
            break
        caps = locate_caps(frame(), cam)
        print("supply caps (Argon):", caps)
        if not caps:
            sys.exit("no cap in the supply zone: put one at the robot's right and restart")
        publish(a.status, board, decision, status="placing", physical_placement=True)
        result = place(caps[0], cells[decision["move"]], decision["move"])
        board[ttt.cell_index(decision["move"])] = "O"
        publish(a.status, board, decision, status="playing", physical_placement=True, attempt=result)
    publish(a.status, board, None, status="done", result=ttt.winner(board) or "draw", physical_placement=True)
    print("result:", ttt.winner(board) or "draw")
