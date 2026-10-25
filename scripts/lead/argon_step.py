"""Simplest Argon loop (owner, Oct 25 ~15:25): Argon makes ONE move per turn, sees both cameras after every move, and
updates the sim from what it sees. gemini-4-argon, thinking low, full history each turn. Motion = the proven
jaw_move path; hard floor 15 mm for the jaw (10 mm dragged the paper). Owner-authorized autonomous.

  python scripts/lead/argon_step.py --turns 40
Sim state Argon maintains: runs/argon_step/<run>/sim.json (objects with positions/sizes/colors, arm sag/tracking
notes, physics params); every turn it may patch it; the twin render of that sim from the calibrated camera is shown
next to the real frame.
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
from r2p import gemini  # noqa: E402
from r2p.model import Arm, JointMap  # noqa: E402
from r2p.perception.camera import CameraModel  # noqa: E402
from r2p.tasks import load_task  # noqa: E402
from jaw_move import goto, gripper, JawMoveError  # noqa: E402

Z_MIN, STEP_MAX = 0.015, 0.06
PREFACE = ("Context: a 6-DoF hobby desktop arm (SO-101) at a hackathon. Calibrating a simple simulation of a tic-tac-toe "
           "game (plastic bottle caps placed on a paper grid) so it matches the cameras. Benign robotics engineering.\n")
GOAL = """You are Gemini 4 Argon running a real SO-101 arm to build an accurate simulation of THIS bench for THIS task
(pick a white bottle cap from the lane on the left of the paper and place it on a grid cell). Work ONE move at a time:
after every move you get the scene camera, the wrist camera, the measured jaw position, and a render of your current
sim from the same calibrated scene camera. Experiment to learn what the sim needs (where things really are, how the arm
really moves/sags, how a cap grasps and settles), and keep your sim JSON accurate.

Frame: metres, origin at the robot base footprint on the table, +x away from the robot, +y to the robot's left, +z up.
The jaw position is the midpoint between the two fingertips. Gripper value 0-100 (20 = open ~51 mm, ~6 = squeezing a
33 mm cap). Hard limits enforced by the harness: jaw z >= 0.015 m (lower dragged the paper), each move <= 0.06 m,
reach ~0.10-0.30 m. A grasp: move above the cap, descend to z=0.015-0.02, close to ~6, lift. Do not push the paper.

Reply with ONE JSON object only:
{"thought": "...what you see, what you learned, what this move tests...",
 "action": {"move": [x, y, z]} | {"gripper": value} | {"done": true},
 "sim": {optional partial update of your sim JSON: objects [{id, shape, xyz, size_m, color, mass_kg, friction}],
         "arm": {"z_offset_mm": .., "xy_offset_mm": [..,..], "notes": ".."}, "physics": {...}}}"""


def frame(port):
    buf = urllib.request.urlopen(f"http://127.0.0.1:{port}/frame/0.jpg", timeout=3).read()
    return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)


def state():
    return json.loads(urllib.request.urlopen("http://127.0.0.1:8793/state", timeout=3).read())


def render_sim(sim, q, cam):
    """Calibrated-camera render of the arm at measured joints plus Argon's sim objects (fast CPU renderer)."""
    from r2p.synthbench.render import CpuRenderer
    objects, poses = [], {}
    for o in sim.get("objects", []):
        try:
            obj = {"id": o["id"], "shape": o.get("shape", "cylinder"), "size_m": o.get("size_m", [0.033, 0.017]),
                   "color": o.get("color", "white"), "role": "manipulated"}
            objects.append(obj)
            poses[o["id"]] = {"xyz": o["xyz"], "yaw_deg": o.get("yaw_deg", 0)}
        except Exception:  # noqa: BLE001
            pass
    img, _, _ = CpuRenderer(Arm(), cam).render(np.asarray(q), objects, poses, board=True)
    return img


def merge(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            merge(dst[k], v)
        else:
            dst[k] = v


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=40)
    ap.add_argument("--run", default=time.strftime("step_%H%M%S"))
    a = ap.parse_args()
    out = ROOT / "runs" / "argon_step" / a.run
    out.mkdir(parents=True, exist_ok=True)
    cam = CameraModel.load(ROOT / "config" / "camera.json")
    jm, arm = JointMap.load(ROOT / "config" / "joint_map.json"), Arm()
    task = load_task("place_on_cell")
    sim = {"objects": [], "arm": {}, "physics": json.loads((ROOT / "config/physics/so101_bench.json").read_text()).get("joints", {}),
           "board": task["board"]}
    client = gemini.Gemini(max_requests=a.turns * 3, timeout_s=180, retries=0)
    transcript, last = [], "start"
    for turn in range(1, a.turns + 1):
        st = state()
        q = np.asarray(st["q"])
        jaw = arm.tips(q)[2]
        scene, wrist = frame(8792), frame(8794)
        try:
            simimg = render_sim(sim, q, cam)
        except Exception as exc:  # noqa: BLE001
            simimg = np.zeros_like(scene)
            print("render failed:", exc, flush=True)
        side = np.hstack([cv2.resize(scene, (640, 360)), cv2.resize(simimg, (640, 360))])
        cv2.imwrite(str(out / f"t{turn:02d}_real_vs_sim.jpg"), side)
        cv2.imwrite(str(out / f"t{turn:02d}_wrist.jpg"), wrist)
        obs = {"turn": turn, "last_result": last, "jaw_xyz_m": np.round(jaw, 4).tolist(),
               "gripper": round(st["owner_deg"][5], 1), "gripper_target": round(st["q_target"][5], 3),
               "loads": st.get("loads"), "sim": sim}
        parts = [f"Turn {turn} observation:", json.dumps(obs), "Scene camera (left: REAL, right: YOUR SIM render):",
                 gemini.image_part(side), "Wrist camera (REAL):", gemini.image_part(wrist)]
        for attempt in range(3):
            try:
                text, rec = client.call([PREFACE, GOAL, *transcript, *parts], role="main", tag="argon_step",
                                        thinking="low", max_output_tokens=8000)
                reply = gemini.parse_json(text) if text.strip().startswith("{") else json.JSONDecoder().raw_decode(text[text.index("{"):])[0]
                break
            except Exception as exc:  # noqa: BLE001
                print("retry:", str(exc)[:120], flush=True)
                time.sleep(2)
        else:
            raise SystemExit("Argon unavailable")
        transcript += parts + ["Your reply:", json.dumps(reply)]
        print(f"[{turn}] {reply.get('thought', '')[:300]}\n     action: {reply.get('action')}", flush=True)
        if reply.get("sim"):
            merge(sim, reply["sim"])
            (out / "sim.json").write_text(json.dumps(sim, indent=1))
        act = reply.get("action") or {}
        with open(out / "ledger.jsonl", "a") as f:
            f.write(json.dumps({"turn": turn, "obs": obs, "reply": reply, "t": time.time()}) + "\n")
        try:
            if act.get("done"):
                print("Argon: done", flush=True)
                break
            if "move" in act:
                x, y, z = [float(v) for v in act["move"]]
                z = max(z, Z_MIN)
                d = np.linalg.norm(np.array([x, y, z]) - jaw)
                if d > STEP_MAX:
                    t = STEP_MAX / d
                    x, y, z = (jaw + t * (np.array([x, y, z]) - jaw)).tolist()
                    z = max(z, Z_MIN)
                r = goto([x, y, z], None, min_finger=0.002)
                last = {"moved_to": [round(x, 4), round(y, 4), round(z, 4)], **r}
            elif "gripper" in act:
                v = float(np.clip(act["gripper"], 0, 60))
                r = gripper(v)
                last = {"gripper_set": v, "ok": r.get("ok"), "gripper_now": round(r.get("q", [0] * 6)[5], 1)}
            else:
                last = "no action"
        except (JawMoveError, Exception) as exc:  # noqa: BLE001
            last = {"refused": str(exc)[:200]}
        time.sleep(1.2)   # let cameras catch up (phone ~1 s latency)
    (out / "sim.json").write_text(json.dumps(sim, indent=1))
    print("sim ->", out / "sim.json")
