"""Fresh phone frame with the FK jaw tips (red fixed, blue moving, green midpoint) and optional target points
(yellow) drawn: python scripts/lead/snap_overlay.py out.jpg [x y z ...]"""
import json, sys, time, urllib.request
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from r2p.model import Arm, JointMap
from r2p.perception.camera import CameraModel
t0 = time.time()
while True:
    info = json.loads(urllib.request.urlopen("http://127.0.0.1:8792/list", timeout=3).read())["0"]
    if time.time() - info["age_s"] > t0 + 0.5: break
    time.sleep(0.2)
img = cv2.imdecode(np.frombuffer(urllib.request.urlopen("http://127.0.0.1:8792/frame/0.jpg").read(), np.uint8), 1)
st = json.loads(urllib.request.urlopen("http://127.0.0.1:8793/state").read())
cam = CameraModel.load(ROOT / "config" / "camera.json")
a, b, m = Arm().tips(JointMap.load(ROOT / "config" / "joint_map.json").to_model(np.array(st["q"])))
pts = [(a, (0, 0, 255)), (b, (255, 0, 0)), (m, (0, 255, 0))]
xs = [float(v) for v in sys.argv[2:]]
pts += [(np.array(xs[i:i + 3]), (0, 255, 255)) for i in range(0, len(xs), 3)]
for p, c in pts:
    u = cam.project(p[None])[0]; cv2.circle(img, (int(u[0]), int(u[1])), 5, c, -1)
cv2.imwrite(sys.argv[1], img[120:540, 60:600])
