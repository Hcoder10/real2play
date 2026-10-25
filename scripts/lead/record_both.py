"""Record the bench from both cameras until stopped: the phone scene camera (phonecam HTTP, every new frame) and the
wrist camera (cv2 device). Writes runs/bench/video/<session>/{scene.mp4, wrist.mp4, scene_ts.txt, wrist_ts.txt} and
reads step marks appended by the lead to marks.txt (one "<unix time> <label>" per line) for cutting clips later.

  python scripts/lead/record_both.py [--wrist-index 1] [--fps 15]
Stop: create runs/bench/video/STOP (or kill the process); files are closed every 60 s so nothing is lost.
"""
import argparse
import json
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VID = ROOT / "runs" / "bench" / "video"


class Writer:
    def __init__(self, folder, name, fps):
        self.folder, self.name, self.fps, self.w, self.part, self.t_open = folder, name, fps, None, 0, 0
        self.ts = open(folder / f"{name}_ts.txt", "a")

    def add(self, img, t):
        if self.w is None or t - self.t_open > 60:
            if self.w is not None:
                self.w.release()
            self.part += 1
            h, w = img.shape[:2]
            self.w = cv2.VideoWriter(str(self.folder / f"{self.name}_{self.part:03d}.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            self.t_open = t
        self.w.write(img)
        self.ts.write(f"{self.part} {t:.3f}\n")
        self.ts.flush()

    def close(self):
        if self.w is not None:
            self.w.release()
        self.ts.close()


def scene_loop(folder, stop, fps):
    w, last_n = Writer(folder, "scene", fps), -1
    while not stop.is_set():
        try:
            info = json.loads(urllib.request.urlopen("http://127.0.0.1:8792/list", timeout=3).read()).get("0", {})
            n = info.get("frames", -1)
            if n != last_n:
                last_n = n
                buf = urllib.request.urlopen("http://127.0.0.1:8792/frame/0.jpg", timeout=3).read()
                img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    w.add(img, time.time() - info.get("age_s", 0))
            else:
                time.sleep(0.05)
        except Exception:  # noqa: BLE001 - stream gap: keep trying
            time.sleep(0.5)
    w.close()


def wrist_loop(folder, stop, index, fps):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    w, period = Writer(folder, "wrist", fps), 1.0 / fps
    while not stop.is_set():
        t0 = time.time()
        ok, img = cap.read()
        if ok:
            w.add(img, time.time())
        time.sleep(max(0.0, period - (time.time() - t0)))
    cap.release()
    w.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrist-index", type=int, default=1)
    ap.add_argument("--fps", type=float, default=15)
    a = ap.parse_args()
    folder = VID / time.strftime("%Y%m%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    (VID / "STOP").unlink(missing_ok=True)
    (VID / "current.txt").write_text(str(folder))
    stop = threading.Event()
    ts = [threading.Thread(target=scene_loop, args=(folder, stop, 2.0), daemon=True),
          threading.Thread(target=wrist_loop, args=(folder, stop, a.wrist_index, a.fps), daemon=True)]
    for t in ts:
        t.start()
    print(f"recording both cameras -> {folder}", flush=True)
    while not (VID / "STOP").exists():
        time.sleep(1)
    stop.set()
    for t in ts:
        t.join(timeout=5)
    print("stopped", flush=True)
