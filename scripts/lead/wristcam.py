"""Serve the gripper (wrist) camera over HTTP with the phone server's API, so one process owns the USB device and
everyone else (recorder, Argon, experimenter, dashboard) reads it: GET /frame/0.jpg, GET /list -> {"0": {age_s, frames}}.

  python scripts/lead/wristcam.py [--index 1] [--port 8794]
"""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

latest = {"jpg": None, "t": 0.0, "n": 0}
lock = threading.Lock()


def grab(index, width, height):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    while True:
        ok, img = cap.read()
        if not ok:
            time.sleep(0.2)
            continue
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        with lock:
            latest.update(jpg=buf.tobytes(), t=time.time(), n=latest["n"] + 1)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        with lock:
            jpg, t, n = latest["jpg"], latest["t"], latest["n"]
        if self.path.startswith("/frame/") and jpg:
            body, kind = jpg, "image/jpeg"
        elif self.path.startswith("/list"):
            body = json.dumps({"0": {"source": "wrist", "age_s": round(time.time() - t, 3), "frames": n}}).encode()
            kind = "application/json"
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=1)
    ap.add_argument("--port", type=int, default=8794)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    a = ap.parse_args()
    threading.Thread(target=grab, args=(a.index, a.width, a.height), daemon=True).start()
    print(f"wrist camera {a.index} on http://127.0.0.1:{a.port}/frame/0.jpg", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
