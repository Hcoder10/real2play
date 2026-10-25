"""Loopback inference adapter: Argon command + RGB cameras -> absolute 30 Hz chunk.

Run behind SSH port forwarding. This service never opens a robot bus or sends motion.
"""
from __future__ import annotations
import argparse
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import time
import numpy as np


def decode_image(value):
    import cv2
    data = base64.b64decode(value, validate=True)
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (192, 256):
        raise ValueError("expected a 256x192 RGB camera encoded as PNG/JPEG")
    return image[..., ::-1].copy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export", required=True)
    p.add_argument("--port", type=int, default=9106)
    p.add_argument("--device", default="cpu")
    p.add_argument("--memory-fraction", type=float)
    p.add_argument("--max-seconds", type=float, default=1800, help="Auto-stop before the assigned lease expires")
    p.add_argument("--gpu-duty", type=float, default=.60)
    a = p.parse_args()
    if not np.isfinite(a.max_seconds) or a.max_seconds <= 0 or not 0 < a.gpu_duty <= 1:
        p.error("positive finite max-seconds and gpu-duty in (0,1] required")
    from .runtime import Policy
    policy = Policy.load(a.export, a.device, a.memory_fraction)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, value):
            payload = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/config":
                self.reply(200, policy.config)
                return
            if self.path != "/health":
                self.reply(404, {"error": "unknown endpoint"})
                return
            self.reply(200, {"ready": True, "weights_sha256": policy.config["weights_sha256"],
                             "deployment_qualified": policy.config.get("deployment_qualified", False),
                             "horizon": 15, "hz": 30, "target_space": "servo"})

        def do_POST(self):
            if self.path not in ("/act", "/reset"):
                self.reply(404, {"error": "unknown endpoint"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 <= size <= 2_000_000:
                    raise ValueError("request exceeds size limit")
                obs = json.loads(self.rfile.read(size) or "{}")
                if self.path == "/reset":
                    policy.reset()
                    self.reply(200, {"reset": True})
                    return
                obs["images"] = {key: decode_image(obs["images"][key]) for key in ("scene", "wrist")}
                start = time.monotonic()
                chunk = policy.act(obs)
                seconds = time.monotonic()-start
                # Leave the remainder of this lease's 60% duty budget idle.
                if policy.device.type == "cuda":
                    time.sleep(seconds*(1/a.gpu_duty-1))
                self.reply(200, {"chunk": chunk.tolist(), "action": "absolute_q", "target_space": "servo",
                                 "hz": 30, "inference_seconds": seconds})
            except (ValueError, KeyError, TypeError) as exc:
                self.reply(400, {"error": str(exc)})

        def log_message(self, *args):
            pass

    print(json.dumps({"listening": f"http://127.0.0.1:{a.port}", "motion": False,
                      "max_seconds": a.max_seconds}), flush=True)
    server = HTTPServer(("127.0.0.1", a.port), Handler)
    server.timeout = 1
    deadline = time.monotonic()+a.max_seconds
    try:
        while time.monotonic() < deadline:
            server.handle_request()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
