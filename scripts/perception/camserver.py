#!/usr/bin/env python3
"""Terminal-launched camera broker: GET /list and /frame/<index>.jpg."""
from __future__ import annotations

import argparse
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from r2p.perception.camera import LiveCamera


def make_handler(cameras):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/list":
                payload = [{"index": index, "width": 1280, "height": 720,
                            "age_s": camera.age_s if camera.age_s != float("inf") else None,
                            "error": camera.error, "url": f"/frame/{index}.jpg"}
                           for index, camera in cameras.items()]
                self._send(200, json.dumps(payload).encode(), "application/json")
                return
            match = re.fullmatch(r"/frame/(\d+)\.jpg", self.path)
            if not match or int(match[1]) not in cameras:
                self.send_error(404, "Camera not found")
                return
            camera = cameras[int(match[1])]
            try:
                frame = camera.read(timeout=.1)
            except TimeoutError:
                self.send_error(503, "Camera has no frame yet")
                return
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                self.send_error(503, "JPEG encode failed")
                return
            self._send(200, encoded.tobytes(), "image/jpeg", camera.age_s)

        def _send(self, status, payload, content_type, age=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            if age is not None:
                self.send_header("X-Frame-Age", f"{age:.6f}")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass
    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8792)
    parser.add_argument("--indices", type=int, nargs="+", help="Explicit devices; default scans every index below --scan")
    parser.add_argument("--scan", type=int, default=10)
    args = parser.parse_args(argv)
    cameras = {}
    try:
        for index in args.indices if args.indices is not None else range(args.scan):
            try:
                camera = LiveCamera({"index": index, "width": 1280, "height": 720, "fps": 30,
                                     "backend": "AVFOUNDATION" if sys.platform == "darwin" else "ANY"})
                cameras[index] = camera
            except OSError:
                continue  # a hole in the device numbering does not end discovery
        if not cameras:
            raise SystemExit("No cameras available. On macOS launch camserver.command from Terminal and grant camera permission.")
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(cameras))
        print(f"Serving cameras {list(cameras)} at http://127.0.0.1:{args.port}/list", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    finally:
        for camera in cameras.values():
            camera.close()


if __name__ == "__main__":
    main()
