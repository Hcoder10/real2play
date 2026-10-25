"""Read-only camera/status publisher. It never connects to the robot bus or armd."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import time
import urllib.request

from r2p.pipeline.storage import atomic_json


class Publisher:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.frame_sha256 = None

    def update(self, status, frame=None):
        self.directory.mkdir(parents=True, exist_ok=True)
        if frame is not None:
            import cv2
            ok, jpeg = cv2.imencode(".jpg", frame)
            if not ok:
                raise ValueError("Cannot encode publisher frame")
            fd, temporary = tempfile.mkstemp(dir=self.directory, suffix=".jpg")
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(jpeg.tobytes())
                os.replace(temporary, self.directory / "frame.jpg")
                self.frame_sha256 = hashlib.sha256(jpeg.tobytes()).hexdigest()
            finally:
                Path(temporary).unlink(missing_ok=True)
        atomic_json(self.directory / "status.json", {**status, "schema": "r2p_robot_live/1", "updated": time.time(),
                                                    "frame_sha256": self.frame_sha256})


def make_server(directory, camera_url, *, host="127.0.0.1", port=8791, timeout=2):
    directory = Path(directory)

    def status():
        try:
            row = json.loads((directory / "status.json").read_text(encoding="utf-8"))
            if time.time() - row["updated"] < 5 and row.get("frame_sha256") and (directory / "frame.jpg").is_file():
                return row
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return {"schema": "r2p_robot_live/1", "source": "real", "status": "camera_only",
                "updated": time.time(), "camera_source": "phone_read_only", "motion": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            try:
                row = status()
                if self.path == "/status.json":
                    body, mime = json.dumps(row).encode(), "application/json"
                elif self.path == "/frame.jpg":
                    if row.get("status") == "camera_only":
                        with urllib.request.urlopen(camera_url, timeout=timeout) as response:
                            body = response.read(8 * 1024 * 1024)
                        if not body.startswith(b"\xff\xd8"):
                            raise ValueError("Phone camera did not return JPEG")
                    else:
                        body = (directory / "frame.jpg").read_bytes()
                        if hashlib.sha256(body).hexdigest() != row["frame_sha256"]:
                            raise ValueError("Publisher changed frame; retry with matching provenance")
                    mime = "image/jpeg"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except (OSError, ValueError):
                self.send_error(503, "Camera unavailable")
    return ThreadingHTTPServer((host, port), Handler)


def main(argv=None):
    from .config import load_config, ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--directory", type=Path, default=ROOT / "out/robot/live")
    args = parser.parse_args(argv)
    cfg = load_config()["camera"]
    server = make_server(args.directory, cfg["url"].format(index=cfg.get("index", 0)), port=args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
