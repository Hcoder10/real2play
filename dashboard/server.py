"""Stdlib HTTP server: python dashboard/server.py --mock (default port 8800)."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import sys
import threading
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.adapters import DashboardData, read_json, real_evidence, synthetic


def qr_png(url):
    """Encode locally using the project's existing OpenCV dependency, no third-party QR service."""
    import cv2
    code = cv2.QRCodeEncoder_create().encode(url)
    code = cv2.copyMakeBorder(code, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
    code = cv2.resize(code, (280, 280), interpolation=cv2.INTER_NEAREST)
    ok, encoded = cv2.imencode(".png", code)
    if not ok:
        raise ValueError("QR encoding failed")
    return encoded.tobytes()


def make_server(data, host="127.0.0.1", port=8800):
    static = Path(__file__).parent / "static"
    lock = threading.Lock()
    cached = {"at": 0, "value": None}

    def snapshot():
        with lock:
            if time.monotonic() - cached["at"] > 1:
                cached.update(at=time.monotonic(), value=data.snapshot())
            return cached["value"]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            # Keep the demo terminal readable; request paths can contain untrusted trace text.
            pass

        def send(self, body, mime="application/json", status=200):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            path = urllib.parse.urlparse(self.path).path
            try:
                if path in {"/", "/runs"}:
                    html = (static / "index.html").read_text(encoding="utf-8").replace("</head>",
                        '<link rel="stylesheet" href="/static/argon.css"><script src="/static/argon.js" defer></script></head>')
                    self.send(html.encode("utf-8"), "text/html; charset=utf-8")
                elif path == "/api/argon":
                    from dashboard.argon import snapshot as argon_snapshot
                    self.send(json.dumps(argon_snapshot(data.root), allow_nan=False).encode())
                elif path == "/api/state":
                    self.send(json.dumps(snapshot(), allow_nan=False).encode())
                elif path == "/api/runs":
                    self.send(json.dumps(snapshot()["runs"], allow_nan=False).encode())
                elif path.startswith("/static/"):
                    file = (static / urllib.parse.unquote(path[len("/static/"):])).resolve()
                    if not file.is_relative_to(static.resolve()) or not file.is_file():
                        return self.send(b"Not found", "text/plain", 404)
                    self.send(file.read_bytes(), mimetypes.guess_type(file.name)[0] or "application/octet-stream")
                elif path.startswith("/media/"):
                    with data.asset_lock:
                        file = data.assets.get(path.split("/")[-1])
                    if not file:
                        return self.send(b"Not found", "text/plain", 404)
                    self.send(file.read_bytes(), mimetypes.guess_type(file.name)[0])
                elif path == "/camera/frame.jpg":
                    # Recheck provenance; do not proxy a synthetic frame under a stale REAL badge.
                    status = data.fetch_json(data.config["camera_status_url"])
                    if not (real_evidence(status) or data.show_dry_runs and synthetic(status)):
                        return self.send(b"No visible real feed", "text/plain", 404)
                    with urllib.request.urlopen(data.config["camera_url"], timeout=1) as response:
                        self.send(response.read(8 * 1024 * 1024), "image/jpeg")
                elif path == "/qr.png":
                    url = data.config.get("game_url")
                    if not url or urllib.parse.urlparse(url).scheme not in {"https", "http"}:
                        return self.send(b"Set game_url in dashboard/config.json", "text/plain", 404)
                    self.send(qr_png(url), "image/png")
                else:
                    self.send(b"Not found", "text/plain", 404)
            except (OSError, ValueError, KeyError, TypeError):
                self.send(b'{"error":"Source unavailable; retry after its publisher updates"}', status=503)

    return ThreadingHTTPServer((host, port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true", help="Illustrative fixtures, visibly labelled MOCK")
    parser.add_argument("--show-dry-runs", action="store_true")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args(argv)
    config = read_json(args.config)
    if not isinstance(config, dict):
        parser.error(f"Invalid dashboard config: {args.config}")
    data = DashboardData(args.root, config, show_dry_runs=args.show_dry_runs, mock=args.mock)
    server = make_server(data, args.host or config.get("host", "127.0.0.1"), args.port if args.port is not None else config.get("port", 8800))
    print(f"Real2Play {'MOCK PREVIEW' if args.mock else 'LIVE SOURCES'} http://{server.server_address[0]}:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
