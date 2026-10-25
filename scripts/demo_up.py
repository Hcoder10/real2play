"""Start the read-only phone publisher, dashboard and authenticated trace receiver.

Ctrl-C stops only servers created here. Set TRACE_KEY to match the Roblox server.
"""
import argparse
import json
import os
from pathlib import Path
import secrets
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def phone_check(url):
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            data = response.read(8 * 1024 * 1024)
        import cv2
        import numpy as np
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Response is not an image")
        return {"available": True, "width": frame.shape[1], "height": frame.shape[0], "read_only": True}
    except (OSError, ValueError) as exc:
        return {"available": False, "error": type(exc).__name__, "read_only": True}


def main(argv=None):
    from dashboard.adapters import DashboardData, read_json
    from dashboard.server import make_server as dashboard_server
    from r2p.robot.config import load_config
    from r2p.robot.publisher import make_server as publisher_server
    from roblox.tools.trace_server import make_server as trace_server
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--check", action="store_true", help="Phone check and URLs only; start no servers")
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--publisher-port", type=int, default=8791)
    parser.add_argument("--trace-port", type=int, default=8770)
    parser.add_argument("--duration", type=float, help="Stop after this many seconds (smoke verification)")
    parser.add_argument("--config", type=Path, default=ROOT / "dashboard/config.json")
    args = parser.parse_args(argv)
    camera = load_config()["camera"]
    url = camera["url"].format(index=camera.get("index", 0))
    print("Phone camera: " + json.dumps(phone_check(url)), flush=True)
    for label, address in (("Dashboard", f"http://127.0.0.1:{args.port}"),
                           ("Runs", f"http://127.0.0.1:{args.port}/runs"),
                           ("Publisher", f"http://127.0.0.1:{args.publisher_port}/status.json"),
                           ("Trace health", f"http://127.0.0.1:{args.trace_port}/health")):
        print(f"{label}: {address}", flush=True)
    if args.check:
        return 0
    config = read_json(args.config)
    if not isinstance(config, dict):
        parser.error("Dashboard config is not a JSON object")
    config.update(camera_url=f"http://127.0.0.1:{args.publisher_port}/frame.jpg",
                  camera_status_url=f"http://127.0.0.1:{args.publisher_port}/status.json")
    key_file = ROOT / "out/demo/trace_key.txt"
    key = os.getenv("TRACE_KEY")
    if not key:
        key_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with key_file.open("x", encoding="utf-8") as stream:
                stream.write(secrets.token_urlsafe(32))
        except FileExistsError:
            pass
        key = key_file.read_text(encoding="utf-8").strip()
        print(f"Trace upload key file: {key_file} (configure this key in the Roblox server)", flush=True)
    servers, threads = [], []
    try:
        servers.append(publisher_server(ROOT / "out/robot/live", url, port=args.publisher_port))
        servers.append(dashboard_server(DashboardData(ROOT, config, mock=args.mock), port=args.port))
        servers.append(trace_server("127.0.0.1", args.trace_port, ROOT / "roblox/out/traces", key))
        for server in servers:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            threads.append(thread)
        print("Ready. Ctrl-C stops these demo servers; arm daemon is untouched.", flush=True)
        deadline = time.monotonic() + args.duration if args.duration else float("inf")
        while time.monotonic() < deadline:
            time.sleep(.25)
    except KeyboardInterrupt:
        pass
    finally:
        for server in servers[:len(threads)]:
            server.shutdown()
        for server in servers:
            server.server_close()
        for thread in threads:
            thread.join(timeout=3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
