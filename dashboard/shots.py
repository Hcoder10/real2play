"""Capture real-time Chrome DevTools screenshots (never virtual-time-budget).

python dashboard/shots.py --url http://127.0.0.1:8800 --out dashboard/shots/mock.png
Uses an isolated Chrome/Edge profile, or attach using --cdp-url. No browser package
or browser download is required; the WebSocket transport below uses the stdlib.
"""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request


class CDP:
    def __init__(self, url):
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "ws" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Only a local ws:// DevTools endpoint is supported")
        self.socket = socket.create_connection((parsed.hostname, parsed.port), timeout=20)
        self.buffer = bytearray()
        key = base64.b64encode(os.urandom(16)).decode()
        target = parsed.path + ("?" + parsed.query if parsed.query else "")
        request = (f"GET {target} HTTP/1.1\r\nHost: {parsed.netloc}\r\nUpgrade: websocket\r\n"
                   f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n")
        self.socket.sendall(request.encode())
        response = bytearray()
        while b"\r\n\r\n" not in response:
            block = self.socket.recv(4096)
            if not block:
                raise ConnectionError("DevTools closed during WebSocket handshake")
            response.extend(block)
        headers, rest = response.split(b"\r\n\r\n", 1)
        accept = base64.b64encode(hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest())
        if not headers.startswith(b"HTTP/1.1 101") or accept not in headers:
            raise ConnectionError("DevTools WebSocket upgrade failed")
        self.buffer.extend(rest)
        self.next_id = 0
        self.errors = []

    def _read(self, n):
        while len(self.buffer) < n:
            block = self.socket.recv(max(4096, n - len(self.buffer)))
            if not block:
                raise ConnectionError("DevTools disconnected")
            self.buffer.extend(block)
        value = bytes(self.buffer[:n])
        del self.buffer[:n]
        return value

    def _send(self, data, opcode=1):
        mask = os.urandom(4)
        length = len(data)
        header = bytes([0x80 | opcode])
        header += bytes([0x80 | length]) if length < 126 else (
            bytes([0x80 | 126]) + struct.pack("!H", length) if length < 65536 else
            bytes([0x80 | 127]) + struct.pack("!Q", length))
        self.socket.sendall(header + mask + bytes(byte ^ mask[i % 4] for i, byte in enumerate(data)))

    def _receive(self):
        payload = bytearray()
        while True:
            first, second = self._read(2)
            length = second & 127
            if length == 126:
                length = struct.unpack("!H", self._read(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._read(8))[0]
            mask = self._read(4) if second & 128 else None
            block = self._read(length)
            if mask:
                block = bytes(b ^ mask[i % 4] for i, b in enumerate(block))
            opcode = first & 15
            if opcode == 8:
                raise ConnectionError("DevTools closed its WebSocket")
            if opcode == 9:
                self._send(block, 10)
                continue
            if opcode in {0, 1}:
                payload.extend(block)
                if first & 128:
                    return json.loads(payload)

    def call(self, method, **params):
        self.next_id += 1
        self._send(json.dumps({"id": self.next_id, "method": method, "params": params}).encode())
        while True:
            response = self._receive()
            if response.get("method") == "Runtime.exceptionThrown":
                self.errors.append(response["params"])
            if response.get("id") == self.next_id:
                if "error" in response:
                    raise RuntimeError(response["error"])
                return response.get("result", {})

    def evaluate(self, expression, *, await_promise=False):
        result = self.call("Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=await_promise)
        if "exceptionDetails" in result:
            raise RuntimeError(result["exceptionDetails"])
        return result.get("result", {}).get("value")

    def close(self):
        self.socket.close()


def browser_path(explicit=None):
    candidates = [explicit] if explicit else []
    candidates += [shutil.which(name) for name in ("google-chrome", "chromium", "chromium-browser", "msedge", "chrome")]
    for var, suffix in (("PROGRAMFILES", "Google/Chrome/Application/chrome.exe"),
                        ("PROGRAMFILES(X86)", "Microsoft/Edge/Application/msedge.exe"),
                        ("LOCALAPPDATA", "Google/Chrome/Application/chrome.exe")):
        if os.getenv(var):
            candidates.append(str(Path(os.environ[var]) / suffix))
    candidates += ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                   "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise FileNotFoundError("Chrome/Edge not found; supply --browser <executable> or --cdp-url http://127.0.0.1:<port>")


def endpoint_json(url):
    with urllib.request.urlopen(url, timeout=3) as response:
        return json.load(response)


def capture(url, out, *, width=1920, height=1080, wait=3, browser=None, cdp_url=None, headful=False):
    process = None
    client = None
    profile = None
    try:
        if not cdp_url:
            profile = tempfile.TemporaryDirectory(prefix="real2play-shots-")
            args = [browser_path(browser), f"--user-data-dir={profile.name}", "--remote-debugging-port=0",
                    "--no-first-run", "--no-default-browser-check", "--disable-background-networking",
                    "--disable-component-update", "--hide-scrollbars", "about:blank"]
            if not headful:
                args.insert(1, "--headless=new")
            process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            port_file = Path(profile.name) / "DevToolsActivePort"
            deadline = time.monotonic() + 20
            while not port_file.exists():
                if time.monotonic() > deadline or process.poll() is not None:
                    raise TimeoutError("Browser did not expose DevTools")
                time.sleep(0.1)
            cdp_url = "http://127.0.0.1:" + port_file.read_text().splitlines()[0]
        tabs = endpoint_json(cdp_url.rstrip("/") + "/json/list")
        tab = next(row for row in tabs if row.get("type") == "page")
        client = CDP(tab["webSocketDebuggerUrl"])
        client.call("Page.enable")
        client.call("Runtime.enable")
        client.call("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=1, mobile=False)
        client.call("Page.navigate", url=url)
        deadline = time.monotonic() + 25
        while not client.evaluate("document.body && document.body.dataset.ready === 'true'"):
            if time.monotonic() > deadline:
                raise TimeoutError("Dashboard did not load /api/state within 25 seconds")
            time.sleep(0.15)
        client.evaluate("document.fonts.ready.then(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))))", await_promise=True)
        time.sleep(max(0, wait))  # actual wall-clock paint time, not virtual-time emulation
        metrics = client.evaluate("({width:innerWidth,height:innerHeight,scrollWidth:document.documentElement.scrollWidth,scrollHeight:document.documentElement.scrollHeight,panels:[...document.querySelectorAll('.panel-heading h1')].map(e=>e.textContent),mock:document.body.dataset.mock})")
        result = client.call("Page.captureScreenshot", format="png", captureBeyondViewport=False, fromSurface=True)
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(base64.b64decode(result["data"]))
        evidence = {"url": url, "screenshot": str(out.resolve()), "wall_clock_wait_s": wait,
                    "virtual_time": False, "metrics": metrics, "javascript_errors": client.errors}
        out.with_suffix(".json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        if client.errors:
            raise RuntimeError(f"Screenshot captured, but JavaScript errors recorded in {out.with_suffix('.json')}")
        return evidence
    finally:
        if client:
            if process:
                try:
                    client.call("Browser.close")
                except (OSError, RuntimeError):
                    pass
            client.close()
        if process:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()  # Only the isolated process started above.
                process.wait(timeout=5)
        if profile:
            # Windows browser children can briefly hold profile handles after Browser.close.
            for attempt in range(10):
                try:
                    profile.cleanup()
                    break
                except PermissionError:
                    if attempt == 9:
                        break
                    time.sleep(0.1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8800")
    parser.add_argument("--out", type=Path, default=Path("dashboard/shots/dashboard.png"))
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--wait", type=float, default=3)
    parser.add_argument("--browser")
    parser.add_argument("--cdp-url")
    parser.add_argument("--headful", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(capture(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
