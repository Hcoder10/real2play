"""Authenticated, idempotent 600-tick trace uploads; Python standard library only.

POST /trace: X-Trace-Key, JSON {trace_id, part, final, meta?, ticks, meta_final?}.
Parts can arrive out of order. Only complete, validated r2p_trace/1 files become
visible under out/traces. No secrets are returned by /health or request logs.
"""
from __future__ import annotations
import argparse
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import threading
import time
import sys
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from r2p.traces import write_trace

MAX_BYTES = 4 * 1024 * 1024
MAX_PARTS = 64


def finite_vector(v, size):
    return isinstance(v, list) and len(v) == size and all(
        isinstance(n, (int, float)) and not isinstance(n, bool) and math.isfinite(n) for n in v)


def validate_tick(tick):
    if not isinstance(tick, dict) or not finite_vector(tick.get("q"), 6) or not finite_vector(tick.get("q_target"), 6):
        raise ValueError("tick needs finite q and q_target of length six")
    if not isinstance(tick.get("t"), (int, float)) or not math.isfinite(tick["t"]) or tick["t"] < 0:
        raise ValueError("tick needs finite nonnegative t")
    if not isinstance(tick.get("poses"), dict) or not isinstance(tick.get("grip_closed"), bool):
        raise ValueError("tick needs poses and grip_closed")
    for pose in tick["poses"].values():
        if not isinstance(pose, dict) or not finite_vector(pose.get("xyz"), 3):
            raise ValueError("pose needs xyz")
        if not isinstance(pose.get("yaw_deg", 0), (int, float)) or not math.isfinite(pose.get("yaw_deg", 0)):
            raise ValueError("pose needs finite yaw")
    if not isinstance(tick.get("command"), str) or not isinstance(tick.get("phase"), str):
        raise ValueError("tick needs command and phase")


class TraceStore:
    def __init__(self, out, key):
        if not key:
            raise ValueError("TRACE_KEY must not be empty")
        self.out = Path(out)
        self.parts = self.out / ".parts"
        self.parts.mkdir(parents=True, exist_ok=True)
        self.key = key
        self.lock = threading.Lock()

    def stats(self):
        tasks = {}
        for path in self.out.glob("*.jsonl"):
            try:
                with path.open(encoding="utf-8") as stream:
                    meta = json.loads(stream.readline())["meta"]
                if meta.get("test_only"):
                    continue
                name = meta["task_name"]
                row = tasks.setdefault(name, {"rounds": 0, "successes": 0, "sources": Counter(), "fail_reasons": Counter()})
                row["rounds"] += 1
                row["successes"] += int(meta["success"])
                row["sources"][meta["source"]] += 1
                if not meta["success"]:
                    row["fail_reasons"][meta.get("fail_reason") or "unknown"] += 1
            except (OSError, ValueError, KeyError):
                continue
        for row in tasks.values():
            row["success_rate"] = row["successes"] / row["rounds"]
        return {"schema": "r2p_game_stats/1", "rounds": sum(r["rounds"] for r in tasks.values()), "tasks": tasks}

    def accept(self, payload):
        trace_id = payload.get("trace_id", "")
        if not isinstance(trace_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", trace_id):
            raise ValueError("invalid trace_id")
        part = payload.get("part")
        if type(part) is not int or not 0 <= part < MAX_PARTS:
            raise ValueError("invalid part")
        ticks = payload.get("ticks")
        if not isinstance(ticks, list) or len(ticks) > 600 or type(payload.get("final")) is not bool:
            raise ValueError("part must contain <=600 ticks and final boolean")
        for tick in ticks:
            validate_tick(tick)
        if part == 0:
            meta = payload.get("meta")
            if not isinstance(meta, dict) or meta.get("schema") != "r2p_trace/1" or meta.get("source") not in ("human", "bot"):
                raise ValueError("part zero needs r2p_trace/1 human/bot metadata")
            if not meta.get("task") or not meta.get("task_name"):
                raise ValueError("metadata needs task and task_name")
        clean = {k: v for k, v in payload.items() if k != "key"}
        encoded = json.dumps(clean, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        with self.lock:
            dest = self.out / f"{trace_id}.jsonl"
            folder = self.parts / trace_id
            folder.mkdir(exist_ok=True)
            chunk = folder / f"{part:03d}.json"
            if chunk.exists():
                if hashlib.sha256(chunk.read_bytes()).hexdigest() != digest:
                    raise ValueError("conflicting retry")
            else:
                if dest.exists():
                    raise ValueError("trace already finalized")
                temp = chunk.with_suffix(".tmp")
                temp.write_bytes(encoded)
                temp.replace(chunk)
            parts = {int(p.stem): json.loads(p.read_text(encoding="utf-8")) for p in folder.glob("*.json")}
            final = [i for i, p in parts.items() if p["final"]]
            if len(final) > 1 or (final and max(parts) > final[0]):
                raise ValueError("inconsistent final part")
            complete = bool(final and all(i in parts for i in range(final[0] + 1)))
            count = sum(len(p["ticks"]) for p in parts.values())
            if complete:
                meta = dict(parts[0]["meta"])
                trailer = parts[final[0]].get("meta_final", {})
                for name in ("schema", "task", "task_name", "source", "player", "level", "seed", "scene", "rig", "physics_sha256", "robot_mjcf_sha256", "finger_contact_sha256"):
                    if name in trailer and name in meta and trailer[name] != meta[name]:
                        raise ValueError(f"meta_final cannot change {name}")
                meta.update(trailer)
                if type(meta.get("success")) is not bool or not isinstance(meta.get("commands"), list):
                    raise ValueError("final metadata needs success and commands")
                if meta.get("schema") != "r2p_trace/1" or meta.get("source") not in ("human", "bot"):
                    raise ValueError("final metadata changed the trace contract")
                ordered = [t for i in range(final[0] + 1) for t in parts[i]["ticks"]]
                if not ordered:
                    raise ValueError("empty trace")
                for i, tick in enumerate(ordered):
                    if abs(tick["t"] - i / 30) > 1e-6:
                        raise ValueError("ticks must start at zero and have exact 30 Hz spacing")
                for command in meta["commands"]:
                    if (not isinstance(command, list) or len(command) != 3 or
                        not isinstance(command[2], str) or not finite_vector(command[:2], 2) or
                        not 0 <= command[0] <= command[1] <= len(ordered) / 30 + 1e-6):
                        raise ValueError("invalid command interval")
                write_trace(dest, meta, ordered)
            return {"ok": True, "trace_id": trace_id, "part": part, "complete": complete, "ticks": count}


def make_server(host, port, out, key):
    store = TraceStore(out, key)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, status, value):
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self.reply(200, {"ok": True, "schema": "r2p_trace/1", "traces": len(list(store.out.glob("*.jsonl")))})
            elif self.path in ("/stats", "/presets"):
                key = self.headers.get("X-Trace-Key", "")
                if not hmac.compare_digest(key.encode(), store.key.encode()):
                    self.reply(401, {"error": "unauthorized"}); return
                if self.path == "/stats":
                    self.reply(200, store.stats())
                else:
                    root = Path(__file__).resolve().parents[2]
                    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
                    self.reply(200, {"game": read(root / "roblox/presets/game/workshop.json"),
                                     "tasks": {p.parent.name: read(p) for p in (root / "tasks").glob("*/task.json")},
                                     "physics": {p.stem: read(p) for p in (root / "config/physics").glob("*.json")}})
            else:
                self.reply(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/trace":
                self.reply(404, {"error": "not found"}); return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BYTES:
                    self.reply(413, {"error": "body size outside limit"}); return
                self.connection.settimeout(15)
                data = self.rfile.read(length)
                payload = json.loads(data.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("body must be an object")
                key = self.headers.get("X-Trace-Key", payload.get("key", ""))
                if not isinstance(key, str) or not hmac.compare_digest(key.encode(), store.key.encode()):
                    self.reply(401, {"error": "unauthorized"}); return
                self.reply(200, store.accept(payload))
            except (ValueError, KeyError, TypeError, UnicodeError) as error:
                self.reply(400, {"error": str(error)})
            except TimeoutError:
                self.reply(408, {"error": "body timeout"})

    server = ThreadingHTTPServer((host, port), Handler)
    server.store = store
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "out/traces")
    parser.add_argument("--key", default=os.environ.get("TRACE_KEY", ""))
    args = parser.parse_args()
    with make_server(args.host, args.port, args.out, args.key) as server:
        print(f"Trace server http://{args.host}:{server.server_port} -> {args.out}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
