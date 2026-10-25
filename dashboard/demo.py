"""Read-only projector view of the experiment ledger and pipeline artifacts.

This module never imports the robot, simulator, model client, or training code.
"""
from collections import deque
import difflib
import json
from pathlib import Path
import time

from dashboard.adapters import finite, read_json, synthetic
from dashboard.video import BrowserVideos

GAME_URL = "https://www.roblox.com/games/86276076528667/SquarianSquares-Place-10252026-1"
CAMERAS = {"phone": "http://127.0.0.1:8792/frame/0.jpg",
           "wrist": "http://127.0.0.1:8794/frame/0.jpg"}
METRICS = {"joint_deg": "joint_rms_deg", "jaw_mm": "jaw_path_rms_mm",
           "object_mm": "object_final_pose_error_mm", "image_similarity": "image_similarity"}


def object_json(path):
    value = read_json(path, {})
    return value if isinstance(value, dict) else {}


def newest(paths):
    stamped = []
    for path in paths:
        try:
            stamped.append((path.stat().st_mtime_ns, str(path), path))
        except OSError:
            pass
    return max(stamped)[2] if stamped else None


def ledger(path, limit=500):
    try:
        with path.open("rb") as stream:
            # Tool responses can include large scene files. Bound dashboard memory.
            stream.seek(0, 2)
            offset = max(0, stream.tell() - 16 * 1024 * 1024)
            stream.seek(offset)
            if offset:
                stream.readline()
            lines = deque(stream.read().decode("utf-8-sig", errors="replace").splitlines(), maxlen=limit)
        rows = []
        for line in lines:
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
            except ValueError:
                pass  # The final append may not have finished yet.
        return rows
    except OSError:
        return []


def state(value):
    value = str(value or "pending").lower()
    if value in {"done", "complete", "completed", "ok", "passed", "success", "goal_reached"}:
        return "done"
    if value in {"blocked", "error", "failed", "refused", "incomplete", "mock_complete"}:
        return "blocked"
    if value in {"running", "thinking", "observing", "tool", "recording", "training", "executing"}:
        return "running"
    return "pending"


def number(value, suffix=""):
    return f"{value:,.2f}".rstrip("0").rstrip(".") + suffix if finite(value) else "Awaiting evidence"


class DemoData:
    def __init__(self, data):
        self.data, self.root = data, data.root
        self.trace_cache = {}
        self.videos = BrowserVideos(self.root)

    def media(self, path):
        if not path:
            return None
        return self.data.asset(str(path), self.root)

    def lab(self):
        # Only direct run directories: acceptance-test subdirectories are not live runs.
        path = newest((self.root / "runs/lab").glob("*/status.json"))
        if not path:
            return {"available": False, "state": "pending", "thought": None, "calls": [],
                    "history": [], "edits": [], "frames": {}, "video": None}
        run, status = path.parent, object_json(path)
        rows = ledger(run / "ledger.jsonl")
        thought, calls, history, edits, contents = None, [], [], {}, {}
        for row in rows:
            response = row.get("response", row)
            if not isinstance(response, dict):
                continue
            if isinstance(response.get("thought"), str):
                thought = response["thought"]
            request_calls = response.get("calls", [])
            results = row.get("results", [])
            for index, result in enumerate(results if isinstance(results, list) else []):
                if not isinstance(result, dict):
                    continue
                tool = result.get("tool", "tool")
                value = result.get("result", {})
                value = value if isinstance(value, dict) else {}
                failed = bool(result.get("error")) or state(value.get("status")) == "blocked" or value.get("ok") is False
                calls.append({"turn": row.get("turn"), "tool": tool, "ok": not failed,
                              "detail": str(result.get("error") or value.get("error") or value.get("status") or "Recorded"),
                              "time": row.get("time")})
                residuals = value.get("residuals", value.get("current", value.get("candidate", value.get("metrics", value))))
                if isinstance(residuals, dict) and any(finite(residuals.get(k)) for k in METRICS.values()):
                    history.append({"turn": row.get("turn"), **{k: residuals.get(v) if finite(residuals.get(v)) else None
                                                               for k, v in METRICS.items()}})
                if tool == "read_sim" and isinstance(value.get("content"), str):
                    contents[value.get("path")] = value["content"]
                validation = value.get("validation", value)
                if tool not in {"edit_sim", "managed_build", "crop_texture"} or result.get("error"):
                    continue
                args = request_calls[index].get("args", {}) if index < len(request_calls) else {}
                names = validation.get("edited", [])
                if not names and args.get("path"):
                    names = [args["path"]]
                for name in names:
                    item = edits.setdefault(name, {"path": name, "edits": 0, "added": 0, "removed": 0,
                                                   "diff_known": True})
                    item["edits"] += 1
                    text = args.get("files", {}).get(name, args.get("content"))
                    if name in contents and isinstance(text, str) and args.get("encoding") != "base64":
                        diff = list(difflib.ndiff(contents[name].splitlines(), text.splitlines()))
                        item["added"] += sum(line.startswith("+ ") for line in diff)
                        item["removed"] += sum(line.startswith("- ") for line in diff)
                    else:
                        item["diff_known"] = False
                    if isinstance(text, str):
                        contents[name] = text
        replay = status.get("latest_replay") or {}
        if not history and isinstance(replay, dict) and isinstance(replay.get("residuals"), dict):
            history.append({"turn": status.get("turn"), **{k: replay["residuals"].get(v) if finite(replay["residuals"].get(v)) else None
                                                         for k, v in METRICS.items()}})
        video = newest(run.rglob("side_by_side*.mp4"))
        video_kind = "comparison"
        if video is None:
            video = newest((run / "replays").rglob("*.mp4"))
            video_kind = "simulated replay"
        browser_video, video_message = self.videos.get(video)
        frames = {}
        for camera in ("scene", "wrist"):
            frame = newest((run / "turns").glob(f"*/{camera}*.jpg"))
            if frame is None:
                frame = newest((run / "turns").glob(f"*/{camera}*.png"))
            frames[camera] = self.media(frame)
        preview = self.media(run / "previews/scene.jpg")
        is_sim = status.get("real_mock") is True or synthetic(status)
        return {"available": True, "run": run.name, "status": status.get("status"),
                "state": state(status.get("status")), "synthetic": is_sim,
                "label": "Simulated reference" if is_sim else "Real bench" if status.get("real_mock") is False else "Unverified source",
                "turn": status.get("turn"), "updated": status.get("updated", path.stat().st_mtime),
                "model": status.get("model"), "thought": thought, "why": status.get("error") or status.get("why"),
                "calls": calls[-10:], "history": history[-80:], "edits": list(edits.values()),
                "frames": frames, "preview": preview, "video": self.media(browser_video), "video_kind": video_kind,
                "video_message": video_message, "original_video": self.media(video),
                "heldout": status.get("heldout", {}), "complete": status.get("complete") is True}

    def game(self):
        groups = dict.fromkeys(("human", "headless", "bot", "expert", "unknown"), 0)
        paths = set()
        for directory in self.data.config.get("traces", ["roblox/out/traces"]):
            paths.update(self.data.path(directory).rglob("*.jsonl"))
        rejected = 0
        for path in paths:
            try:
                stat = path.stat()
                stamp = (stat.st_size, stat.st_mtime_ns)
                cached = self.trace_cache.get(path)
                if cached and cached[0] == stamp:
                    meta = cached[1]
                else:
                    with path.open(encoding="utf-8-sig") as stream:
                        meta = json.loads(stream.readline()).get("meta", {})
                    if not isinstance(meta, dict) or meta.get("schema") != "r2p_trace/1":
                        raise ValueError("Unknown trace")
                    self.trace_cache[path] = (stamp, meta)
                if meta.get("test_only"):
                    continue
                source = "headless" if meta.get("headless") else meta.get("source", "unknown")
                # A synthetic human fixture must never count as human play.
                if source == "human" and synthetic(meta):
                    source = "unknown"
                groups[source if source in groups else "unknown"] += 1
            except (OSError, ValueError, AttributeError):
                rejected += 1
        try:
            health = self.data.fetch_json("http://127.0.0.1:8770/health")
            health = health if isinstance(health, dict) else {}
        except (OSError, ValueError):
            health = {}
        return {"url": self.data.config.get("game_url") or GAME_URL, "groups": groups,
                "receiver_online": health.get("ok") is True, "receiver_traces": health.get("traces"),
                "invalid": rejected, "total": sum(groups.values())}

    def deploy(self):
        status = object_json(self.data.path(self.data.config.get("ttt_status", "out/ttt/status.json")))
        decision = status.get("decision", status)
        decision = decision if isinstance(decision, dict) else {}
        board = decision.get("board", status.get("board"))
        if isinstance(board, list) and len(board) == 3 and all(isinstance(r, list) for r in board):
            board = [v for row in board for v in row]
        valid = isinstance(board, list) and len(board) == 9
        marks = {"X": "X", "O": "O", 1: "X", 2: "O"}
        board = [marks.get(x, "") if isinstance(x, (str, int)) else "" for x in board] if valid else [""] * 9
        latency = decision.get("latency_s", status.get("latency_s", decision.get("elapsed_s", decision.get("latency"))))
        if not finite(latency) and finite(decision.get("latency_ms")):
            latency = decision["latency_ms"] / 1000
        return {"available": bool(status), "board": board, "board_valid": valid,
                "move": decision.get("move"), "why": decision.get("why", decision.get("reason", status.get("argon_says"))),
                "model": decision.get("model", status.get("model")), "source": decision.get("source", status.get("source")),
                "latency_s": latency if finite(latency) else None, "updated": status.get("updated"),
                "status": status.get("status"), "synthetic": synthetic(status),
                "physical_placement": status.get("physical_placement"),
                "cameras": {name: f"/demo/camera/{name}.jpg" for name in CAMERAS}}

    def chain(self, lab, game, policy, deploy):
        def node(name, status="pending", metric="Awaiting evidence", provenance="", detail=""):
            return {"name": name, "state": status, "metric": metric, "provenance": provenance, "detail": detail}
        stages = {n: node(n) for n in ("Real", "Twin", "Game", "Play", "Replay", "Train", "Deploy")}
        if lab["available"]:
            stages["Real"] = node("Real", "done" if lab["frames"].get("scene") else "pending",
                                  number(lab.get("turn"), " turns"), lab["label"], lab.get("run", ""))
            stages["Twin"] = node("Twin", lab["state"], number(len(lab["edits"]), " edited files"), lab["label"], lab.get("why"))
        build_path = newest((self.root / "runs").glob("game_live_*/result.json"))
        build = object_json(build_path) if build_path else {}
        stages["Game"] = node("Game", "done", "Published game", "Roblox", "Published link supplied for the judges' demo")
        if build:
            stages["Game"]["detail"] = f"Latest build checks: {build.get('passed')}; this build published: {build.get('published')}"
        stages["Play"] = node("Play", "done" if game["total"] else "pending", f"{game['groups']['human']} human / {game['groups']['headless']} headless",
                              "Trace metadata", f"{game['groups']['bot']} bots; {game['groups']['expert']} experts")
        loop = object_json(self.root / "runs/game_loop_latest.json")
        counts = loop.get("counts", {})
        if counts:
            stages["Replay"] = node("Replay", state(loop.get("status")), number(counts.get("relabeled"), " relabeled"), "Twin replay")
        manifests = sorted((self.root / "runs").glob("*/manifest.json"), key=lambda p: p.stat().st_mtime)
        for path in manifests:
            manifest = object_json(path)
            for row in manifest.get("rounds", {}).values():
                for key, item in row.get("stages", {}).items():
                    target = {"label": "Replay", "train": "Train", "deploy": "Deploy"}.get(key)
                    if target and item.get("status"):
                        stages[target] = node(target, state(item["status"]), str(item["status"]).lower(),
                                              "Synthetic pipeline" if synthetic(manifest) else "Pipeline", item.get("message", ""))
        if policy.get("available"):
            stages["Train"] = node("Train", "done", number((policy.get("clean") or 0) * 100, "% clean"),
                                   policy.get("evidence", "Simulation eval"), policy.get("name"))
        if deploy["available"]:
            is_done = deploy.get("physical_placement") is True and state(deploy.get("status")) == "done"
            stages["Deploy"] = node("Deploy", "done" if is_done else "pending" if deploy["synthetic"] else state(deploy.get("status")),
                                    "Simulated board" if deploy["synthetic"] else str(deploy.get("status") or "Board read"),
                                    "Synthetic" if deploy["synthetic"] else "Recorded board", deploy.get("why"))
        path = newest((self.root / "runs/argon_orchestrator").glob("*/status.json"))
        orchestrator = object_json(path) if path else {}
        if orchestrator:
            rows = ledger(path.with_name("ledger.jsonl"))
            results = dict(orchestrator.get("results") or {})
            for row in rows:
                if row.get("kind") == "result" and isinstance(row.get("result"), dict):
                    results[row.get("tool")] = row["result"]
            mapping = {"Real": ("measure_scene", "calibrate_arm"), "Twin": ("active_twin", "analyze_experiments", "twin_gate"),
                       "Game": ("build_game",), "Play": ("collect_play",), "Replay": ("replay_and_train",),
                       "Train": ("replay_and_train",), "Deploy": ("deploy_real",)}
            preferred = {"Real": ("heldout_px", " px"), "Twin": ("jaw_mm", " mm jaw"),
                         "Game": ("local_checks", " checks"), "Play": ("human", " human"),
                         "Replay": ("replayed", " replayed"), "Train": ("eval_n", " eval scenes"), "Deploy": ("attempts", " attempts")}
            for name, tools in mapping.items():
                result = next((results[t] for t in reversed(tools) if isinstance(results.get(t), dict)), None)
                active = orchestrator.get("last_tool") in tools
                provenance = "Synthetic orchestrator" if synthetic(orchestrator) else "Orchestrator"
                if result:
                    key, suffix = preferred[name]
                    metrics = result.get("metrics", {})
                    stages[name] = node(name, "done" if result.get("ok") is True else "blocked",
                                        number(metrics.get(key), suffix), provenance, result.get("error") or result.get("evidence") or "")
                elif active:
                    stages[name] = node(name, state(orchestrator.get("status")), str(orchestrator.get("last_tool")), provenance,
                                        orchestrator.get("error") or orchestrator.get("why"))
                else:
                    stages[name] = node(name, provenance=provenance)
                if active and state(orchestrator.get("status")) == "blocked":
                    stages[name].update(state="blocked", detail=orchestrator.get("error"))
        return {"stages": list(stages.values()), "source": path.parent.name if path else "Individual artifacts",
                "synthetic": synthetic(orchestrator), "updated": orchestrator.get("updated"),
                "status": orchestrator.get("status"), "why": orchestrator.get("error") or orchestrator.get("why")}

    def snapshot(self):
        lab, game, deploy = self.lab(), self.game(), self.deploy()
        policy = self.data.policy()
        return {"updated": time.time(), "lab": lab, "game": game, "deploy": deploy, "policy": policy,
                "chain": self.chain(lab, game, policy, deploy), "mock": False}
