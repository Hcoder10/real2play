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
        step = newest((self.root / "runs/argon_step").glob("*/ledger.jsonl"))
        if step and newest(p for p in (step, path) if p) == step:
            return self.step_lab(step)
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

    def step_lab(self, path):
        """The step loop records observations BEFORE each proposed action."""
        run = path.parent
        rows = ledger(path, limit=None)
        turns, actions, thought = {}, [], None
        for row in rows:
            turn, reply = row.get("turn"), row.get("reply")
            if not isinstance(turn, int) or isinstance(turn, bool) or not isinstance(reply, dict):
                continue
            turns[turn] = row
            if isinstance(reply.get("thought"), str):
                thought = reply["thought"]
            action = reply.get("action", {})
            if not isinstance(action, dict):
                continue
            labels = []
            move = action.get("move")
            if isinstance(move, list) and len(move) == 3 and all(finite(v) for v in move):
                labels.append("move [" + ", ".join(f"{v:g}" for v in move) + "] m")
            if finite(action.get("gripper")):
                labels.append(f"gripper {action['gripper']:g}")
            if action.get("done") is True:
                labels.append("done")
            if labels:
                actions.append({"turn": turn, "label": " / ".join(labels), "action": action})
        slides = []
        for frame in run.glob("t*_real_vs_sim.jpg"):
            token = frame.name.split("_", 1)[0][1:]
            if not token.isdigit():
                continue
            turn = int(token)
            slides.append({"turn": turn, "comparison": self.media(frame),
                           "wrist": self.media(run / f"t{token}_wrist.jpg")})
        slides.sort(key=lambda slide: slide["turn"])
        last = turns[max(turns)] if turns else {}
        action = last.get("reply", {}).get("action", {})
        complete = isinstance(action, dict) and action.get("done") is True
        updated = path.stat().st_mtime
        status = "done" if complete else "running" if time.time() - updated < 120 else "pending"
        sim_path = run / "sim.json"
        sim = object_json(sim_path)
        arm = sim.get("arm", {})
        arm = arm if isinstance(arm, dict) else {}
        xy = arm.get("xy_offset_mm")
        summary = {"available": bool(sim), "source": sim_path.relative_to(self.root).as_posix(),
                   "objects": len(sim["objects"]) if isinstance(sim.get("objects"), list) else None,
                   "z_mm": arm.get("z_offset_mm") if finite(arm.get("z_offset_mm")) else None,
                   "xy_mm": xy if isinstance(xy, list) and len(xy) == 2 and all(finite(v) for v in xy) else None,
                   "notes": str(arm.get("notes") or sim.get("notes") or "No arm notes recorded")}
        placement = complete and "r1c1" in summary["notes"].lower() and "picked" in summary["notes"].lower()
        latest = slides[-1] if slides else {}
        return {"available": True, "kind": "argon_step", "run": run.name,
                "source": path.relative_to(self.root).as_posix(), "status": status, "state": status,
                "synthetic": False, "label": "Real bench", "model": "Gemini 4 Argon",
                "turn": last.get("turn"), "turn_count": len(turns), "updated": updated,
                "thought": thought, "actions": actions[-6:], "slides": slides, "sim": summary,
                "frames": {"comparison": latest.get("comparison"), "wrist": latest.get("wrist")},
                "placement": "Real pick-and-place into r1c1" if placement else None,
                "calls": [], "history": [], "edits": [], "video": None, "complete": complete}

    def after_step(self):
        root = self.root / "runs/after_step"
        # Choose one run, then keep its game and loop evidence together.
        paths = list(root.glob("*/game/**/result.json")) + list(root.glob("*/loop/game_loop.json"))
        latest = newest(paths)
        if not latest:
            return {}
        run = root / latest.relative_to(root).parts[0]
        build_path = newest((run / "game").rglob("result.json"))
        loop_path = run / "loop/game_loop.json"
        build = object_json(build_path) if build_path else {}
        loop = object_json(loop_path)
        return {"run": run.name, "updated": latest.stat().st_mtime,
                "build": {key: build[key] for key in ("passed", "published", "deployment_ready", "status", "synthetic", "dry_run") if key in build},
                "build_source": build_path.relative_to(self.root).as_posix() if build_path else None,
                "loop": {key: loop[key] for key in ("status", "stage", "error", "counts", "started", "finished", "evidence") if key in loop},
                "loop_source": loop_path.relative_to(self.root).as_posix()}

    def policy(self):
        path = self.root / "exports/wide_v3/config.json"
        config = object_json(path)
        evaluation = config.get("eval", {})
        evaluation = evaluation if isinstance(evaluation, dict) else {}
        values, counts = {}, {}
        for field, key in (("clean", "clean"), ("half", "realx0.5"), ("real", "real")):
            report = evaluation.get(key, {})
            report = report if isinstance(report, dict) else {}
            value, count = report.get("success"), report.get("n", evaluation.get("n"))
            values[field] = value if finite(value) and 0 <= value <= 1 else None
            counts[field] = count if finite(count) and count > 0 else None
        available = any(value is not None for value in values.values())
        ns = set(counts.values())
        return {"available": available, "name": "wide_v3", "source": path.relative_to(self.root).as_posix(),
                "evidence": "Twin eval", "n": next(iter(ns)) if len(ns) == 1 else None,
                "counts": counts, **values,
                "message": "state policy, twin eval; real-arm deployment pending (bench-profile expert)"
                           if available else "Awaiting wide_v3 recorded twin evaluations"}

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
                "invalid": rejected, "total": sum(groups.values()), "after_step": self.after_step()}

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
        # The judges' chain is evidence from individual stages. An orchestrator
        # rehearsal is never allowed to overwrite these recorded artifacts.
        def node(name, status="pending", metric="Awaiting evidence", provenance="", detail="", source=None):
            return {"name": name, "state": status, "metric": metric, "provenance": provenance,
                    "detail": detail, "source": source}

        stages = {n: node(n) for n in ("Real", "Twin", "Game", "Play", "Replay", "Train", "Deploy")}
        if lab["available"]:
            if lab.get("kind") == "argon_step":
                stages["Real"] = node("Real", lab["state"], number(lab["turn_count"], " turns"),
                                      lab.get("placement") or "Real bench experiments",
                                      lab.get("thought"), lab["source"])
                sim = lab["sim"]
                xy = ", ".join(number(v) for v in sim["xy_mm"]) if sim["xy_mm"] else "unknown"
                offsets = f"z {number(sim['z_mm'])} mm / xy [{xy}] mm"
                stages["Twin"] = node("Twin", "done" if sim["available"] else "pending",
                                      number(sim["objects"], " objects"), offsets,
                                      sim["notes"], sim["source"])
            else:
                stages["Real"] = node("Real", "done" if lab["frames"].get("scene") else "pending",
                                      number(lab.get("turn"), " turns"), lab["label"], lab.get("run"))
                stages["Twin"] = node("Twin", "done" if lab.get("complete") else "pending",
                                      number(len(lab["edits"]), " edited files"), lab["label"], lab.get("why"))
        after = game.get("after_step", {})
        build, loop = after.get("build", {}), after.get("loop", {})
        if build:
            passed = build.get("passed") is True
            build_state = "done" if passed else "running" if state(build.get("status")) == "running" else "pending"
            provenance = "Synthetic build" if synthetic(build) else "Recorded game build"
            stages["Game"] = node("Game", build_state, "Build checks passed" if passed else "Build checks pending",
                                  provenance, f"Published: {build.get('published')}; deployment ready: {build.get('deployment_ready')}",
                                  after.get("build_source"))
        groups = game["groups"]
        loop_state = state(loop.get("status"))
        play_state = "running" if loop_state == "running" else "done" if game["total"] else "pending"
        if loop and loop_state in {"blocked", "pending"}:
            play_state = "pending"
        stages["Play"] = node("Play", play_state, f"{groups['human']} human / {groups['headless']} headless",
                              f"{groups['bot']} bot / {groups['expert']} expert / {groups['unknown']} unknown",
                              f"Loop: {loop.get('status', 'pending')} at {loop.get('stage', 'unknown')}. "
                              + str(loop.get("error") or "Saved trace metadata"), after.get("loop_source"))
        if policy.get("available"):
            def rate(key):
                value = policy.get(key)
                return f"{value * 100:.1f}%" if finite(value) else "pending"
            n = policy.get("n")
            evidence = f"wide_v3 · twin · n={n}" if n else "wide_v3 · twin"
            complete = all(finite(policy.get(key)) for key in ("clean", "half", "real"))
            stages["Replay"] = node("Replay", "done" if complete else "pending", f"{rate('half')} half noise",
                                    evidence, "Recorded twin evaluations; separate from the current play loop.", policy["source"])
            stages["Train"] = node("Train", "done" if complete else "pending", f"{rate('clean')} / {rate('real')}",
                                   evidence, "Clean / full-noise twin success. " + policy["message"], policy["source"])
        if deploy["available"]:
            physical = deploy.get("physical_placement") is True and not deploy["synthetic"]
            done = physical and state(deploy.get("status")) == "done"
            running = not deploy["synthetic"] and state(deploy.get("status")) == "running"
            stages["Deploy"] = node("Deploy", "done" if done else "running" if running else "pending",
                                    "Physical placement" if done else "Real-arm pending",
                                    "Mock board recorded" if deploy["synthetic"] else "Recorded board",
                                    deploy.get("why"), self.data.config.get("ttt_status", "out/ttt/status.json"))
        return {"stages": list(stages.values()), "source": "Recorded artifacts · " + (lab.get("run") or "waiting for lab"),
                "synthetic": False, "updated": after.get("updated", lab.get("updated"))}

    def snapshot(self):
        lab, game, deploy = self.lab(), self.game(), self.deploy()
        policy = self.policy()
        return {"updated": time.time(), "lab": lab, "game": game, "deploy": deploy, "policy": policy,
                "chain": self.chain(lab, game, policy, deploy), "mock": False}
