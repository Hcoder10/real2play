"""Read-only adapters. Missing evidence is null, never a fabricated zero or success."""

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import threading
import time
import urllib.request


def read_json(path, default=None):
    try:
        with Path(path).open(encoding="utf-8-sig") as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return default


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def synthetic(row):
    return (row.get("synthetic") is True or row.get("dry_run") is True or
            row.get("mode") in {"dry-run", "dry_run", "synthetic", "mock"} or
            row.get("session_kind") in {"synthetic", "mock", "sim"} or
            row.get("source") in {"synthetic", "mock", "sim", "expert", "bot", "oracle", "mock_truth", "synthbench_truth"} or
            row.get("physical_placement") is False)


def real_evidence(row):
    return not synthetic(row) and (row.get("source") == "real" or row.get("mode") == "real" or
                                   row.get("session_kind") in {"real", "bench"})


def ratio(value):
    if isinstance(value, dict):
        value = value.get("success_rate", value.get("rate", value.get("success")))
    return value if finite(value) and 0 <= value <= 1 else None


class DashboardData:
    def __init__(self, root, config, *, show_dry_runs=False, mock=False, fetch_json=None):
        self.root = Path(root).resolve()
        self.config = config
        self.show_dry_runs = show_dry_runs
        self.mock = mock
        self.fetch_json = fetch_json or self._fetch
        self.assets = {}
        self.asset_lock = threading.Lock()
        self.trace_cache = {}
        self.publisher = {}

    def path(self, value):
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def _fetch(self, url):
        with urllib.request.urlopen(url, timeout=self.config.get("publisher_timeout_s", 0.6)) as response:
            return json.load(response)

    def asset(self, value, base):
        if not value or not isinstance(value, str):
            return None
        candidate = Path(value)
        path = (candidate if candidate.is_absolute() else base / candidate).resolve()
        if not path.is_relative_to(self.root) or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"} or not path.is_file():
            return None
        token = hashlib.sha256(str(path).encode()).hexdigest()[:24]
        with self.asset_lock:
            self.assets[token] = path
        return f"/media/{token}"

    def trace(self, path):
        try:
            stat = path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
            if self.trace_cache.get(path, (None,))[0] == stamp:
                return self.trace_cache[path][1]
            with path.open(encoding="utf-8-sig") as stream:
                meta = json.loads(next(stream)).get("meta", {})
                if meta.get("schema") != "r2p_trace/1":
                    raise ValueError("unsupported trace schema")
                times = []
                for line in stream:
                    if line.strip():
                        tick = json.loads(line)
                        if finite(tick.get("t")):
                            times.append(tick["t"])
            row = {"meta": meta, "duration_s": max(0, max(times) - min(times) + 1 / 30) if times else 0,
                   "path": str(path.relative_to(self.root))}
            self.trace_cache[path] = (stamp, row)
            return row
        except (OSError, ValueError, KeyError, StopIteration, TypeError, AttributeError):
            return None

    def real(self):
        try:
            status = self.fetch_json(self.config["camera_status_url"])
            status = status if isinstance(status, dict) else {}
            error = None
        except (OSError, ValueError, KeyError) as exc:
            status, error = {}, "Camera publisher offline"
        self.publisher = status
        is_synthetic = synthetic(status)
        visible = real_evidence(status) or (self.show_dry_runs and is_synthetic)
        attempts = []
        for path in sorted(self.path(self.config.get("deploy_runs", "runs")).rglob("*.jsonl")):
            if "attempt" not in str(path).lower() and path.name != "trace.jsonl":
                continue
            row = self.trace(path)
            if row and (real_evidence(row["meta"]) or (self.show_dry_runs and synthetic(row["meta"]))):
                attempts.append({"path": row["path"], "success": row["meta"].get("success"),
                                 "synthetic": synthetic(row["meta"]), "duration_s": row["duration_s"]})
        return {"available": visible, "frame": "/camera/frame.jpg" if visible else None,
                "status": status if visible else {}, "synthetic": is_synthetic if visible else False,
                "label": "SIM DRY RUN" if visible and is_synthetic else "REAL" if visible else "NO REAL FEED",
                "message": error or ("Synthetic publisher hidden" if is_synthetic and not visible else
                                      "Publisher needs explicit real/synthetic provenance" if not visible else "Camera publisher connected"),
                "attempts": attempts[-12:], "argon_says": status.get("argon_says") if visible else None}

    def twin(self):
        pinned = self.config.get("real2sim_run")
        discovered = None
        if not pinned and self.config.get("auto_discover"):
            candidates = list(self.path(self.config.get("real2sim_live", "out/live")).rglob("run.json"))
            candidates += list(self.path(self.config.get("runs", "runs")).glob("*/r*/twin/*/run.json"))
            candidates = [p for p in candidates if (read_json(p, {}) or {}).get("schema") == "r2s_live/1"]
            if candidates:
                # Prefer recorded bench evidence; synthetic runs stay explicitly labelled.
                discovered = max(candidates, key=lambda p: (not synthetic(read_json(p, {})), p.stat().st_mtime_ns))
                pinned = str(discovered.parent.relative_to(self.root))
        if not pinned:
            return {"available": False, "message": "Pin real2sim_run in dashboard/config.json"}
        base = discovered.parent if discovered else self.path(self.config.get("real2sim_live", "out/live")) / pinned
        run = read_json(base / "run.json")
        if not isinstance(run, dict) or run.get("schema") != "r2s_live/1":
            return {"available": False, "message": f"Pinned twin {pinned} is missing or incomplete"}
        history, iterations = [], []
        best_gap = None
        for entry in run.get("iters", []):
            detail = read_json(base / entry.get("file", f"iter_{entry['iter']}.json"), {})
            metrics = detail.get("metrics", entry.get("metrics", {}))
            objects = metrics.get("objects", {})
            ious = [obj["iou"] for obj in objects.values()
                    if isinstance(obj, dict) and finite(obj.get("iou")) and 0 <= obj["iou"] <= 1]
            gap = 1 - sum(ious) / len(ious) if ious else None
            kept = entry.get("kept", detail.get("kept", False))
            if gap is not None and kept:
                best_gap = min(best_gap, gap) if best_gap is not None else gap
            motion = metrics.get("physics", {}).get("traj_err_px")
            row = {"iter": entry["iter"], "gap": gap, "best_gap": best_gap,
                   "motion_px": motion if finite(motion) else None, "kept": kept,
                   "verdict": entry.get("verdict", detail.get("why", ""))}
            history.append(row)
            iterations.append((entry, detail, row))
        selected = None
        if iterations:
            if run.get("status") in {"done", "complete", "completed"}:
                selected = next((item for item in iterations if item[0]["iter"] == run.get("best_iter")), None)
            else:
                selected = iterations[-1]
        images = []
        if selected:
            entry, detail, _ = selected
            links = detail.get("images", {})
            overlay = self.asset(links.get("overlay_0", entry.get("thumb")), base)
            sim = self.asset(links.get("sim"), base)
            real = self.asset(links.get("real"), base)
            motion = self.asset(links.get("physics", links.get("motion")), base)
            for label, src, crop in (("SIM", sim or overlay, "full" if sim else "right"),
                                     ("REAL", real or overlay, "full" if real else "left"),
                                     ("MOTION", motion, "full")):
                if src:
                    images.append({"label": label, "src": src, "crop": crop})
        return {"available": True, "run": pinned, "status": run.get("status"), "stage": run.get("stage"),
                "smoke_fixture": bool(run.get("smoke_fixture")),
                "synthetic": synthetic(run), "session_kind": run.get("session_kind", "unknown"),
                "history": history, "gap": selected[2]["gap"] if selected else None,
                "motion_px": selected[2]["motion_px"] if selected else None, "images": images,
                "best_iter": run.get("best_iter"), "shown_iter": selected[0]["iter"] if selected else None,
                "stop_reason": run.get("stop_reason"), "calls": run.get("calls", {}),
                "message": ("Synthetic oracle fixture; no model critique" if run.get("smoke_fixture") else
                            "Best accepted twin" if run.get("status") == "done" else run.get("stage", "In progress"))}

    def game(self):
        groups = {source: {"rounds": 0, "successes": 0, "hours": 0.0} for source in ("human", "bot", "expert", "other")}
        levels, players = {}, {}
        invalid = 0
        paths = set()
        for directory in self.config.get("traces", ["roblox/out/traces"]):
            paths.update(self.path(directory).rglob("*.jsonl"))
        for path in sorted(paths):
            row = self.trace(path)
            if not row:
                invalid += 1
                continue
            meta = row["meta"]
            if meta.get("test_only"):
                continue
            source = meta.get("source", "other")
            source = source if source in groups else "other"
            target = groups[source]
            target["rounds"] += 1
            target["successes"] += meta.get("success") is True
            target["hours"] += row["duration_s"] / 3600
            task = meta.get("task", {})
            task = task.get("name", "Unknown task") if isinstance(task, dict) else task
            level = str(meta.get("level") or meta.get("task_name") or task)
            level_row = levels.setdefault(level, {"name": level, "human": 0, "bot": 0, "other": 0, "successes": 0})
            level_row[source if source in {"human", "bot"} else "other"] += 1
            level_row["successes"] += meta.get("success") is True and source == "human"
            if source == "human":
                player = str(meta.get("player", "Anonymous"))
                score = players.setdefault(player, {"player": player, "rounds": 0, "successes": 0})
                score["rounds"] += 1
                score["successes"] += meta.get("success") is True
        return {"url": self.config.get("game_url"), "groups": groups, "levels": sorted(levels.values(), key=lambda r: r["name"]),
                "leaderboard": sorted(players.values(), key=lambda r: (-r["successes"], -r["rounds"], r["player"]))[:5],
                "invalid_traces": invalid, "available": bool(paths)}

    def policy(self):
        pinned = self.config.get("headline_eval")
        if not pinned and self.config.get("auto_discover"):
            candidates = [p for p in (self.root / "exports").glob("*/config.json")
                          if (read_json(p, {}) or {}).get("eval", {}).get("n", 0)]
            if candidates:
                chosen = max(candidates, key=lambda p: (not synthetic(read_json(p, {})), p.stat().st_mtime_ns))
                pinned = str(chosen.relative_to(self.root))
        result = {"available": False, "message": "Pin headline_eval in dashboard/config.json", "scaling": []}
        if pinned:
            report = read_json(self.path(pinned))
            if isinstance(report, dict):
                evaluation = report.get("eval", report)
                clean, real = ratio(evaluation.get("clean")), ratio(evaluation.get("real"))
                # Pipeline paired reports carry per-scene booleans rather than aggregate rates.
                for noise in ("clean", "real"):
                    rows = report.get("results", {}).get(noise, [])
                    if rows and all(type(r.get("success")) is bool for r in rows):
                        value = sum(r["success"] for r in rows) / len(rows)
                        if noise == "clean":
                            clean = value
                        else:
                            real = value
                result.update(available=clean is not None or real is not None, clean=clean, real=real,
                              n=evaluation.get("n", len(report.get("results", {}).get("clean", [])) or None),
                              name=report.get("name", report.get("policy", Path(pinned).parent.name)), pinned=pinned,
                              evidence="SMOKE EVALUATION" if report.get("smoke_only") or report.get("synthetic") else "TWIN EVALUATION",
                              message=("Tiny offline smoke test; not a production success rate" if report.get("smoke_only") or report.get("synthetic")
                                       else "Pinned evaluation • simulation"))
            else:
                result["message"] = f"Pinned evaluation unavailable: {pinned}"
        scaling = read_json(self.path(self.config.get("scaling", "runs/scaling.json")), [])
        if isinstance(scaling, dict):
            scaling = scaling.get("points", scaling.get("runs", []))
        for row in scaling if isinstance(scaling, list) else []:
            if not isinstance(row, dict):
                continue
            n = row.get("episodes", row.get("n", row.get("traces")))
            clean = ratio(row.get("clean", row.get("success_rate")))
            if finite(n) and n >= 0 and clean is not None:
                result["scaling"].append({"episodes": n, "clean": clean, "real": ratio(row.get("real"))})
        result["scaling"].sort(key=lambda r: r["episodes"])
        return result

    def runs(self):
        result = []
        for path in sorted(self.path(self.config.get("runs", "runs")).glob("*/manifest.json")):
            manifest = read_json(path)
            if isinstance(manifest, dict) and manifest.get("schema") == "r2p_run/1":
                result.append({**manifest, "label": "SIM DRY RUN" if synthetic(manifest) else
                               "REAL" if manifest.get("mode") == "real" else "PLAN"})
        return result

    def ttt(self):
        status = read_json(self.path(self.config.get("ttt_status", "out/ttt/status.json")), {})
        if not status:
            status = self.publisher.get("ttt", self.publisher)
        decision = status.get("decision", status)
        board = decision.get("board")
        if not isinstance(board, list) or len(board) != 9 or any(x not in ("X", "O", "", None, ".") for x in board):
            return {"available": False, "board": [""] * 9, "why": "Waiting for the next board reading", "model": None}
        return {"available": True, "board": [x if x in {"X", "O"} else "" for x in board],
                "why": decision.get("why", status.get("argon_says", "No explanation recorded")),
                "model": decision.get("model", status.get("model")), "source": decision.get("source"),
                "move": decision.get("move"), "synthetic": synthetic(status),
                "status": status.get("status"), "updated": status.get("updated"),
                "label": "SYNTHETIC" if synthetic(status) else "RECORDED BOARD"}

    def snapshot(self):
        if self.mock:
            from .mock import snapshot
            state = snapshot()
            state["game"]["url"] = self.config.get("game_url")
            return state
        real = self.real()
        return {"mock": False, "updated": time.time(), "real": real, "twin": self.twin(), "game": self.game(),
                "policy": self.policy(), "ttt": self.ttt(), "runs": self.runs()}
