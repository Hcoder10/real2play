"""Local adapters for the Argon controller. No arbitrary commands from model output."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

from r2p import gemini
from r2p.orchestrator import THRESHOLDS, digest, read_json


def load_script(name, root):
    path = root / "scripts/lead" / (name + ".py")
    spec = importlib.util.spec_from_file_location("r2p_lead_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def managed_helpers(root):
    # These scripts deliberately share a helper without importing their CLI entry points.
    from scripts.argon import _managed
    return _managed


def gate_metrics(metrics, baseline=None):
    reasons = []
    for key in ("joint_deg", "jaw_mm", "object_mm", "iou"):
        value = metrics.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            reasons.append("missing/nonfinite " + key)
        elif key == "iou" and value < THRESHOLDS[key] or key != "iou" and value > THRESHOLDS[key]:
            reasons.append(key + " outside threshold")
    if not metrics.get("rows") or any(row.get("split") != "heldout" for row in metrics["rows"]):
        reasons.append("missing held-out experiments")
    for row in metrics.get("rows", []):
        for key in ("joint_deg", "jaw_mm", "object_mm", "iou"):
            value = row.get(key)
            if type(value) not in (int, float) or not math.isfinite(value):
                reasons.append("missing per-experiment " + key)
            elif key == "iou" and value < THRESHOLDS[key] or key != "iou" and value > THRESHOLDS[key]:
                reasons.append("per-experiment " + key + " outside threshold")
    if baseline:
        for oid, row in baseline.get("objects", {}).items():
            if metrics.get("objects", {}).get(oid, {}).get("iou", -1) < row["iou"] - .01:
                reasons.append("object IoU regression: " + oid)
    return not reasons, reasons


class LocalTools:
    def __init__(self, runner):
        from r2p.sim.twin import load_physics
        self.r = runner
        self.root, self.out, self.dry = runner.root, runner.out, runner.dry_run
        self.task = deepcopy(runner.task)
        self.physics = load_physics(self.root / "config/physics/so101_bench.json")
        self.observations = []
        self.campaign = None
        self.folders = {}
        self.generation = 0

    def folder(self, name):
        self.generation += 1
        folder = self.out / f"{self.generation:03d}_{name}"
        self.folders[name] = folder
        return folder

    def action(self, name, args):
        details = {
            "measure_scene": "GET the phone camera, measure the grid, and spend one Argon request locating supply caps; no motion",
            "calibrate_arm": "Point at recorded jaw images with Argon and fit an inactive candidate map; no motion or activation",
            "active_twin": f"Run {args.get('budget', 4 if self.dry else 6)} guarded experiments via the existing armd; record camera and joint telemetry",
            "deploy_real": f"Run ONE learned-policy placement on cell {args.get('cell', 'r1c1')} via armd, timeout 60s, export {getattr(self, 'export', 'not selected')}",
            "replay_and_train": "Replay traces locally; write a GPU request for the lead (NO GPU launch); wait for a matching export receipt",
            "build_game": "Run paid Antigravity game generation and local Luau/parity tests" +
                          ("; request owner Roblox publication and await receipt" if args.get("publish") else "; save unpublished candidate"),
            "analyze_experiments": "Run paid Antigravity experiment analysis; verify bounded candidate against local held-out recordings",
            "author_task": "Run paid Antigravity task author; save a separate task candidate and checker vectors",
            "train_plan": "Run paid Antigravity training configuration author and validate its data mix locally",
        }
        return ("SIMULATION ONLY: " if self.dry else "") + details.get(name, name + " with " + json.dumps(args))

    def execute(self, name, args):
        return getattr(self, name)(**args)

    def bench_state(self):
        """File evidence only for the one-call live planning smoke; zero camera or arm commands."""
        paths = ["config/bench.json", "config/joint_map.json", "runs/bench/grid.json", "out/ttt/status.json"]
        paths += [str(p.relative_to(self.root)) for p in sorted((self.root / "runs/active_twin").glob("*/status.json"))[-3:]]
        paths += [str(p.relative_to(self.root)) for p in sorted((self.root / "out/live").glob("*/run.json"))[-2:]]
        evidence = {}
        for name in paths:
            path = self.root / name
            if path.is_file():
                value = read_json(path)
                if name == "config/bench.json":
                    value = {k: value[k] for k in ("robot_id", "daemon_url", "gravity_comp", "safety") if k in value}
                evidence[name] = {"modified": path.stat().st_mtime, "data": value}
        from r2p.agents import assert_no_secrets
        assert_no_secrets(json.dumps(evidence))
        return {"source": "recorded files; freshness is not current physical verification", "files": evidence,
                "task": self.task, "physics": self.physics}

    def measure_scene(self):
        import numpy as np
        from r2p.perception import CameraModel
        folder = self.folder("measure_scene")
        if self.dry:
            from r2p.real2sim.active import bench_scene
            self.scene = bench_scene(self.task)
            self.camera = CameraModel.look_at(position=(.43, -.42, .43), target=(.18, -.06, .035), image_size=(320, 240))
            caps = [o["xyz"][:2] for o in self.scene["objects"] if not o.get("fixed")]
            cells, cell_m, age = self.task.get("board", {}).get("cells", {}), self.task.get("board", {}).get("cell_m", 0), 0.
            evidence = "synthetic scene fixture; no observed real grid/caps"
        else:
            import cv2
            from r2p.real2sim.active import http_json
            from urllib.request import urlopen
            self.camera = CameraModel.load(self.root / "config/camera.json")
            health = http_json("http://127.0.0.1:8792/list")["0"]
            age = float(health["age_s"])
            if not math.isfinite(age) or not 0 <= age < 2:
                raise ValueError("Camera is stale; cannot measure scene")
            with urlopen("http://127.0.0.1:8792/frame/0.jpg", timeout=5) as response:
                frame = cv2.imdecode(np.frombuffer(response.read(8 * 1024 * 1024), np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Camera returned no image")
            try:
                cells, cell_m, preview = load_script("measure_grid", self.root).measure(frame, self.camera)
            except SystemExit as exc:
                raise ValueError(str(exc)) from None
            caps = load_script("ttt_armd", self.root).locate_caps(frame, self.camera)
            if not caps or not .025 < cell_m < .05:
                raise ValueError("No supply caps or invalid grid dimensions")
            folder.mkdir(parents=True)
            cv2.imwrite(str(folder / "grid.jpg"), preview)
            cv2.imwrite(str(folder / "frame.jpg"), frame)
            scene_path = self.r.options.get("scene")
            if not scene_path:
                raise ValueError("Live experiment geometry requires --scene with a measured scene")
            self.scene = read_json(scene_path)
            self.scene["board"] = {**self.task.get("board", {}), "cells": cells, "cell_m": cell_m,
                                   "center_xy": cells["r1c1"]}
            piece = next(o for o in self.scene["objects"] if not o.get("fixed"))
            piece["xyz"][:2] = caps[0]
            self.task["board"] = deepcopy(self.scene["board"])
            for obj in self.scene["objects"]:
                if obj["id"] in cells:
                    obj["xyz"][:2] = cells[obj["id"]]
            for obj in self.task["objects"]:
                if obj["id"] in cells:
                    obj["xy"] = cells[obj["id"]]
            evidence = "camera GET + calibrated grid measurement + Argon cap pointing"
        self.caps = caps
        gemini.atomic_json(folder / "scene.json", self.scene)
        gemini.atomic_json(folder / "task.json", self.task)
        self.camera.save(folder / "camera.json")
        return {"metrics": {"grid_cells": len(cells), "cell_mm": float(cell_m) * 1000,
                            "caps": len(caps), "camera_age_s": age}, "evidence": evidence, "folder": str(folder)}

    def calibrate_arm(self):
        import numpy as np
        from r2p.model import Arm, JointMap
        from r2p.sysid import visual
        folder = self.folder("calibrate_arm")
        if self.dry:
            arm, base = Arm(), JointMap(sign=np.array([-1, 1, 1, 1, -1.]))
            truth = JointMap(sign=base.sign, offset=np.array([2., -3., 4., -5., 8.]))
            rng, rows = np.random.default_rng(123), []
            for i in range(8):
                owner = np.r_[rng.uniform([-30, -40, 20, 20, -60], [30, 20, 70, 70, 60]), rng.uniform(20, 70)]
                tips = self.camera.project(np.array(arm.tips(truth.to_model(owner))[:2]))
                rows.append({"k": i, "owner_deg": owner.tolist(), "visible": True,
                             "tips_px": dict(zip(("fixed_tip", "moving_tip"), tips.tolist()))})
        else:
            source = self.r.options.get("poses")
            if not source:
                raise ValueError("Provide --poses recorded poses, including open-jaw frames; calibration never moves the arm")
            base = JointMap.load(self.root / "config/joint_map.json")
            rows = visual.point(source, out_path=folder / "pointed.json", camera=self.camera, joint_map=base)
        gaps = [np.linalg.norm(np.asarray(row["tips_px"]["fixed_tip"]) - row["tips_px"]["moving_tip"])
                for row in rows if row.get("visible")]
        if sum(gap > 5 for gap in gaps) < 3:
            raise ValueError("Need at least three open-jaw frames to observe wrist roll/gripper")
        result = visual.fit(rows, self.camera, base, gate_px=THRESHOLDS["calibration_px"])
        visual.report(result, folder)
        gemini.atomic_json(folder / "candidate_map.json", result["candidate"])
        return {"metrics": {"passed": int(result["accepted"]), "heldout_px": result["loo_p90_px"],
                            "poses": result["n_poses"], "open_jaw_frames": int(sum(g > 5 for g in gaps)), "activated": 0},
                "candidate": str(folder / "candidate_map.json"), "limitations": result["limitations"]}

    def active_twin(self, budget=None):
        from r2p.real2sim.active import SimMover, ArmdMover, run_campaign
        folder = self.folder("active_twin")
        self.observations = []
        if self.dry:
            mover = SimMover(self.scene, truth_physics=self.physics, camera=self.camera)
        else:
            live = load_script("active_live", self.root)
            piece = next(o for o in self.scene["objects"] if not o.get("fixed"))
            perceiver = live.ArgonCapPerceiver(self.camera, piece["id"], piece["xyz"][:2])
            mover = ArmdMover(self.scene, self.camera, perceiver=perceiver, reset_timeout_s=30)
        execute = mover.execute
        def record(experiment, destination, **kwargs):
            if not self.dry:
                frame, _, _ = mover.capture()
                xyz = perceiver.read(frame)
                if xyz is None:
                    raise ValueError("Experiment cap not visible; no motion")
                perceiver.last = (time.monotonic(), xyz)
                next(o for o in mover.scene["objects"] if o["id"] == piece["id"])["xyz"] = xyz.tolist()
            observation = execute(experiment, destination, **kwargs)
            self.observations.append(observation)
            return observation
        mover.execute = record
        result = run_campaign(mover, budget=budget or (4 if self.dry else 6), out_dir=folder,
                              physics=self.physics, candidate_dir=folder / "candidates", max_nfev=2 if self.dry else 9)
        self.campaign = result
        self.physics = read_json(result["candidate_physics"])
        self.scene = read_json(folder / "candidate_scene.json")
        return {"metrics": {"experiments": result["experiments_used"], "before_error": result["before"]["error"],
                            "after_error": result["heldout"]["error"], **{k: result["heldout"][k] for k in THRESHOLDS if k != "calibration_px"}},
                "candidate": result["candidate_physics"], "status": str(folder / "status.json"), "activated": False}

    def analyze_experiments(self):
        from r2p.orchestrator_managed import analyze
        folder = self.folder("analyze_experiments")
        return analyze(self, folder)

    def twin_gate(self):
        from r2p.real2sim.active import score_observations
        folder = self.folder("twin_gate")
        metrics = score_observations([o for o in self.observations if o.split == "heldout"], self.physics, self.scene,
                                     previews=folder / "heldout")
        passed, reasons = gate_metrics(metrics, self.campaign["before"])
        if passed:
            # Task sampling copies object/table materials, which override physics defaults.
            # Carry the accepted scene's values into the run-local game/training task.
            self.task["table"] = {**self.task.get("table", {}), **deepcopy(self.scene.get("table", {}))}
            objects = {o["id"]: o for o in self.scene["objects"]}
            for obj in self.task["objects"]:
                candidate = objects.get(obj["id"], {})
                for key in ("friction", "mass_kg", "size_m", "color"):
                    if key in candidate:
                        obj[key] = deepcopy(candidate[key])
                if not obj.get("fixed"):
                    rule = self.task.setdefault("randomize", {}).setdefault(obj["id"], {})
                    for key in ("friction", "mass_kg"):
                        if key in candidate:
                            rule[key] = candidate[key]
        self.frozen = {"task": digest(self.task), "physics": digest(self.physics), "scene": digest(self.scene),
                       "camera": digest(self.camera.to_dict())}
        if not self.dry:
            import hashlib
            self.frozen_files = {name: hashlib.sha256((self.root / name).read_bytes()).hexdigest() for name in (
                "config/camera.json", "config/joint_map.json", "r2p/model.py", "r2p/frames.py", "r2p/sim/twin.py",
                "scripts/robot/run_real_policy.py")}
            self.frozen["files"] = self.frozen_files
        report = {"passed": passed, "held_out": True, "synthetic": self.dry, "metrics": metrics,
                  "reasons": reasons, "input_sha256": self.frozen}
        gemini.atomic_json(folder / "gate.json", report)
        gemini.atomic_json(folder / "physics.json", self.physics)
        gemini.atomic_json(folder / "task.json", self.task)
        return {"metrics": {"passed": int(passed), **{k: metrics[k] for k in ("joint_deg", "jaw_mm", "object_mm", "iou")}},
                "evidence": str(folder / "gate.json"), "reasons": reasons}

    def check_frozen(self):
        current = {"task": digest(self.task), "physics": digest(self.physics), "scene": digest(self.scene),
                   "camera": digest(self.camera.to_dict())}
        if not self.dry:
            import hashlib
            current["files"] = {name: hashlib.sha256((self.root / name).read_bytes()).hexdigest()
                                for name in self.frozen_files}
        if current != self.frozen:
            raise ValueError("Twin inputs changed since the held-out gate")

    @staticmethod
    def managed_record(report, dry):
        rec = report["record"]
        return {"agent_id": rec.get("interaction_id"), "agent": rec.get("agent"),
                "duration_s": rec.get("latency", 0), "tests_passed": int(report["passed"]), "synthetic": dry}

    def author_task(self):
        helper = managed_helpers(self.root)
        files = helper.task_sources()
        folder = self.folder("author_task")
        report = helper.run_stage("author_task", helper.task_prompt(self.task["instruction"], "Current measured bench", files),
                                  folder, dry_run=self.dry, goal=self.task["instruction"], source_files=files)
        return {"metrics": {"passed": int(report["passed"]), "files": len(report["files"])},
                "managed": self.managed_record(report, self.dry), "report": str(folder / "result.json")}

    def build_game(self, publish=False):
        self.check_frozen()
        helper = managed_helpers(self.root)
        folder = self.folder("build_game")
        gate_path = self.folders["twin_gate"] / "gate.json"
        import hashlib
        inputs = {"task": self.task, "physics": self.physics, "board": self.task.get("board", {}), "camera": self.camera.to_dict()}
        verification = {"schema": "r2p_twin_verification/1", "passed": True,
                        "input_sha256": {k: helper.digest(v) for k, v in inputs.items()},
                        "evidence": [{"path": str(gate_path), "sha256": hashlib.sha256(gate_path.read_bytes()).hexdigest()}]}
        context = helper.game_context(self.task, self.physics, self.camera.to_dict(), verification)
        files = helper.game_sources(context)
        report = helper.run_stage("build_game", helper.game_prompt(context, files), folder, dry_run=self.dry,
                                  context=context, source_files=files)
        metrics = {"passed": int(report["passed"]), "local_checks": len(report["verification"]["checks"]),
                   "parity_passed": int(report["verification"]["parity"]["pass"]), "published": 0}
        if publish and not self.dry:
            request = {"workspace": report["workspace"], "files": report["record"]["files"], "twin": self.frozen}
            gemini.atomic_json(self.out / "publish_request.json", request)
            receipt = self.wait_receipt(self.out / "publish_receipt.json", digest(request))
            if not str(receipt.get("url", "")).startswith("https://www.roblox.com/games/"):
                raise ValueError("Publication receipt requires a Roblox game URL")
            metrics["published"] = 1
        return {"metrics": metrics, "managed": self.managed_record(report, self.dry), "report": str(folder / "result.json")}

    def collect_play(self, count=1):
        from r2p.pipeline.game_player import collect_players
        from r2p.traces import read_trace
        folder = self.folder("collect_play")
        self.play = folder / "traces"
        if self.dry:
            paths = collect_players(self.task, self.play, n=count, physics=self.physics, seed=2026)
        else:
            source = Path(self.r.options["from_game"])
            self.play.mkdir(parents=True)
            paths = []
            for path in sorted(source.rglob("*.jsonl")):
                meta, ticks = read_trace(path)
                if meta.get("task_name") == self.task["name"] and ticks and not meta.get("synthetic"):
                    target = self.play / (digest(path.read_text(encoding="utf-8")) + ".jsonl")
                    shutil.copyfile(path, target)
                    paths.append(target)
            if not paths:
                raise ValueError("No completed real game traces; publish/connect the receiver and collect rounds")
        meta_rows = [read_trace(path)[0] for path in paths]
        return {"metrics": {"traces": len(paths), "human": sum(m.get("source") == "human" for m in meta_rows),
                            "headless": sum(bool(m.get("headless")) for m in meta_rows)}, "folder": str(self.play)}

    def train_plan(self):
        from r2p.orchestrator_managed import training_plan
        self.check_frozen()
        return training_plan(self, self.folder("train_plan"))

    def wait_receipt(self, path, request_hash):
        seconds = float(self.r.options.get("wait_seconds", 60))
        if not math.isfinite(seconds) or not 0 <= seconds <= 86400:
            raise ValueError("wait-seconds must be finite and within 0..86400")
        deadline = time.monotonic() + seconds
        while True:
            if path.is_file():
                receipt = read_json(path)
                if receipt.get("request_sha256") != request_hash:
                    raise ValueError("Receipt does not match this exact request")
                return receipt
            self.r.publish(stage="waiting", next=f"Owner receipt: {path}; request_sha256={request_hash}")
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Waiting for owner receipt {path}; no GPU/publish operation launched")
            time.sleep(min(1, max(0, deadline - time.monotonic())))

    def replay_and_train(self):
        from r2p.pipeline.game_loop import run_loop, ingest
        self.check_frozen()
        folder = self.folder("replay_and_train")
        if self.dry:
            manifest = run_loop(self.play, self.task, root=self.out, output=folder, physics=self.physics,
                                experts=self.training["experts"], steps=self.training["steps"],
                                eval_n=self.training["eval_n"], max_ticks=30, seed=2026)
            selected = "after" if manifest["gate"]["accepted"] else "before"
            self.export = folder / selected
            scores = next(row for row in manifest["curve"] if row["label"] == selected)
            return {"metrics": {"replayed": manifest["counts"]["relabeled"], "eval_n": manifest["evaluation"]["n"],
                                "clean": scores["clean"], "real": scores["real"], "improved": int(manifest["gate"]["accepted"])},
                    "export": str(self.export), "smoke": True, "transfer_qualified": False}
        from r2p.pipeline.collection import collect
        from r2p.pipeline.evaluation import make_plan, evaluate_pair
        from r2p.pipeline.gate import compare
        from r2p.tasks import sample_scene
        from r2p.policy import load
        import numpy as np
        replay = ingest(self.play, folder / "replayed", self.task, self.physics)
        if not replay["counts"]["relabeled"]:
            raise ValueError("No game traces passed physical replay")
        collect(self.task, self.training["experts"], folder / "expert", workers=1, physics=self.physics, seed=2026)
        task_file, physics_file = folder / "task.json", folder / "physics.json"
        gemini.atomic_json(task_file, self.task)
        gemini.atomic_json(physics_file, self.physics)
        import hashlib
        traces = sorted((folder / "replayed").glob("*.jsonl")) + sorted((folder / "expert").glob("*.jsonl"))
        request = {"schema": "r2p_train_request/1", "task": str(task_file), "physics": str(physics_file),
                   "twin": self.frozen, "training": self.training,
                   "traces": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in traces},
                   "instructions": "Lead only: rent landlord GPU, apply lease env and VRAM cap; train absolute 15x6 targets. "
                                   "Write train_receipt.json with request_sha256, export and incumbent paths; no activation."}
        gemini.atomic_json(self.out / "train_request.json", request)
        receipt = self.wait_receipt(self.out / "train_receipt.json", digest(request))
        export, incumbent = Path(receipt["export"]), Path(receipt["incumbent"])
        for path in (export, incumbent):
            config = read_json(path / "config.json")
            if config.get("task") != self.task["name"] or digest(config.get("task_spec")) != digest(self.task):
                raise ValueError("Export was trained on a different task revision")
            if config.get("action") != "absolute_q" or config.get("horizon") != 15:
                raise ValueError("Export action contract is wrong")
            if path == export and digest(config.get("physics")) != digest(self.physics):
                raise ValueError("Candidate export did not train against the frozen physics")
            load(path)
        plan = make_plan(self.task, sample_scene(self.task, np.random.default_rng(100000)), self.training["eval_n"], 200000)
        plan["physics"] = self.physics
        after, before = evaluate_pair(export, incumbent, self.task, plan)
        gate = compare(after, before, plan)
        gemini.atomic_json(folder / "evaluation.json", {"plan": plan, "after": after, "before": before, "gate": gate})
        self.export = export if gate["accepted"] else incumbent
        measured = after if gate["accepted"] else before
        scores = {noise: sum(row["success"] for row in rows) / len(rows) for noise, rows in measured["results"].items()}
        return {"metrics": {"replayed": replay["counts"]["relabeled"], "eval_n": self.training["eval_n"],
                            "clean": scores["clean"], "real": scores["real"], "improved": int(gate["accepted"])},
                "export": str(self.export), "request": str(self.out / "train_request.json")}

    def deploy_real(self, cell="r1c1"):
        from r2p.traces import read_trace
        self.check_frozen()
        folder = self.folder("deploy_real")
        folder.mkdir(parents=True)
        task_file = folder / "task.json"
        gemini.atomic_json(task_file, self.task)
        if self.dry:
            argv = [sys.executable, str(self.root / "scripts/robot/run_policy.py"), "--dry-run", "--policy", str(self.export),
                    "--task", str(task_file), "--record", str(folder / "attempt"), "--timeout", "2", "--frame-stride", "15"]
        else:
            setup = self.r.options.get("perception_setup")
            if not setup or not Path(setup).is_file():
                raise ValueError("Live policy deployment needs --perception-setup; no motion sent")
            argv = [sys.executable, str(self.root / "scripts/robot/run_real_policy.py"), "--real", "--transport", "armd",
                    "--export", str(self.export), "--cell", cell, "--supply-xy", *map(str, self.caps[0]),
                    "--bench-profile", str(self.r.options["bench_profile"]), "--perception-setup", str(setup),
                    "--record", str(folder / "attempt"), "--timeout", "60"]
        completed = subprocess.run(argv, cwd=self.root, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        (folder / "command.log").write_text(gemini._safe(completed.stdout + completed.stderr), encoding="utf-8")
        traces = list((folder / "attempt").rglob("trace.jsonl"))
        if len(traces) != 1 or completed.returncode not in (0, 1):
            raise RuntimeError("Robot runner did not record one complete attempt; see " + str(folder / "command.log"))
        meta, ticks = read_trace(traces[0])
        summary = meta.get("summary", {})
        return {"metrics": {"attempts": 1, "ticks": len(ticks), "success": int(bool(meta.get("success")))},
                "outcome": summary.get("outcome", meta.get("fail_reason")), "trace": str(traces[0]),
                "transfer_qualified": False if self.dry else bool(meta.get("success"))}
