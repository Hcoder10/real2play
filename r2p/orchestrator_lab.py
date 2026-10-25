"""Lab evidence -> game -> replay -> lead training receipt -> one bench turn.

No GPU launch, hardware connection or model request happens on import/creation.
The Lab workspace is read-only; downstream work has its own frozen snapshots.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

from r2p import gemini
from r2p.orchestrator import digest, read_json, WaitingForOwner
from r2p.orchestrator_tools import LocalTools, load_script, managed_helpers


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lab_revision(directory):
    """Match SimWorkspace.revision without constructing/mutating a workspace."""
    value = hashlib.sha256()
    for path in sorted(Path(directory).rglob("*")):
        if path.is_file() and "robot_assets" not in path.parts:
            value.update(path.relative_to(directory).as_posix().encode())
            value.update(path.read_bytes())
    return value.hexdigest()


class LabTools(LocalTools):
    def __init__(self, runner):
        super().__init__(runner)
        name = Path(runner.lab_run)
        self.lab = (name if name.is_absolute() else self.root / name if (self.root / name).is_dir()
                    else self.root / "runs/lab" / name).resolve()
        if not (self.lab / "sim/physics.json").is_file() or not (self.lab / "sim/scene.xml").is_file():
            raise ValueError("Lab run requires sim/physics.json and sim/scene.xml: " + str(self.lab))
        self.physics = read_json(self.lab / "sim/physics.json")
        from r2p.perception import CameraModel
        self.camera = CameraModel.load(self.root / "config/camera.json")
        self.frozen = None
        self.turn_started = False

    def action(self, name, args):
        if name == "twin_gate":
            return f"Read sealed Lab held-out evidence and freeze hashes from {self.lab}; no experiments"
        if name == "collect_play":
            return f"Snapshot human receiver traces; collect {args.get('count', 1)} labelled CPU headless players in Lab physics"
        if name == "train_plan":
            return "Write the policy trainer recipe; GPU work remains a lead action under a landlord lease"
        if name == "deploy_real":
            return "Argon reads current board and supply; execute ONE legal turn via a qualified policy or calibrated ttt_armd.place; log executor"
        return super().action(name, args)

    def bench_state(self):
        evidence = super().bench_state()
        evidence["lab"] = {"path": str(self.lab), "revision": lab_revision(self.lab / "sim"),
                           "heldout": read_json(self.lab / "heldout.json") if (self.lab / "heldout.json").is_file() else None,
                           "status": read_json(self.lab / "status.json") if (self.lab / "status.json").is_file() else None}
        return evidence

    def twin_gate(self):
        folder = self.folder("twin_gate")
        held_path = self.lab / "heldout.json"
        held = read_json(held_path) if held_path.is_file() else {}
        result = held.get("result") or {}
        status = read_json(self.lab / "status.json") if (self.lab / "status.json").is_file() else {}
        synthetic = held.get("synthetic") is True or status.get("real_mock") is True
        if synthetic and not self.dry:
            raise ValueError("Synthetic Lab evidence cannot authorize a live chain")
        revision = lab_revision(self.lab / "sim")
        passed = result.get("sealed") is True and result.get("improved") is True
        reasons = [] if passed else ["Lab needs result.sealed=true and result.improved=true"]
        if result.get("revision") and result["revision"] != revision:
            passed = False
            reasons.append("Lab scene/physics revision changed after held-out scoring")
        # Preserve the Lab numbers verbatim, including unavailable/null residuals.
        residuals = {str(p.relative_to(self.lab)): read_json(p)
                     for p in sorted((self.lab / "replays").glob("*/residuals.json"))}
        self.physics = read_json(self.lab / "sim/physics.json")
        sources = {str(p): file_hash(p) for p in sorted((self.lab / "sim").rglob("*")) if p.is_file()}
        if held_path.is_file():
            sources[str(held_path)] = file_hash(held_path)
        for path in (self.root / "config/camera.json", self.root / "config/joint_map.json",
                     self.root / "config/physics/so101_bench.json",
                     self.root / "tasks" / self.task["name"] / "task.json"):
            if path.is_file():
                sources[str(path)] = file_hash(path)
        snapshot = folder / "sim"
        shutil.copytree(self.lab / "sim", snapshot)
        self.physics_path = snapshot / "physics.json"
        self.frozen = {"lab_run": str(self.lab), "revision": revision, "task": digest(self.task),
                       "physics": file_hash(self.physics_path), "scene": file_hash(snapshot / "scene.xml"),
                       "sources": sources, "snapshot": str(snapshot),
                       "snapshot_files": {str(p): file_hash(p) for p in snapshot.rglob("*") if p.is_file()}}
        self.lab_passed = passed
        report = {"schema": "r2p_lab_chain_gate/1", "passed": passed, "held_out": True,
                  "lab_run": str(self.lab), "synthetic": self.dry or synthetic, "result": result,
                  "episodes": held.get("episodes", []), "residuals": residuals,
                  "input_sha256": self.frozen, "reasons": reasons}
        gemini.atomic_json(folder / "gate.json", report)
        gemini.atomic_json(folder / "task.json", self.task)
        metrics = {"passed": int(passed), "sealed": int(result.get("sealed") is True),
                   "improved": int(result.get("improved") is True), "episodes": len(held.get("episodes", []))}
        for split in ("baseline", "current"):
            for key, value in (result.get(split) or held.get(split) or {}).items():
                if type(value) in (int, float) and math.isfinite(value):
                    metrics[f"{split}_{key}"] = value
        self.check_frozen()
        return {"metrics": metrics, "evidence": str(folder / "gate.json"), "reasons": reasons,
                "lab_result": result, "frozen": self.frozen}

    def check_frozen(self):
        if not self.frozen or digest(self.task) != self.frozen["task"]:
            raise ValueError("Missing or changed frozen Lab/task inputs")
        if lab_revision(self.lab / "sim") != self.frozen["revision"]:
            raise ValueError("Lab workspace changed since the held-out gate")
        for name, expected in {**self.frozen["sources"], **self.frozen["snapshot_files"]}.items():
            if not Path(name).is_file() or file_hash(name) != expected:
                raise ValueError("Frozen Lab input changed: " + name)
        if digest(self.physics) != digest(read_json(self.physics_path)):
            raise ValueError("In-memory Lab physics changed")

    def build_game(self, publish=False):
        self.check_frozen()
        helper = managed_helpers(self.root)
        folder = self.folder("build_game")
        gate_path = self.folders["twin_gate"] / "gate.json"
        verification = None
        if self.lab_passed:
            inputs = {"task": self.task, "physics": self.physics, "board": self.task.get("board", {}),
                      "camera": self.camera.to_dict()}
            verification = {"schema": "r2p_twin_verification/1", "passed": True,
                "input_sha256": {key: helper.digest(value) for key, value in inputs.items()},
                "evidence": [{"path": str(gate_path), "sha256": file_hash(gate_path)}]}
        # Same verification and managed builder as scripts/lead/lab_to_game.py.
        context = helper.game_context(self.task, self.physics, self.camera.to_dict(), verification)
        files = helper.game_sources(context)
        report = helper.run_stage("build_game", helper.game_prompt(context, files), folder,
                                  dry_run=self.dry, context=context, source_files=files)
        metrics = {"passed": int(report["passed"]), "verified": int(self.lab_passed),
                   "provisional": int(not self.lab_passed), "local_checks": len(report["verification"]["checks"]),
                   "parity_passed": int(report["verification"]["parity"]["pass"]), "published": 0}
        if publish and not self.dry:
            if not self.lab_passed or not report["passed"]:
                raise ValueError("Provisional/failed game builds cannot request publication")
            request = {"workspace": report["workspace"], "files": report["record"]["files"], "twin": self.frozen}
            gemini.atomic_json(self.out / "publish_request.json", request)
            receipt = self.wait_receipt(self.out / "publish_receipt.json", digest(request))
            if not str(receipt.get("url", "")).startswith("https://www.roblox.com/games/"):
                raise ValueError("Publication receipt requires a Roblox game URL")
            metrics["published"] = 1
        self.check_frozen()
        return {"metrics": metrics, "managed": self.managed_record(report, self.dry),
                "label": ("SYNTHETIC Lab gate fixture" if self.dry and self.lab_passed else
                          "LAB VERIFIED" if self.lab_passed else "PROVISIONAL - Lab gate has not passed"),
                "report": str(folder / "result.json"), "published": bool(metrics["published"])}

    def collect_play(self, count=1):
        from r2p.pipeline.game_player import collect_players
        from r2p.traces import read_trace
        self.check_frozen()
        folder = self.folder("collect_play")
        self.play = folder / "traces"
        human = self.play / "human"
        human.mkdir(parents=True)
        source = Path(self.r.options.get("from_game") or self.root / "roblox/out/traces")
        source = source if source.is_absolute() else self.root / source
        copied, rejected, seen = [], [], set()
        # Dry mode never reclassifies real receiver files as synthetic evidence.
        for path in ([] if self.dry else sorted(source.rglob("*.jsonl"))):
            try:
                meta, ticks = read_trace(path)
                if meta.get("source") != "human" or any(meta.get(k) for k in ("synthetic", "headless", "test_only", "attract")):
                    raise ValueError("Not a real human receiver trace")
                if not ticks or meta.get("task_name") != self.task["name"]:
                    raise ValueError("Empty trace or different task")
                fingerprint = file_hash(path)
                if fingerprint in seen:
                    raise ValueError("Duplicate trace")
                seen.add(fingerprint)
                target = human / (fingerprint + ".jsonl")
                shutil.copyfile(path, target)
                copied.append(str(target))
            except (ValueError, KeyError, TypeError, OSError) as exc:
                rejected.append({"path": str(path), "reason": str(exc)})
        players = collect_players(self.task, self.play / "headless", n=count, seed=2026, physics=self.physics)
        report = {"human": copied, "headless": list(map(str, players)), "rejected": rejected,
                  "physics_path": str(self.physics_path), "physics_sha256": self.frozen["physics"],
                  "lab_scene_sha256": self.frozen["scene"],
                  "simulation": "L's Python SkillOperator port in TaskTwin using Lab physics.json and sampled task scenes"}
        gemini.atomic_json(folder / "sources.json", report)
        self.check_frozen()
        return {"metrics": {"traces": len(copied)+len(players), "human": len(copied), "headless": len(players),
                            "rejected": len(rejected)}, "folder": str(self.play), "sources": str(folder / "sources.json")}

    def train_plan(self):
        self.check_frozen()
        folder = self.folder("train_plan")
        self.training = {"experts": 1 if self.dry else 4, "steps": 2 if self.dry else 60000,
                         "eval_n": 1 if self.dry else 12, "device": "cpu" if self.dry else "lead-leased-GPU",
                         "action": "absolute_q", "horizon": 15, "hz": 30,
                         "recipe_source": "P2 committed trainer recipe: width512 blocks3 seeds0,1,2 no-augmentation"}
        gemini.atomic_json(folder / "training.json", self.training)
        return {"metrics": {"passed": 1, "experts": self.training["experts"], "steps": self.training["steps"]},
                "config": str(folder / "training.json"), "gpu_launched": False}

    def replay_and_train(self):
        from r2p.pipeline.game_loop import ingest
        self.check_frozen()
        if self.dry:
            result = super().replay_and_train()
            result["metrics"]["policy_qualified"] = 0
            return result
        if (self.out / "train_request.json").exists():
            raise WaitingForOwner("A training request already exists for this run; refusing to submit a replacement")
        from r2p.pipeline.collection import collect
        folder = self.folder("replay_and_train")
        replay = ingest(self.play, folder / "replayed", self.task, self.physics)
        gemini.atomic_json(folder / "replay.json", replay)
        if not replay["counts"]["relabeled"]:
            raise ValueError("No game traces passed replay through the Lab physics")
        collect(self.task, self.training["experts"], folder / "expert", workers=1, physics=self.physics, seed=2026)
        mixed = folder / "mixed"
        mixed.mkdir()
        traces = sorted((folder / "replayed").glob("*.jsonl")) + sorted((folder / "expert").glob("*.jsonl"))
        for path in traces:
            shutil.copyfile(path, mixed / (file_hash(path) + ".jsonl"))
        task_path = folder / "task.json"
        gemini.atomic_json(task_path, self.task)
        name = self.out.name + "-lab-policy"
        # Argument vector, not shell text; no lease id is borrowed from another job.
        recipe = ["python", "-m", "r2p.policy.train", "--task", str(task_path), "--traces", str(mixed),
                  "--name", name, "--out-dir", str(self.out / "export"), "--steps", "60000", "--batch-size", "256",
                  "--seeds", "0", "1", "2", "--width", "512", "--blocks", "3", "--device", "cuda:0",
                  "--no-augmentation", "--eval-n", "30"]
        request = {"schema": "r2p_train_request/2", "name": name, "task": str(task_path),
                   "physics": str(self.physics_path), "physics_path": str(self.physics_path),
                   "scene_path": str(Path(self.frozen["snapshot"]) / "scene.xml"),
                   "trace_dirs": [str(folder / "replayed"), str(folder / "expert")], "mixed_dir": str(mixed),
                   "recipe": recipe, "twin": self.frozen, "training": self.training,
                   "traces": {str(p): file_hash(p) for p in mixed.glob("*.jsonl")},
                   "instructions": "Lead only: copy inputs to rig and rebase paths, rent a landlord GPU, apply returned env "
                       "and torch memory fraction (LANDLORD_MEMORY_FRACTION or --memory-fraction). "
                       "Write train_receipt.json with request_sha256, export and optional incumbent paths accessible here. "
                       "No activation. Without an incumbent comparison the calibrated IK executor remains selected."}
        gemini.atomic_json(self.out / "train_request.json", request)
        self.r.event("training_request", request=str(self.out / "train_request.json"), request_sha256=digest(request), gpu_launched=False)
        receipt = self.wait_receipt(self.out / "train_receipt.json", digest(request))
        self.check_frozen()
        for path, fingerprint in request["traces"].items():
            if file_hash(path) != fingerprint:
                raise ValueError("Training data changed while waiting for receipt")
        measured = self.evaluate_receipt(receipt, folder)
        return {"metrics": {"replayed": replay["counts"]["relabeled"], **measured},
                "export": str(self.export), "request": str(self.out / "train_request.json"),
                "evaluation": str(folder / "evaluation.json")}

    def evaluate_receipt(self, receipt, folder):
        import numpy as np
        from r2p.policy import load
        from r2p.pipeline.evaluation import make_plan, evaluate_pair, rollout
        from r2p.pipeline.game_commands import GameCommands
        from r2p.pipeline.gate import compare
        from r2p.tasks import sample_scene
        from r2p.traces import read_trace
        self.export = Path(receipt["export"])
        if not self.export.is_absolute():
            self.export = self.root / self.export
        incumbent = Path(receipt["incumbent"]) if receipt.get("incumbent") else None
        if incumbent and not incumbent.is_absolute():
            incumbent = self.root / incumbent
        export_hashes = {str(p): file_hash(p) for p in self.export.rglob("*") if p.is_file()}
        for path in (self.export, incumbent):
            if path is None:
                continue
            policy = load(path)
            config = policy.config
            if config.get("task") != self.task["name"] or digest(config.get("task_spec")) != digest(self.task):
                raise ValueError("Receipt export task revision does not match Lab chain")
            if config.get("action") != "absolute_q" or config.get("horizon") != 15:
                raise ValueError("Receipt export must emit absolute 15x6 targets")
            if config.get("hz") != 30 or config.get("obs_frame") != "spec_footprint":
                raise ValueError("Receipt export must use the 30 Hz footprint-frame contract")
            for key in ("start_q", "obs_layout", "obs_const", "eval"):
                if key not in config:
                    raise ValueError("Receipt export lacks " + key)
            start_q = np.asarray(config["start_q"], dtype=float)
            if start_q.shape != (6,) or not np.isfinite(start_q).all() or not config["eval"]:
                raise ValueError("Receipt export needs a finite start_q and recorded evaluation")
            if path == self.export and digest(config.get("physics")) != digest(self.physics):
                raise ValueError("Receipt export physics does not match Lab chain")
        plan = make_plan(self.task, sample_scene(self.task, np.random.default_rng(100000)), self.training["eval_n"], 200000)
        plan["physics"] = self.physics
        plan["training_scene_hashes"] = sorted({digest(read_trace(path)[0]["scene"])
                                                 for path in (folder / "mixed").glob("*.jsonl")})
        if any(digest(row["scene"]) in plan["training_scene_hashes"] for row in plan["episodes"]):
            raise ValueError("Held-out evaluation scene overlaps training data")
        def evaluate(*args):
            return rollout(*args, tracker_factory=GameCommands)
        after, before = evaluate_pair(self.export, incumbent or self.export, self.task, plan, evaluator=evaluate)
        if {str(p): file_hash(p) for p in self.export.rglob("*") if p.is_file()} != export_hashes:
            raise ValueError("Receipt export changed during evaluation")
        gate = compare(after, before, plan) if incumbent else {
            "accepted": False, "decision": "HOLD", "reason": "No incumbent supplied for paired improvement gate"}
        scores = {noise: sum(r["success"] for r in rows)/len(rows) for noise, rows in after["results"].items()}
        improved = bool(incumbent and gate["accepted"])
        config = read_json(self.export / "config.json")
        qualified = improved and min(scores.values()) >= .8 and self.training["eval_n"] >= 12 and not config.get("smoke_only")
        gemini.atomic_json(folder / "evaluation.json", {"plan": plan, "after": after, "before": before, "gate": gate,
            "incumbent_supplied": incumbent is not None, "policy_qualified": qualified,
            "executor": "learned_policy" if qualified else "calibrated_ik"})
        self.export_hashes = export_hashes
        return {"eval_n": self.training["eval_n"], "clean": scores["clean"], "real": scores["real"],
                "improved": int(improved), "policy_qualified": int(qualified)}

    def read_frame(self):
        """Bounded, read-only phone-camera capture; never call the unbounded ttt frame loop."""
        import cv2
        import numpy as np
        from urllib.request import urlopen
        with urlopen("http://127.0.0.1:8792/list", timeout=5) as response:
            age = float(json.load(response)["0"]["age_s"])
        if not math.isfinite(age) or not 0 <= age < 2:
            raise ValueError("Camera stale; no placement")
        with urlopen("http://127.0.0.1:8792/frame/0.jpg", timeout=5) as response:
            frame = cv2.imdecode(np.frombuffer(response.read(8*1024*1024), np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("No camera image; no placement")
        return frame

    def supply_caps(self, frame, client):
        import numpy as np
        script = load_script("ttt_armd", self.root)
        text, record = client.call([gemini.image_part(frame),
            "Find every white plastic bottle cap, open side up. Give TOP rim centre [y,x] normalized 0..1000 "
            "as rim_center and on_grid boolean. Supply caps are off the paper grid."],
            role="main", tag="ttt:caps", schema=script.CAP_SCHEMA, thinking="high")
        caps = []
        h, w = frame.shape[:2]
        camera = self.camera if self.camera.image_size == (w, h) else self.camera.resized((w, h))
        for cap in json.loads(text)["caps"]:
            point = cap["rim_center"]
            if cap["on_grid"] or len(point) != 2 or any(type(v) not in (int, float) or not 0 <= v <= 1000 for v in point):
                continue
            xy = camera.deproject(np.array([[point[1]*w/1000, point[0]*h/1000]]), h=.017)[0][:2]
            edge = min(p[1] for p in self.task["board"]["cells"].values())-self.task["board"]["cell_m"]/2-.008
            if .09 < xy[0] < .25 and -.17 < xy[1] < edge:
                caps.append(xy.round(4).tolist())
        if not caps:
            raise ValueError("Argon found no cap in the supply lane; no motion")
        self.r.event("supply_observation", caps=caps, record=record)
        return sorted(caps, key=lambda p: (p[0] < .11, np.hypot(*p)))

    def deploy_real(self, cell=None):
        self.check_frozen()
        if self.dry:
            result = super().deploy_real(cell=cell or "r1c1")
            result.update(executor="mock_policy", transfer_qualified=False)
            self.r.event("deployment", executor="mock_policy", motion=False, result=result)
            return result
        if not self.lab_passed:
            raise ValueError("Real deployment requires a sealed, improved Lab gate")
        if self.turn_started or (self.out / "deployment_started.json").exists():
            raise WaitingForOwner("One turn already started; review the recorded outcome before any new run")
        from r2p import ttt
        from r2p.orchestrator_client import RobustGemini
        client = self.r.controller if isinstance(self.r.controller, RobustGemini) else RobustGemini(
            self.out / "board_cache", max_requests=8, retries=0, timeout_s=90)
        board_path = Path(self.r.options.get("board") or self.root / "out/ttt/status.json")
        if not board_path.is_absolute():
            board_path = self.root / board_path
        history = read_json(board_path) if board_path.is_file() else {"board": [""]*9}
        decision = ttt.decide(self.read_frame(), self.camera, history, "O", "white plastic bottle caps, open side up",
                              "flat metallic coins", ask=client.call)
        if decision.get("source") != "gemini" or decision.get("move") is None:
            raise WaitingForOwner("No current legal Argon robot turn: " + str(decision.get("why")))
        # A model-supplied cell argument never overrides the freshly observed board decision.
        cell = decision["move"]
        folder = self.folder("deploy_real")
        folder.mkdir(parents=True)
        gemini.atomic_json(folder / "decision.json", decision)
        gemini.atomic_json(folder / "board.json", {"board": decision["board"]})
        self.caps = self.supply_caps(self.read_frame(), client)
        metrics = self.r.state["results"]["replay_and_train"]["metrics"]
        qualified = metrics.get("policy_qualified") == 1
        self.check_frozen()
        preflight = {"passed": False, "reason": "Policy did not pass the paired twin evaluation gate"}
        if qualified:
            for name, expected in self.export_hashes.items():
                if file_hash(name) != expected:
                    raise ValueError("Evaluated export changed before deployment")
            preflight = self.policy_preflight(cell, folder)
        executor = "learned_policy" if qualified and preflight["passed"] else "calibrated_ik"
        action = {"executor": executor, "cell": cell, "cap_xy": self.caps[0],
                  "cell_xy": self.task["board"]["cells"][cell], "decision": decision,
                  "policy_twin_qualified": qualified, "policy_preflight": preflight,
                  "authorization": self.r.approve, "time": time.time()}
        self.r.event("deployment", **action)
        self.turn_started = True
        gemini.atomic_json(self.out / "deployment_started.json", action)
        if executor == "learned_policy":
            # Reuse K's guarded runner, including its independent map/start/perception checks.
            self.r.options["deployment_board"] = folder / "board.json"
            result = super().deploy_real(cell=cell)
            result["executor"] = executor
        else:
            script = load_script("ttt_armd", self.root)
            attempt = script.place(self.caps[0], self.task["board"]["cells"][cell], cell)
            # Stall detection is not proof of a placement. Keep outcome explicitly unverified.
            result = {"metrics": {"attempts": 1, "success": 0, "success_verified": 0},
                      "executor": executor, "attempt": attempt, "transfer_qualified": False,
                      "outcome": "calibrated IK turn completed; task success not independently measured"}
        gemini.atomic_json(folder / "outcome.json", result)
        self.r.event("deployment_result", result=result)
        return result

    def policy_preflight(self, cell, folder):
        """K's no-motion audit can select IK BEFORE any learned attempt starts."""
        setup = self.r.options.get("perception_setup")
        if not setup or not Path(setup).is_file():
            return {"passed": False, "reason": "No saved policy perception setup supplied"}
        # The runner's compensation must use the same physics as the evaluated
        # policy. Do not overwrite the bench's calibration to force a handover.
        if digest(read_json(self.root / "config/physics/so101_bench.json")) != digest(self.physics):
            return {"passed": False, "reason": "Bench compensation physics differs from evaluated Lab physics"}
        from r2p.model import JointMap
        if not JointMap.load(self.root / "config/joint_map.json").verified:
            return {"passed": False, "reason": "Policy runner requires a verified joint map"}
        command = [sys.executable, str(self.root / "scripts/robot/run_real_policy.py"), "--check",
                   "--export", str(self.export), "--cell", cell, "--supply-xy", *map(str, self.caps[0]),
                   "--board", str(folder / "board.json"), "--bench-profile",
                   str(self.r.options.get("bench_profile") or self.root / "scripts/robot/K_STREAM_PROFILE.json")]
        try:
            completed = subprocess.run(command, cwd=self.root, capture_output=True, text=True,
                                       encoding="utf-8", errors="replace", timeout=60)
            log = folder / "policy_preflight.log"
            log.write_text(gemini._safe(completed.stdout+completed.stderr), encoding="utf-8")
            return {"passed": completed.returncode == 0, "motion": False, "log": str(log),
                    "reason": "K runner audit passed" if completed.returncode == 0 else "K runner audit refused; see log"}
        except (OSError, subprocess.SubprocessError) as exc:
            return {"passed": False, "motion": False, "reason": gemini._safe(exc)}
