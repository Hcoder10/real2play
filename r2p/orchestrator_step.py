"""Argon step experiments -> managed game -> B200 learning -> one bench turn."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from r2p import gemini
from r2p.orchestrator import WaitingForOwner, digest, read_json
from r2p.orchestrator_remote import B200, FakeB200, sha
from r2p.orchestrator_tools import LocalTools, load_script


def physics_from_sim(base, sim, name):
    """after_step's conversion, with finite numeric patches and explicit provenance."""
    physics = deepcopy(base)
    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)
    for key, patch in (sim.get("physics") or {}).items():
        if key in physics.get("joints", {}) and isinstance(patch, dict):
            physics["joints"][key].update({k: v for k, v in patch.items() if finite(v)})
        elif key in physics and finite(patch):
            physics[key] = patch
        elif key in physics and isinstance(patch, dict) and isinstance(physics[key], dict):
            physics[key].update({k: v for k, v in patch.items() if finite(v)})
    caps = [o for o in sim.get("objects", []) if "cap" in str(o.get("id", "")).lower()]
    for key in ("mass_kg", "friction"):
        if caps and finite(caps[0].get(key)) and caps[0][key] > 0:
            physics.setdefault("object", {})[key] = caps[0][key]
    physics["notes"] = f"Argon step-loop candidate ({name}); measured arm offsets: {json.dumps(sim.get('arm', {}))[:400]}"
    return physics


def snapshot_humans(source, destination, task, *, dry=False):
    from r2p.traces import read_trace
    destination.mkdir(parents=True, exist_ok=True)
    accepted, rejected, seen = [], [], set()
    for path in ([] if dry else sorted(Path(source).rglob("*.jsonl"))):
        try:
            meta, ticks = read_trace(path)
            if meta.get("source") != "human" or any(meta.get(k) for k in ("synthetic", "headless", "test_only", "attract")):
                raise ValueError("Not a real human receiver trace")
            if any(p.lower() in {"fixtures", "test", "tests"} for p in path.parts):
                raise ValueError("Test fixtures are not receiver evidence")
            if not ticks or meta.get("task_name") != task["name"]:
                raise ValueError("Empty or wrong-task trace")
            fingerprint = sha(path)
            if fingerprint in seen:
                raise ValueError("Duplicate human trace")
            seen.add(fingerprint)
            target = destination / (fingerprint + ".jsonl")
            shutil.copyfile(path, target)
            accepted.append({"source": str(path), "snapshot": str(target), "sha256": fingerprint})
        except (ValueError, KeyError, TypeError, OSError) as exc:
            rejected.append({"source": str(path), "reason": str(exc)})
    return {"accepted": accepted, "rejected": rejected, "human": len(accepted)}


class StepTools(LocalTools):
    def __init__(self, runner):
        super().__init__(runner)
        if self.task["name"] != "place_on_cell":
            raise ValueError("The step-run bench chain currently supports place_on_cell")
        for key, default, maximum in (("headless", 32, 10000), ("experts", 256, 10000), ("experiment_turns", 40, 100)):
            value = self.r.options.get(key, default)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{key} must be 1..{maximum}")
        self.inputs = self.out / "inputs"
        self.inputs.mkdir()
        self.remote = FakeB200(runner) if self.dry else B200(runner)
        self.frozen = {}
        self.export = None

    def action(self, name, args):
        return {
            "experiment_sim": "Read existing step sim/ledger, or run scripts/lead/argon_step.py for guarded two-camera experiments",
            "build_game": "Convert Argon's physics; run Antigravity managed game builder and local parity checks; provisional candidate",
            "collect_play": "Snapshot real human receiver traces; Git-bundle sync and headless play on rented B200",
            "train": "Replay and collect experts on B200; train 60000 steps, batch256, seeds0/1/2, width512 blocks3, no augmentation, eval30; copy artifacts home",
            "evaluate": "B200 no-motion rehearsal of all nine cells with captured real map/daemon limits and recorded twin evaluations",
            "deploy": "ONE real turn: Argon chooses a cell, rehearsal-qualified policy via armd or calibrated ttt_armd IK; never retry",
        }.get(name, name)

    def command(self, argv, log, *, timeout=600, check=True):
        self.r.event("command", argv=list(map(str, argv)), log=str(log), motion="--real" in argv)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(list(map(str, argv)), cwd=self.root, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
        if check and result.returncode:
            raise RuntimeError(f"Command exited {result.returncode}; inspect {log}")
        return result.returncode

    def seal(self):
        self.frozen = {str(p): sha(p) for p in self.inputs.rglob("*") if p.is_file()}
        gemini.atomic_json(self.out / "frozen.json", self.frozen)

    def check_frozen(self):
        if not self.frozen:
            raise ValueError("No frozen experiment inputs")
        for path, expected in self.frozen.items():
            if not Path(path).is_file() or sha(path) != expected:
                raise ValueError("Frozen step input changed: " + path)

    def experiment_sim(self):
        if (self.out / "experiment_started.json").exists():
            raise WaitingForOwner("Experiment already started; do not repeat possibly completed physical moves")
        name = self.r.step_run
        folder = self.folder("experiment_sim")
        folder.mkdir()
        if self.dry:
            sim = {"objects": [], "arm": {"xy_offset_mm": [10, 11], "notes": "Synthetic smoke fixture"},
                   "physics": {}, "board": self.task.get("board", {}), "synthetic": True}
            gemini.atomic_json(self.inputs / "sim.json", sim)
            (self.inputs / "step_ledger.jsonl").write_text(json.dumps({"synthetic": True, "fake_argon": True}) + "\n", encoding="utf-8")
            source = "synthetic fixture; no step experiment executed"
        else:
            if name == "new":
                name = self.out.name
                gemini.atomic_json(self.out / "experiment_started.json", {"run": name, "time": time.time()})
                self.command([sys.executable, "scripts/lead/argon_step.py", "--run", name, "--turns",
                              str(self.r.options.get("experiment_turns", 40))], folder / "experiment.log", timeout=14400)
            candidate = Path(name)
            source = candidate if candidate.is_absolute() else self.root / candidate
            if not source.is_dir():
                source = self.root / "runs/argon_step" / candidate
            for original, target in (("sim.json", "sim.json"), ("ledger.jsonl", "step_ledger.jsonl")):
                shutil.copyfile(source / original, self.inputs / target)
            sim = read_json(self.inputs / "sim.json")
            if sim.get("synthetic") or not (self.inputs / "step_ledger.jsonl").stat().st_size:
                raise ValueError("Live step chain requires real, nonempty experiment evidence")
        self.physics = physics_from_sim(self.physics, sim, name)
        self.physics_path = self.inputs / "physics.json"
        gemini.atomic_json(self.physics_path, self.physics)
        gemini.atomic_json(self.inputs / "task.json", self.task)
        self.seal()
        return {"metrics": {"sim_ready": 1, "offset_fields": len(sim.get("arm", {})), "verified": 0},
                "sim": str(self.inputs / "sim.json"), "ledger": str(self.inputs / "step_ledger.jsonl"),
                "source": str(source), "measured_offsets": sim.get("arm", {}),
                "label": "PROVISIONAL: step observations are not a held-out fidelity gate"}

    def build_game(self, publish=False):
        self.check_frozen()
        if publish:
            raise ValueError("Step chain produces a provisional candidate; publication needs separate verified evidence")
        folder = self.folder("build_game")
        argv = [sys.executable, "scripts/argon/build_game.py", "--task", str(self.inputs / "task.json"),
                "--physics", str(self.physics_path), "--out", str(folder), "--allow-provisional-twin"]
        if self.dry:
            argv.append("--dry-run")
        self.command(argv, self.out / "build_game.log", timeout=3600)
        report = read_json(folder / "result.json")
        if not report.get("passed"):
            raise ValueError("Managed game build/parity failed")
        self.check_frozen()
        return {"metrics": {"passed": 1, "provisional": 1, "published": 0},
                "label": "PROVISIONAL - no held-out twin fidelity pass", "report": str(folder / "result.json"),
                "managed": self.managed_record(report, self.dry)}

    def collect_play(self, count=None):
        from r2p.robot.config import load_config, config_path
        self.check_frozen()
        source = Path(self.r.options.get("from_game") or "roblox/out/traces")
        source = source if source.is_absolute() else self.root / source
        humans = snapshot_humans(source, self.inputs / "human", self.task, dry=self.dry)
        gemini.atomic_json(self.out / "human_sources.json", humans)
        profile_path = Path(self.r.options.get("bench_profile") or "scripts/robot/K_STREAM_PROFILE.json")
        profile_path = profile_path if profile_path.is_absolute() else self.root / profile_path
        profile = read_json(profile_path)
        cfg = load_config(self.root / "config/bench.json")
        self.bench_config_hash = digest(cfg)
        map_path = config_path(cfg.get("joint_map", "config/joint_map.json"))
        shutil.copyfile(map_path, self.inputs / "joint_map.json")
        gemini.atomic_json(self.inputs / "bench.json", cfg)
        gemini.atomic_json(self.inputs / "profile.json", profile)
        # Rehearsal uses the real bench physics; training uses Argon's candidate.
        rig = cfg.get("rig", "so101_bench")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", rig):
            raise ValueError("Unsafe bench rig identifier")
        shutil.copyfile(self.root / f"config/physics/{rig}.json", self.inputs / "bench_physics.json")
        log = subprocess.run(["git", "log", "-1", "--format=%H %s", "--", "r2p/sim/expert.py"],
                             cwd=self.root, capture_output=True, text=True, check=True).stdout.strip()
        # The earlier e8cd98b added the profile interface but its expert stalled.
        # Only a later committed expert fix may enable the P3 bench recipe.
        history = subprocess.run(["git", "log", "e8cd98b..HEAD", "--format=%H %s", "--", "r2p/sim/expert.py"],
                                 cwd=self.root, capture_output=True, text=True, check=True).stdout.strip()
        request = {"schema": "r2p_step_b200/1", "name": self.out.name + "-policy", "task": self.task,
                   "physics": self.physics, "profile": profile, "cap": self.r.options.get("cap", 0),
                   "headless": count or self.r.options.get("headless", 32), "experts": self.r.options.get("experts", 256),
                   "bench_expert_committed": bool(history), "expert_commit": log,
                   "synthetic": self.dry, "step_sim_sha256": sha(self.inputs / "sim.json")}
        gemini.atomic_json(self.inputs / "request.json", request)
        self.request = request
        self.live_inputs = {str(profile_path): sha(profile_path), str(map_path): sha(map_path),
                            str(self.root / "tasks/place_on_cell/task.json"): sha(self.root / "tasks/place_on_cell/task.json"),
                            str(self.root / f"config/physics/{rig}.json"): sha(self.root / f"config/physics/{rig}.json")}
        self.seal()
        self.remote.sync(self.inputs)
        report = self.remote.run("collect_play")
        return {"metrics": {k: report[k] for k in ("traces", "human", "headless")},
                "rejected": humans["rejected"], "executor": "fake_ssh" if self.dry else "B200",
                "expert_mode": "P3 bench-profile" if history else "default; P3 fix not committed", "expert_commit": log}

    def train(self):
        self.check_frozen()
        report = self.remote.run("train")
        source = self.out / "remote/export"
        config = read_json(source / "config.json")
        self.validate_export(config)
        if not self.dry and sha(source / "weights.npz") != config.get("weights_sha256"):
            raise ValueError("Returned policy weights do not match their export hash")
        self.export = self.root / "exports" / self.request["name"]
        shutil.copytree(source, self.export)  # Never overwrite an earlier export.
        self.export_hashes = {str(p): sha(p) for p in self.export.rglob("*") if p.is_file()}
        self.r.event("export_returned", path=str(self.export), files=self.export_hashes)
        return {"metrics": {"replayed": report["replayed"], "experts": report["experts"], "exported": 1},
                "export": str(self.export), "expert_mode": report["expert_mode"], "recipe": report["recipe"]}

    def validate_export(self, config):
        if config.get("task") != self.task["name"]:
            raise ValueError("Export has the wrong task")
        if digest(config.get("task_spec")) != digest(self.task) or digest(config.get("physics")) != digest(self.physics):
            raise ValueError("Returned export differs from frozen task/physics")
        for key, value in {"action": "absolute_q", "horizon": 15, "hz": 30, "obs_frame": "spec_footprint"}.items():
            if config.get(key) != value:
                raise ValueError("Export violates " + key)
        for key in ("start_q", "obs_layout", "obs_const", "eval"):
            if key not in config:
                raise ValueError("Export lacks " + key)
        q = config["start_q"]
        if not isinstance(q, list) or len(q) != 6 or any(type(v) not in (int, float) or not math.isfinite(v) for v in q):
            raise ValueError("Invalid exported start_q")
        if not config["eval"]:
            raise ValueError("Export has no recorded evaluation")
        if not self.dry:
            if config.get("synthetic") or config.get("smoke_only"):
                raise ValueError("Synthetic/smoke export is not a live training result")
            for noise in ("clean", "real", "realx0.5"):
                score = config["eval"].get(noise) or {}
                if score.get("n", 0) < 30 or type(score.get("success")) not in (int, float) or not 0 <= score["success"] <= 1:
                    raise ValueError("Export needs 30 recorded evaluation scenes for " + noise)
                if score.get("weights_sha256") != config.get("weights_sha256"):
                    raise ValueError("Recorded evaluation weights differ from the export")

    def evaluate(self):
        self.check_frozen()
        report = self.remote.run("evaluate")
        if not self.dry:
            actual = {p.relative_to(self.export).as_posix(): sha(p) for p in self.export.rglob("*") if p.is_file()}
            if actual != report.get("export_hashes"):
                raise ValueError("Rehearsed export differs from the export returned home")
        self.evaluation = report
        return {"metrics": {k: int(report[k]) if isinstance(report[k], bool) else report[k]
                            for k in ("eval_n", "clean", "real", "rehearsal_n", "rehearsal_goals", "policy_qualified")},
                "reason": report["reason"], "report": str(self.out / "remote/evaluate.json")}

    def deploy(self):
        self.check_frozen()
        marker = self.out / "deployment_started.json"
        if marker.exists():
            raise WaitingForOwner("One turn already started; never retry or fall back after possible motion")
        for path, expected in self.export_hashes.items():
            if not Path(path).is_file() or sha(path) != expected:
                raise ValueError("Evaluated export changed before deployment")
        if not self.dry:
            from r2p.robot.config import load_config
            if digest(load_config(self.root / "config/bench.json")) != self.bench_config_hash:
                raise ValueError("Bench runner configuration changed since rehearsal")
            for path, expected in self.live_inputs.items():
                if sha(path) != expected:
                    raise ValueError("Bench map/task/profile/physics changed since rehearsal: " + path)
        qualified = bool(self.evaluation["policy_qualified"])
        executor = "learned_policy" if qualified else "calibrated_ik"
        action = {"executor": executor, "reason": self.evaluation["reason"], "export": str(self.export),
                  "cap": self.request["cap"], "authorization": self.r.approve, "synthetic": self.dry}
        gemini.atomic_json(marker, action)
        self.r.event("deployment", **action)
        folder = self.folder("deploy")
        folder.mkdir()
        if self.dry:
            from r2p.robot.io import MockRobot
            robot = MockRobot()
            try:
                # Exercise the actual mock transport; a mock turn is not task success.
                for _ in range(30):
                    robot.write(robot.read())
                    robot.step()
                result = {"metrics": {"attempts": 1, "ticks": 30, "success": 0, "success_verified": 0},
                          "executor": executor, "transport": "MockRobot", "outcome": "synthetic turn completed"}
            finally:
                robot.close()
        elif qualified:
            argv = [sys.executable, "scripts/robot/run_real_policy.py", "--export", str(self.export),
                    "--argon-perception", "--transport", "armd", "--bench-profile", str(self.inputs / "profile.json"),
                    "--cap", str(self.request["cap"]), "--real", "--allow-unverified", "--record", str(folder / "attempt")]
            board = self.r.options.get("board")
            if board:
                argv += ["--board", str(board)]
            rc = self.command(argv, folder / "policy.log", timeout=300, check=False)
            from r2p.traces import read_trace
            traces = list((folder / "attempt").rglob("trace.jsonl"))
            if len(traces) != 1:
                raise WaitingForOwner("Policy runner did not record exactly one attempt; no IK retry; inspect policy.log")
            meta, ticks = read_trace(traces[0])
            result = {"metrics": {"attempts": 1, "ticks": len(ticks), "success": int(bool(meta.get("success"))),
                                  "success_verified": int(bool(meta.get("success")))},
                      "executor": executor, "exit_code": rc, "trace": str(traces[0]),
                      "outcome": meta.get("summary", {}).get("outcome", meta.get("fail_reason"))}
        else:
            result = self.ik_turn(folder)
        gemini.atomic_json(folder / "outcome.json", result)
        self.r.event("deployment_result", result=result)
        return result

    def ik_turn(self, folder):
        from r2p import ttt
        from r2p.orchestrator_client import RobustGemini
        from r2p.orchestrator_lab import LabTools
        from r2p.perception import CameraModel
        client = self.r.controller if isinstance(self.r.controller, RobustGemini) else RobustGemini(
            self.out / "board_cache", max_requests=8, retries=0, timeout_s=90)
        self.camera = CameraModel.load(self.root / "config/camera.json")
        board_path = Path(self.r.options.get("board") or self.root / "out/ttt/status.json")
        history = read_json(board_path) if board_path.is_file() else {"board": [""] * 9}
        if isinstance(history, list):
            history = {"board": history}
        decision = ttt.decide(LabTools.read_frame(self), self.camera, history, "O", "white plastic bottle caps, open side up",
                              "flat metallic coins", ask=lambda *a, **kw: client.call(*a, **{**kw, "thinking": "low"}))
        if decision.get("source") != "gemini" or decision.get("move") is None:
            raise WaitingForOwner("Argon supplied no legal turn; no motion, no automatic retry")
        self.read_frame = lambda: LabTools.read_frame(self)
        from types import SimpleNamespace
        low_client = SimpleNamespace(call=lambda *a, **kw: client.call(*a, **{**kw, "thinking": "low"}))
        caps = LabTools.supply_caps(self, self.read_frame(), low_client)
        cell = decision["move"]
        script = load_script("ttt_armd", self.root)
        self.r.event("ik_turn", decision=decision, supply_xy=caps[0], cell=cell,
                     measured_held_offset_m=script.HELD_OFFSET, implementation="scripts/lead/ttt_armd.py")
        gemini.atomic_json(folder / "decision.json", decision)
        attempt = script.place(caps[0], self.task["board"]["cells"][cell], cell)
        return {"metrics": {"attempts": 1, "success": 0, "success_verified": 0}, "executor": "calibrated_ik",
                "attempt": attempt, "decision": decision, "outcome": "Calibrated IK turn completed; placement unverified"}

    def numeric_gates(self):
        results = self.r.state["results"]
        reasons = ["Missing successful " + name for name in self.r.order if not results.get(name, {}).get("ok")]
        if reasons:
            return False, reasons
        if results["build_game"]["metrics"].get("passed") != 1:
            reasons.append("Game parity/build did not pass")
        if results["collect_play"]["metrics"].get("traces", 0) < 1:
            reasons.append("No play traces")
        if results["train"]["metrics"].get("exported") != 1:
            reasons.append("No returned trained export")
        if results["train"]["metrics"].get("replayed", 0) < 1:
            reasons.append("No physically replayed play data")
        if results["deploy"]["metrics"].get("attempts") != 1:
            reasons.append("No completed turn")
        if results["deploy"].get("executor") not in ("learned_policy", "calibrated_ik"):
            reasons.append("Missing executor provenance")
        return not reasons, reasons
