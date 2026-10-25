"""Fixed B200 transport and workers for W's step chain. No work happens on import.

Code travels as a Git bundle into a new detached worktree, never a reset of the
shared checkout. Inputs and incremental result archives are independent of Git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import threading
import time
import zipfile

from r2p import gemini

LEASE = {
    "CUDA_VISIBLE_DEVICES": "0", "LANDLORD_LEASE_ID": "apt_8fc32d1f",
    "LANDLORD_BUILDING": "vast-57730383", "LANDLORD_VRAM_GB": "170.0",
    "LANDLORD_MAX_PCT": "100", "LANDLORD_MEMORY_FRACTION": "0.949",
    "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "R2P_CACHE_ONLY": "1",
}
# The supplied rental ends at 17:30 PDT. Do not silently launch on a stale lease.
RENTAL_END = 1792974600  # 2026-10-26 00:30:00 UTC


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unpack(archive, destination):
    """Validate every member before extracting any; remote artifacts cannot escape."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(destination) or "\\" in info.filename or ":" in info.filename:
                raise ValueError("Unsafe archive member: " + info.filename)
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive symlinks are forbidden")
        bundle.extractall(destination)


class B200:
    def __init__(self, runner):
        self.r, self.root, self.out = runner, runner.root, runner.out
        self.host = "root@ssh9.vast.ai"
        self.key = str(Path.home() / ".ssh/id_ed25519")
        self.ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-p", "32926", "-i", self.key, self.host]
        self.scp = ["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-P", "32926", "-i", self.key]
        self.remote = "/root/real2play-w/" + self.out.name
        self.ready = False
        self.last_snapshot = None

    def call(self, argv, *, timeout=120, check=True):
        result = subprocess.run(list(map(str, argv)), cwd=self.root, capture_output=True, text=True,
                                timeout=timeout, encoding="utf-8", errors="replace")
        if check and result.returncode:
            raise RuntimeError(gemini._safe(result.stderr or result.stdout)[-3000:])
        return result

    def remote_call(self, argv, **kwargs):
        return self.call([*self.ssh, shlex.join(list(map(str, argv)))], **kwargs)

    def sync(self, inputs):
        if self.ready:
            return
        if time.time() >= RENTAL_END:
            raise RuntimeError("The supplied B200 rental expired; obtain a current lease before launching")
        bundle = self.out / "source.bundle"
        commit = self.call(["git", "rev-parse", "HEAD"]).stdout.strip()
        # A committed orchestrator is required: bundle-only transfer cannot carry dirty source.
        for name in ("r2p/orchestrator.py", "r2p/orchestrator_step.py", "r2p/orchestrator_remote.py",
                     "r2p/orchestrator_client.py"):
            tracked = self.call(["git", "show", f"HEAD:{name}"], check=False)
            if tracked.returncode or self.call(["git", "diff", "HEAD", "--", name]).stdout:
                raise ValueError("Commit W's source before live bundle sync: " + name)
        self.call(["git", "bundle", "create", str(bundle), "HEAD"], timeout=180)
        archive = self.out / "inputs.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for path in sorted(Path(inputs).rglob("*")):
                if path.is_file():
                    z.write(path, "w_inputs/" + path.relative_to(inputs).as_posix())
        self.remote_call(["mkdir", "-p", "/root/real2play-w"])
        remote_bundle = self.remote + ".bundle"
        remote_inputs = self.remote + ".inputs.zip"
        self.call([*self.scp, bundle, self.host + ":" + remote_bundle], timeout=180)
        self.call([*self.scp, archive, self.host + ":" + remote_inputs], timeout=180)
        self.remote_call(["git", "-C", "/root/real2play", "fetch", remote_bundle, "HEAD"], timeout=180)
        self.remote_call(["git", "-C", "/root/real2play", "worktree", "add", "--detach", self.remote, commit], timeout=180)
        self.remote_call(["python3", "-m", "zipfile", "-e", remote_inputs, self.remote])
        self.ready = True
        self.r.event("b200_sync", commit=commit, bundle_sha256=sha(bundle), inputs_sha256=sha(archive),
                     remote=self.remote, lease=LEASE, rental_end=RENTAL_END)

    def pull(self, *, required=False):
        stamp = self.remote_call(["stat", "-c", "%Y:%s", self.remote + "/w_results/artifacts.zip"], check=False)
        if stamp.returncode:
            if required:
                raise RuntimeError("B200 did not produce an artifact snapshot")
            return
        if stamp.stdout == self.last_snapshot and not required:
            return
        archive = self.out / "b200-artifacts.zip"
        self.call([*self.scp, self.host + ":" + self.remote + "/w_results/artifacts.zip", archive], timeout=180)
        unpack(archive, self.out / "remote")
        self.last_snapshot = stamp.stdout
        self.r.event("b200_results_saved", archive=str(archive), sha256=sha(archive), timestamp=stamp.stdout.strip())

    def run(self, stage):
        from r2p.orchestrator import WaitingForOwner
        marker = self.out / ("b200_" + stage + "_started.json")
        if marker.exists():
            raise WaitingForOwner(f"B200 {stage} already launched; inspect its saved log/results before any new run")
        if not self.ready or time.time() >= RENTAL_END:
            raise RuntimeError("B200 is not synchronized or the supplied rental expired")
        # The fixed environment prefixes EVERY remote worker command, including CUDA training.
        command = ["env", *[f"{k}={v}" for k, v in LEASE.items()], "python3", "-u", "-m",
                   "r2p.orchestrator_remote", "--stage", stage, "--inputs", "w_inputs", "--out", "w_results"]
        script = "cd " + shlex.quote(self.remote) + " && " + shlex.join(command)
        gemini.atomic_json(marker, {"stage": stage, "command": command, "remote": self.remote, "time": time.time()})
        self.r.event("b200_launch", stage=stage, command=command, remote=self.remote)
        log_path = self.out / ("b200_" + stage + ".log")
        deadline = min(RENTAL_END - 30, time.time() + float(self.r.options.get("wait_seconds", 3600)))
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen([*self.ssh, script], cwd=self.root, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                try:
                    self.pull()
                except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                    self.r.event("b200_copy_retry", error=gemini._safe(exc))
                if time.time() >= deadline:
                    # Never kill a remote tenant on timeout or automatically duplicate its work.
                    raise WaitingForOwner(f"B200 {stage} remains unresolved; inspect {log_path}; no relaunch")
                print(f"B200 {stage}: running; results copied to {self.out / 'remote'}", flush=True)
                time.sleep(15)
        self.pull(required=True)
        report = read(self.out / "remote" / (stage + ".json"))
        if process.returncode or not report.get("ok"):
            raise WaitingForOwner(f"B200 {stage} failed; no automatic duplicate launch: {report.get('error', log_path)}")
        return report


class FakeB200:
    """Offline SSH seam: bounded artifacts, never imports a trainer or opens SSH."""
    def __init__(self, runner):
        self.r, self.out = runner, runner.out

    def sync(self, inputs):
        self.inputs = Path(inputs)
        self.r.event("fake_ssh_sync", inputs=str(inputs), synthetic=True)

    def run(self, stage):
        out = self.out / "remote"
        out.mkdir(exist_ok=True)
        request = read(self.inputs / "request.json")
        if stage == "collect_play":
            report = {"ok": True, "synthetic": True, "human": 0, "headless": 1, "traces": 1}
        elif stage == "train":
            export = out / "export"
            export.mkdir(exist_ok=True)
            gemini.atomic_json(export / "config.json", {"smoke_only": True, "synthetic": True,
                "task": request["task"]["name"], "task_spec": request["task"], "physics": request["physics"],
                "action": "absolute_q", "horizon": 15, "hz": 30, "obs_frame": "spec_footprint",
                "start_q": [0]*6, "obs_layout": [], "obs_const": {},
                "eval": {"n": 1, "clean": {"n": 1, "success": 0}, "real": {"n": 1, "success": 0}}})
            report = {"ok": True, "synthetic": True, "replayed": 1, "experts": 1, "export": "export",
                      "expert_mode": "fake", "recipe": training_recipe(request)}
        else:
            report = {"ok": True, "synthetic": True, "eval_n": 1, "clean": 0, "real": 0,
                      "rehearsal_n": 9, "rehearsal_goals": 0, "policy_qualified": False,
                      "reason": "Fake SSH rehearsal fails deliberately; exercise calibrated IK routing"}
        gemini.atomic_json(out / (stage + ".json"), report)
        self.r.event("fake_ssh", stage=stage, lease=LEASE, result=report)
        return report


def training_recipe(request):
    return [sys.executable, "-u", "-m", "r2p.policy.train", "--task", "w_inputs/task.json",
            "--traces", "w_results/mixed", "--name", request["name"], "--out-dir", "w_results/export",
            "--steps", "60000", "--batch-size", "256", "--seeds", "0", "1", "2", "--width", "512",
            "--blocks", "3", "--device", "cuda:0", "--memory-fraction", "0.949", "--no-augmentation", "--eval-n", "30"]


def snapshot(out):
    """Atomic cumulative archive, refreshed while outputs/logs are being produced."""
    temporary = out / "artifacts.zip.tmp"
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in sorted(out.rglob("*")):
            if path.is_file() and path.name not in ("artifacts.zip", "artifacts.zip.tmp"):
                archive.write(path, path.relative_to(out).as_posix())
    temporary.replace(out / "artifacts.zip")


def collect_play(request, inputs, out):
    from r2p.pipeline.game_player import collect_players
    source = inputs / "human"
    destination = out / "play/human"
    shutil.copytree(source, destination)
    players = collect_players(request["task"], out / "play/headless", n=request["headless"],
                              seed=500000, physics=request["physics"])
    human = len(list(destination.glob("*.jsonl")))
    return {"ok": True, "human": human, "headless": len(players), "traces": human + len(players),
            "provenance": "Receiver source=human separately from simulated SkillOperator players"}


def train(request, inputs, out):
    import numpy as np
    from r2p.frames import HOME_Q
    from r2p.pipeline.game_loop import ingest
    from r2p.sim import TaskTwin
    from r2p.sim.expert import Expert
    from r2p.tasks import sample_scene
    from r2p.traces import read_trace, write_trace

    replay = ingest(out / "play", out / "replayed", request["task"], request["physics"])
    gemini.atomic_json(out / "replay.json", replay)
    if not replay["counts"]["relabeled"]:
        raise ValueError("No play trace passed physical replay; training was not launched")
    profile = request["profile"] if request["bench_expert_committed"] else None
    mixed = out / "mixed"
    mixed.mkdir()
    sources = []

    def expert(scene, seed, path, lineage=None):
        with TaskTwin(scene, physics=request["physics"], q0=profile["start_q"] if profile else HOME_Q,
                      joint_limits=profile["model_limits"] if profile else None) as twin:
            result = Expert(request["task"], deployment=profile, seed=seed, dart=.01).run(twin, seed=seed)
        if lineage:
            result["meta"]["play_lineage"] = lineage
        write_trace(path, result["meta"], result["ticks"])
        sources.append({"path": str(path), "success": result["meta"]["success"], "play_lineage": lineage})

    for i in range(request["experts"]):
        seed = 200000 + i
        expert(sample_scene(request["task"], np.random.default_rng(seed)), seed, mixed / f"expert_{seed}.jsonl")
        if i % 16 == 0:
            print(f"EXPERT {i+1}/{request['experts']}", flush=True)
    for path in sorted((out / "replayed").glob("*.jsonl")):
        meta, _ = read_trace(path)
        if profile:
            # Bench start/limits differ from game start. Never relabel metadata to
            # pretend incompatible actions were produced under the bench profile.
            # Re-demonstrate the replayed scene physically with the bench expert.
            expert(meta["scene"], int(meta.get("seed") or 600000), mixed / ("play_" + path.name),
                   {"replay": str(path), "source_sha256": sha(path), "source": meta["source"],
                    "method": "bench expert re-demonstration of accepted replay scene"})
        else:
            shutil.copyfile(path, mixed / ("play_" + path.name))
    data = {"experts": request["experts"], "replayed": replay["counts"]["relabeled"],
            "expert_mode": "P3 bench-profile" if profile else "default (P3 fix not committed at bundle revision)",
            "expert_commit": request["expert_commit"], "sources": sources,
            "play_training": "bench expert re-demonstrations" if profile else "replayed play actions"}
    gemini.atomic_json(out / "training_data.json", data)
    if not any(row["success"] for row in sources):
        raise ValueError("No successful expert trace; refusing an ungrounded training run")
    recipe = training_recipe(request)
    gemini.atomic_json(out / "training_recipe.json", {"argv": recipe, "lease": LEASE})
    with (out / "train.log").open("w", encoding="utf-8") as log:
        subprocess.run(recipe, check=True, stdout=log, stderr=subprocess.STDOUT)
    return {"ok": True, **data, "recipe": recipe, "export": "export"}


def evaluate(request, inputs, out):
    config = read(out / "export/config.json")
    scores = config.get("eval", {})
    clean, real = scores.get("clean") or {}, scores.get("real") or {}
    report_path = out / "rehearsal.json"
    command = [sys.executable, "scripts/robot/rehearse_policy.py", "--export", str(out / "export"),
               "--bench-profile", str(inputs / "profile.json"), "--config", str(inputs / "bench.json"),
               "--measured-task", str(inputs / "task.json"), "--transport", "armd",
               "--cap", str(request["cap"]), "--out", str(report_path)]
    with (out / "rehearsal.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    rehearsal = read(report_path) if report_path.is_file() else {}
    n = rehearsal.get("n", 0)
    goals = rehearsal.get("successes", 0)
    qualified = result.returncode == 0 and n == 9 and goals == n and not config.get("smoke_only")
    return {"ok": True, "eval_n": min(clean.get("n", 0), real.get("n", 0)),
            "clean": clean.get("success", 0), "real": real.get("success", 0),
            "rehearsal_n": n, "rehearsal_goals": goals, "rehearsal_exit": result.returncode,
            "policy_qualified": qualified, "command": command,
            "reason": "All nine cell rehearsals reached goals" if qualified else
                      "Rehearsal failed or was incomplete; use one calibrated IK turn",
            "export_hashes": {p.relative_to(out / "export").as_posix(): sha(p)
                              for p in (out / "export").rglob("*") if p.is_file()},
            "physical_success_demonstrated": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("collect_play", "train", "evaluate"), required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if any(os.environ.get(k) != v for k, v in LEASE.items()):
        raise RuntimeError("Remote worker requires the exact authorized B200 lease environment")
    if time.time() >= RENTAL_END:
        raise RuntimeError("Supplied B200 rental has expired")
    args.out.mkdir(parents=True, exist_ok=True)
    request = read(args.inputs / "request.json")
    # The local profile/config are frozen and rebased; no laptop-specific paths.
    bench = read(args.inputs / "bench.json")
    bench["joint_map"] = str(args.inputs / "joint_map.json")
    gemini.atomic_json(args.inputs / "bench.json", bench)
    # Mirror the captured bench configuration only inside this isolated worktree.
    # sag_gain and the rehearsal both resolve physics by the rig's config name.
    rig_path = Path("config/physics") / (bench["rig"] + ".json")
    if not rig_path.resolve().is_relative_to(Path("config/physics").resolve()):
        raise ValueError("Bench rig path escapes physics directory")
    shutil.copyfile(args.inputs / "bench_physics.json", rig_path)
    stop = threading.Event()
    def copy_loop():
        while not stop.wait(20):
            try:
                snapshot(args.out)
            except OSError as exc:
                print("Snapshot retry:", exc, flush=True)
    thread = threading.Thread(target=copy_loop, daemon=True)
    thread.start()
    report = {"ok": False}
    try:
        report = globals()[args.stage](request, args.inputs, args.out)
    except Exception as exc:
        report = {"ok": False, "error": gemini._safe(exc), "stage": args.stage}
    finally:
        stop.set()
        thread.join()
        gemini.atomic_json(args.out / (args.stage + ".json"), report)
        snapshot(args.out)
    print(json.dumps(report), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
