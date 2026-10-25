"""Adapters for the independently delivered components. No eager ML/robot imports."""

from contextlib import contextmanager
import copy
import importlib
import inspect
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from .engine import Blocked
from .gate import compare
from .lease import validate_lease
from .storage import atomic_json, digest, file_digest, read_json


def safe_error(error):
    message = str(error)
    secret = os.getenv("GEMINI_API_KEY")
    if secret:
        message = message.replace(secret, "[REDACTED]")
    return message[:8000]


def dependency(module, name=None):
    try:
        loaded = importlib.import_module(module)
    except ImportError as exc:
        raise Blocked(f"Missing dependency {module}: {exc}. Install/deliver this component, then rerun.") from exc
    if name is not None:
        if not callable(getattr(loaded, name, None)):
            raise Blocked(f"Dependency {module}.{name} is not available yet; rerun after its owner delivers it.")
        return getattr(loaded, name)
    return loaded


def entrypoint(candidates):
    for module, name in candidates:
        try:
            return dependency(module, name)
        except Blocked:
            continue
    raise Blocked("Missing component entry point: " + " or ".join(f"{m}.{n}" for m, n in candidates))


def invoke(function, **values):
    """Bind supported explicit keywords; reject missing parameters before side effects."""
    signature = inspect.signature(function)
    keywords = {name: values[name] for name, p in signature.parameters.items()
                if name in values and p.kind != p.POSITIONAL_ONLY}
    try:
        signature.bind(**keywords)
    except TypeError as exc:
        raise Blocked(f"Adapter for {function.__module__}.{function.__name__} needs {exc}") from exc
    return function(**keywords)


@contextmanager
def environment(values):
    old = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def task_of(meta):
    task = meta.get("task")
    return meta.get("task_name") or (task.get("name") if isinstance(task, dict) else task)


def trace_meta(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.loads(next(stream))["meta"]


class ComponentStages:
    def __init__(self, pipeline):
        self.p = pipeline
        self.o = pipeline.options
        self.root = pipeline.root

    def path(self, value):
        value = Path(value)
        return value if value.is_absolute() else self.root / value

    def folder(self, round_name, stage):
        path = self.p.run_dir / round_name / stage
        path.mkdir(parents=True, exist_ok=True)
        return path

    def attempt_folder(self, round_name, stage):
        row = self.p.manifest.get("rounds", {}).get(round_name, {}).get("stages", {}).get(stage, {})
        return self.folder(round_name, stage) / "attempts" / f"generation_{row.get('generation', 1):03d}"

    def write(self, path, value):
        atomic_json(path, value)
        return self.p.artifact(path)

    def inputs(self, round_name, stage):
        files = []
        task = self.path(self.o.task)
        task = task if task.is_file() else self.root / "tasks" / self.o.task / "task.json"
        if task.is_file():
            files.append(task)
        if stage == "session" and self.o.video:
            video = self.path(self.o.video)
            files.extend([video] if video.is_file() else sorted(video.rglob("*")))
        if stage == "data":
            files.extend(sorted((self.root / "roblox/out/traces").rglob("*.jsonl")))
        if stage in {"session", "deploy"} and self.o.perception_setup:
            files.append(self.path(self.o.perception_setup))
        if stage in {"close", "data", "eval"}:
            files.append(self.root / "config/physics/so101_bench.json")
        if stage in {"eval", "gate", "deploy"}:
            for role in ("incumbent",):
                value = self.p.result(round_name, "train").get(role)
                if value:
                    files.extend(self.path(value).glob("*"))
        return {str(path): file_digest(path) for path in files if path.is_file()}

    def run(self, round_name, stage):
        with environment(self.o.environment()):
            return getattr(self, stage)(round_name)

    def task(self, round_name):
        return read_json(self.p.result(round_name, "session")["task_file"])

    def load_task(self, folder):
        target = self.path(self.o.task)
        if target.is_file() or (self.root / "tasks" / self.o.task / "task.json").is_file():
            value = target if target.is_file() else self.root / "tasks" / self.o.task / "task.json"
            task = read_json(value)
        elif " " in self.o.task:
            if not self.o.video:
                raise Blocked("An instruction needs --video <recorded session> for task_from_video")
            function = entrypoint([("r2p.real2sim", "task_from_video"), ("r2p.real2sim.task_from_video", "task_from_video")])
            task = invoke(function, source=self.path(self.o.video), session=self.path(self.o.video), video=self.path(self.o.video),
                          instruction=self.o.task, out_dir=folder, out_root=folder / "inferred", fit_twin=False,
                          out=folder / "task.json")
            if isinstance(task, tuple):
                task = task[0]
            if isinstance(task, (str, Path)):
                task = read_json(task)
            if isinstance(task, dict) and "task" in task and "objects" not in task:
                task = task["task"]
        else:
            task = dependency("r2p.tasks", "load_task")(self.o.task)
        errors = dependency("r2p.tasks", "validate")(task)
        if errors:
            raise Blocked("Task validation failed: " + "; ".join(map(str, errors)))
        atomic_json(folder / "task.json", task)
        return task

    def command(self, argv, folder, extra_env=None, *, allow_unsuccessful_attempt=False):
        env = {**os.environ, **self.o.environment(), **(extra_env or {})}
        # Capture only a bounded tail in the manifest; never expose the Gemini key.
        completed = subprocess.run(argv, cwd=self.root, env=env, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", check=False)
        output = safe_error(completed.stdout + "\n" + completed.stderr)
        (folder / "command.log").write_text(output, encoding="utf-8")
        if completed.returncode and not (allow_unsuccessful_attempt and completed.returncode == 1):
            raise Blocked(f"Component exited {completed.returncode}: {shlex.join(map(str, argv))}\n{output}")

    def bench_command(self, policy, task_file, out, dry):
        argv = [sys.executable, "scripts/robot/run_policy.py", "--policy", str(policy), "--task", str(task_file),
                "--attempts", "1", "--record", str(out)]
        if not dry:
            argv += ["--perception-setup", str(self.path(self.o.perception_setup or "config/perception_setup.json"))]
        if dry:
            argv.append("--dry-run")
        return argv

    def recorded_session(self, folder):
        candidates = sorted(folder.rglob("*.jsonl"))
        for trace in candidates:
            if (trace_meta(trace).get("schema") == "r2p_trace/1" and
                    any((trace.parent / "frames").glob("*.jpg"))):
                return trace.parent
        raise Blocked(f"No recorded r2p_trace/1 session with frames in {folder}; run the printed bench command")

    def session(self, name):
        folder = self.folder(name, "session")
        self.load_task(folder)
        task_file = folder / "task.json"
        if int(name[1:]) > 1:
            previous = self.p.result(f"r{int(name[1:])-1}", "deploy")
            if not previous.get("session"):
                raise Blocked(f"{name} needs the preceding round's recorded deployment attempt")
            session = self.path(previous["session"])
        elif self.o.video:
            session = self.path(self.o.video)
            if not session.exists():
                raise Blocked(f"Recorded session does not exist: {session}")
        else:
            out = self.attempt_folder(name, "session")
            try:
                session = self.recorded_session(out)
            except Blocked:
                argv = self.bench_command("stub", task_file, out, self.o.mode == "dry-run")
                if self.o.mode == "plan":
                    raise Blocked("Probe attempt required. Exact bench command: " + shlex.join(argv))
                if self.o.mode == "dry-run":
                    dependency("r2p.synthbench", "World")
                if not (self.root / argv[1]).is_file():
                    raise Blocked("Missing C1 runner scripts/robot/run_policy.py. Command: " + shlex.join(argv))
                self.command(argv, folder, allow_unsuccessful_attempt=True)
                session = self.recorded_session(out)
        source_files = [session] if session.is_file() else sorted(p for p in session.rglob("*") if p.is_file())
        synthetic = self.o.mode == "dry-run"
        if session.is_dir():
            traces = list(session.glob("*.jsonl"))
            if traces:
                meta = trace_meta(traces[0])
                synthetic = synthetic or bool(meta.get("synthetic")) or meta.get("source") != "real"
        return {"session": str(session), "task_file": str(task_file), "synthetic": synthetic,
                "artifacts": [self.p.artifact(task_file), *[self.p.artifact(p) for p in source_files]],
                "message": ("SYNTHETIC" if synthetic else "RECORDED SESSION") + f" {session}"}

    def twin(self, name):
        # D preserves previous run evidence; a retry uses a fresh versioned live directory.
        folder = self.folder(name, "twin") / f"attempt_{self.p.manifest['rounds'][name]['stages']['twin']['attempt']:03d}"
        session = self.p.result(name, "session")
        function = entrypoint([("r2p.real2sim", "run_twin"), ("r2p.real2sim.loop", "run_twin"),
                               ("r2p.real2sim", "run")])
        source = Path(session["session"])
        loader = getattr(dependency("r2p.real2sim"), "load_session", None)
        if session["synthetic"] and callable(loader):
            source = loader(source)
            source.meta = {**source.meta, "session_kind": "synthetic"}
        result = invoke(function, source=source, session=source, task=self.task(name),
                        out_dir=folder, out=folder, run_dir=folder, max_iters=self.o.twin_iters,
                        iterations=self.o.twin_iters, synthetic=session["synthetic"])
        # Require actual scene evidence, not merely a successful exit code.
        scene = folder / "scene_best.json"
        if isinstance(result, dict):
            value = result.get("best_scene", result.get("scene"))
            if isinstance(value, dict):
                atomic_json(scene, value)
            elif value:
                value = Path(value)
                scene = value if value.is_absolute() else folder / value
        if not scene.is_file():
            raise Blocked(f"Twin completed without a best scene: expected {scene}")
        # Source bot/expert traces are synthetic even if a component's older loader calls them bench.
        live = folder / "run.json"
        if live.exists() and session["synthetic"]:
            metadata = read_json(live)
            metadata.update(session_kind="synthetic", synthetic=True)
            atomic_json(live, metadata)
        return {"scene": str(scene), "live": str(folder / "run.json"),
                "artifacts": [self.p.artifact(p) for p in sorted(folder.glob("*.json"))],
                "message": f"Twin fitted from {session['session']} ({'synthetic' if session['synthetic'] else 'recorded'})"}

    def close(self, name):
        folder = self.folder(name, "close")
        session = self.p.result(name, "session")
        scene = read_json(self.p.result(name, "twin")["scene"])
        sysid = dependency("r2p.sysid")
        physics = read_json(self.root / "config/physics/so101_bench.json")
        traces = sorted(Path(session["session"]).glob("*.jsonl"))
        if not traces:
            raise Blocked("Closing the loop requires a recorded trace, not only a video")
        meta, ticks = dependency("r2p.traces", "read_trace")(traces[0])
        if not ticks:
            raise Blocked("Closing the loop requires a nonempty recorded trace")
        focus = copy.deepcopy(scene)
        # Perception is authoritative for positions. Never substitute twin guesses for observations.
        poses = ticks[0].get("poses", {})
        for obj in focus.get("objects", []):
            pose = poses.get(obj["id"])
            if pose:
                for key in ("xyz", "yaw_deg"):
                    if key in pose:
                        obj[key] = pose[key]
        attempt = {"id": str(traces[0]), "meta": {**meta, "task": self.task(name)}, "scene": focus,
                   "ticks": ticks, "outcome": meta.get("success")}
        prior_attempts = []
        for number in range(1, int(name[1:])):
            previous = self.p.result(f"r{number}", "close").get("attempt")
            if previous:
                prior_attempts.append(read_json(previous))
        if prior_attempts:
            result = sysid.fit_parameters(prior_attempts, physics, heldout=[attempt], max_nfev=20)
            # A failed held-out gate must never update the physics used downstream.
            if result["gate"]["passed"]:
                physics = result["physics"]
            result["physics_applied"] = bool(result["gate"]["passed"])
        else:
            validation = sysid.validation_gate([attempt], physics)
            result = {"physics": physics, "gate": validation, "physics_applied": False,
                      "fit_status": "needs_independent_training_and_heldout_attempts",
                      "note": "Baseline replay only. No parameters fitted or accuracy claim made from one attempt."}
        residuals = {key: {"value": value, "tolerance": result["gate"]["thresholds"].get(key, 1)}
                     for key, value in result["gate"].get("metrics", {}).items() if isinstance(value, (int, float))}
        result["next_experiments"] = sysid.plan_experiments(residuals, q0=ticks[0]["q"])
        artifact = self.write(folder / "sysid.json", result)
        attempt_artifact = self.write(folder / "attempt.json", attempt)
        physics_artifact = self.write(folder / "physics.json", physics)
        if not result["gate"]["passed"]:
            originals = {o["id"]: o for o in self.task(name)["objects"]}
            for obj in focus["objects"]:
                for key in ("mass_kg", "friction"):
                    obj[key] = originals[obj["id"]].get(key, physics.get("object", {}).get(key, .03 if key == "mass_kg" else .6))
        focus["provenance"] = {"source": str(traces[0]), "synthetic": session["synthetic"],
                               "positions": "first recorded perception", "physics": "sysid.json"}
        focus_file = folder / "focus_scene.json"
        focus_artifact = self.write(focus_file, focus)
        # The report, including a failed accuracy gate, stays visible. Fitting is not a claim of exactness.
        return {"focus_scene": str(focus_file), "physics": str(folder / "physics.json"),
                "physics_applied": result["physics_applied"], "attempt": str(folder / "attempt.json"),
                "sysid": str(folder / "sysid.json"), "artifacts": [artifact, focus_artifact, attempt_artifact, physics_artifact],
                "message": "Replay compared; focus uses recorded poses; " +
                           ("held-out fitted physics applied" if result["physics_applied"] else "baseline physics retained; identification pending")}

    def data(self, name):
        folder = self.folder(name, "data")
        generate = dependency("r2p.sim", "gen_traces")
        # The public wrapper forwards **kwargs; bind the concrete implementation's optional seed.
        if any(p.kind == p.VAR_KEYWORD for p in inspect.signature(generate).parameters.values()):
            generate = dependency("r2p.sim.traces", "gen_traces")
        task = self.task(name)
        focus_task = copy.deepcopy(task)
        focus = read_json(self.p.result(name, "close")["focus_scene"])
        focus_by_id = {o["id"]: o for o in focus.get("objects", [])}
        # Focus half of the generated data around observed starting positions.
        randomize = focus_task.setdefault("randomize", {})
        for obj in focus_task.get("objects", []):
            if obj["id"] in focus_by_id:
                fitted = focus_by_id[obj["id"]]
                obj["xyz"] = fitted["xyz"]
                randomize[obj["id"]] = {"xy_choice": [fitted["xyz"][:2]], "yaw_deg": [fitted.get("yaw_deg", 0)] * 2}
        attempt = self.p.manifest["rounds"][name]["stages"]["data"]["attempt"]
        batch = folder / f"attempt_{attempt:03d}"
        prior_dir, focus_dir = batch / "prior", batch / "focus"
        physics = self.p.result(name, "close")["physics"]
        if self.p.result(name, "close")["physics_applied"]:
            from .collection import collect
            generate = collect
        base_seed = self.o.seed + int(name[1:]) * 100000
        invoke(generate, task=task, n=max(1, self.o.traces // 2), out_dir=prior_dir, workers=self.o.workers,
               physics=physics, seed=base_seed)
        if self.o.traces > 1:
            invoke(generate, task=focus_task, n=self.o.traces - max(1, self.o.traces // 2), out_dir=focus_dir,
                   workers=self.o.workers, physics=physics, seed=base_seed + self.o.traces)
        traces = sorted(batch.rglob("*.jsonl"))
        task_name = task["name"]
        for path in sorted((self.root / "roblox/out/traces").rglob("*.jsonl")):
            try:
                if task_of(trace_meta(path)) == task_name:
                    traces.append(path)
            except (OSError, ValueError, KeyError, StopIteration):
                continue
        # Include previous rounds' experts, filtered using metadata rather than filenames.
        for number in range(1, int(name[1:])):
            previous_index = self.p.result(f"r{number}", "data").get("index")
            if previous_index:
                traces.extend(Path(p) for p in read_json(previous_index)["traces"])
        traces = sorted({path.resolve() for path in traces})
        traces = [path for path in traces if task_of(trace_meta(path)) == task_name]
        if not traces:
            raise Blocked(f"No r2p_trace/1 traces for task {task_name}")
        index = folder / "index.json"
        artifact = self.write(index, {"task": task_name, "traces": [str(p) for p in traces]})
        return {"index": str(index), "count": len(traces),
                "artifacts": [artifact, *[self.p.artifact(p) for p in traces]],
                "message": f"{len(traces)} task-matched expert / Roblox traces"}

    def label(self, name):
        folder = self.folder(name, "label") / f"attempt_{self.p.manifest['rounds'][name]['stages']['label']['attempt']:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        segment = dependency("r2p.labeling", "segment")
        read = dependency("r2p.traces", "read_trace")
        write = dependency("r2p.traces", "write_trace")
        outputs = []
        for value in read_json(self.p.result(name, "data")["index"])["traces"]:
            path = Path(value)
            meta, ticks = read(path)
            commands = invoke(segment, trace=path, task=self.task(name))
            meta["commands"] = commands
            for tick in ticks:
                matches = [c for start, end, c in commands if start <= tick["t"] < end]
                if matches:
                    tick["command"] = matches[-1]
            out = folder / (file_digest(path)[:20] + ".jsonl")
            write(out, meta, ticks)
            outputs.append(self.p.artifact(out))
        return {"traces": str(folder), "artifacts": outputs,
                "message": f"{len(outputs)} episodes labelled with task skill commands"}

    def train(self, name):
        folder = self.folder(name, "train")
        env = validate_lease(self.path(self.o.lease), task=self.p.key, round_name=name)
        dependency("r2p.policy.train")
        attempt = self.p.manifest["rounds"][name]["stages"]["train"]["attempt"]
        export_name = f"{self.p.key}_{name}_a{attempt}"
        task_file = self.p.result(name, "session")["task_file"]
        argv = [sys.executable, "-m", "r2p.pipeline.train_worker", "--task", task_file,
                "--traces", self.p.result(name, "label")["traces"], "--name", export_name,
                "--steps", str(self.o.steps), "--device", self.o.device]
        self.command(argv, folder, env)
        export = self.root / "exports" / export_name
        required = [export / filename for filename in ("config.json", "norm.json", "weights.npz", "runtime.py")]
        if not all(p.is_file() for p in required):
            raise Blocked(f"Training did not produce the complete policy export at {export}")
        config = read_json(export / "config.json")
        if any(k not in config for k in ("task", "start_q", "obs_layout", "obs_const", "eval")) or config.get("action") != "absolute_q":
            raise Blocked("Policy export is missing its declared task/start_q/obs_layout/obs_const/eval or absolute_q action")
        previous = self.p.result(f"r{int(name[1:])-1}", "gate") if name != "r1" else {}
        incumbent = previous.get("selected", self.o.incumbent or str(self.root / "exports" / f"{self.task(name)['name']}_v1"))
        return {"export": str(export), "incumbent": incumbent, "artifacts": [self.p.artifact(p) for p in required],
                "message": f"Trained {export_name} under lease {env['LANDLORD_LEASE_ID']}"}

    def eval(self, name):
        from .evaluation import make_plan, evaluate_pair
        folder = self.folder(name, "eval")
        trained = self.p.result(name, "train")
        for policy in (trained["export"], trained["incumbent"]):
            if not self.path(policy).is_dir() and not (self.root / "exports" / policy).is_dir():
                raise Blocked(f"Policy {policy} is missing; supply --incumbent <export> for a paired gate")
        plan = make_plan(self.task(name), read_json(self.p.result(name, "close")["focus_scene"]),
                         self.o.eval_n, self.o.seed + int(name[1:]))
        plan["physics"] = read_json(self.p.result(name, "close")["physics"])
        self.write(folder / "scenes.json", plan)
        challenger, incumbent = evaluate_pair(trained["export"], trained["incumbent"], self.task(name), plan)
        outputs = [self.write(folder / "challenger.json", challenger), self.write(folder / "incumbent.json", incumbent),
                   self.p.artifact(folder / "scenes.json")]
        return {"plan": str(folder / "scenes.json"), "challenger": str(folder / "challenger.json"),
                "incumbent": str(folder / "incumbent.json"), "artifacts": outputs,
                "message": f"Challenger vs incumbent {trained['incumbent']} on IDENTICAL scenes {digest(plan)[:12]} and noise"}

    def gate(self, name):
        evaluation = self.p.result(name, "eval")
        result = compare(read_json(evaluation["challenger"]), read_json(evaluation["incumbent"]), read_json(evaluation["plan"]))
        result["artifacts"] = [self.write(self.folder(name, "gate") / "decision.json", result)]
        return result

    def deploy(self, name):
        folder = self.folder(name, "deploy")
        selected = self.p.result(name, "gate")["selected"]
        out = self.attempt_folder(name, "deploy")
        argv = self.bench_command(selected, self.p.result(name, "session")["task_file"], out, self.o.mode == "dry-run")
        try:
            session = self.recorded_session(out)
        except Blocked:
            if self.o.mode == "plan":
                raise Blocked("Deploy ready. Exact bench command: " + shlex.join(argv))
            if self.o.mode == "dry-run":
                dependency("r2p.synthbench", "World")
            if not (self.root / argv[1]).is_file():
                raise Blocked("Missing C1 runner. Command: " + shlex.join(argv))
            self.command(argv, folder, allow_unsuccessful_attempt=True)
            session = self.recorded_session(out)
        return {"policy": selected, "session": str(session), "synthetic": self.o.mode == "dry-run",
                "command": argv, "artifacts": [self.p.artifact(p) for p in sorted(session.rglob("*")) if p.is_file()],
                "message": f"{'SIM DRY RUN' if self.o.mode == 'dry-run' else 'REAL'} attempt recorded: {session}"}
