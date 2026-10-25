"""Small persistent state machine with explicit cache dependencies."""

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import re
import time

from .storage import atomic_json, digest, file_digest, read_json, run_lock, utcnow

STAGES = ("session", "twin", "close", "data", "label", "train", "eval", "gate", "deploy")


class Blocked(RuntimeError):
    """A dependency or required evidence is unavailable; retry after it arrives."""


@dataclass
class RunOptions:
    task: str
    root: Path = field(default_factory=Path.cwd)
    video: str | None = None
    rounds: int = 1
    mode: str = "plan"
    main_model: str = field(default_factory=lambda: os.getenv("R2P_MODEL_MAIN", "gemini-4-argon"))
    small_model: str = field(default_factory=lambda: os.getenv("R2P_MODEL_SMALL", "gemini-3.8-flash"))
    allow_fallback: bool = False
    cache_only: bool = True
    force: tuple[str, ...] = ()
    no_cascade: bool = False
    device: str = "cpu"
    traces: int | None = None
    workers: int = 1
    steps: int | None = None
    eval_n: int | None = None
    seed: int = 2026
    incumbent: str | None = None
    lease: str = "runs/lease.json"
    twin_iters: int = 4
    perception_setup: str | None = None

    def __post_init__(self):
        self.root = Path(self.root).resolve()
        for field_name, smoke, full in (("traces", 2, 200), ("steps", 5, 10000), ("eval_n", 2, 100)):
            if getattr(self, field_name) is None:
                setattr(self, field_name, smoke if self.mode == "dry-run" else full)
        if self.mode not in {"plan", "dry-run", "real"}:
            raise ValueError("mode must be plan, dry-run or real")
        if self.device not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be cpu, cuda or mps")
        if not 1 <= self.workers <= 4:
            raise ValueError("workers must be between 1 and 4 (shared laptop RAM)")
        if min(self.rounds, self.traces, self.steps, self.eval_n, self.twin_iters) < 1:
            raise ValueError("rounds, traces, steps, eval-n and twin-iters must be positive")
        for force in self.force:
            if not re.fullmatch(r"r[1-9][0-9]*:(" + "|".join(STAGES) + r"|sysid)", force):
                raise ValueError(f"Invalid --force {force!r}; expected rN:<stage>")

    def environment(self):
        return {"R2P_MODEL_MAIN": self.main_model, "R2P_MODEL_SMALL": self.small_model,
                "R2P_ALLOW_FALLBACK": "1" if self.allow_fallback else "0",
                "R2P_CACHE_ONLY": "1" if self.cache_only else "0"}


def task_key(value):
    candidate = Path(value)
    if candidate.suffix.lower() == ".json":
        value = candidate.parent.name if candidate.stem == "task" else candidate.stem
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")[:64] or "task"
    if " " in value or len(value) > 64:
        slug += "_" + digest(value)[:8]
    return slug


class Pipeline:
    def __init__(self, options, *, adapters=None, emit=print):
        self.options = options
        self.root = options.root
        self.key = task_key(options.task) + ("-dry" if options.mode == "dry-run" else "")
        self.run_dir = self.root / "runs" / self.key
        self.path = self.run_dir / "manifest.json"
        self.emit = emit
        if adapters is None:
            from .stages import ComponentStages
            adapters = ComponentStages(self)
        self.adapters = adapters
        self.manifest = {}

    def save(self):
        self.manifest["updated"] = utcnow()
        atomic_json(self.path, self.manifest)

    def result(self, round_name, stage):
        return self.manifest["rounds"].get(round_name, {}).get("stages", {}).get(stage, {}).get("result", {})

    def artifact(self, path):
        path = Path(path).resolve()
        return {"path": str(path), "sha256": file_digest(path)}

    def artifacts_valid(self, result):
        for artifact in result.get("artifacts", []):
            path = Path(artifact["path"])
            if not path.is_file() or file_digest(path) != artifact["sha256"]:
                return False
        return True

    def invalidate(self):
        for force in self.options.force:
            round_name, stage = force.split(":")
            stage = "close" if stage == "sysid" else stage
            number, index = int(round_name[1:]), STAGES.index(stage)
            for name, row in self.manifest["rounds"].items():
                for target in list(row["stages"]):
                    downstream = int(name[1:]) > number or (name == round_name and STAGES.index(target) >= index)
                    if (self.options.no_cascade and name == round_name and target == stage) or (
                            not self.options.no_cascade and downstream):
                        old = row["stages"][target]
                        row["stages"][target] = {"status": "INVALIDATED", "result": {}, "attempt": old.get("attempt", 0),
                                                 "generation": old.get("generation", 1)}

    @staticmethod
    def generation(previous):
        # A retry can adopt the attempt recorded by a printed plan command. A force or
        # changed completed stage must use a new location, never silently adopt the old attempt.
        return previous.get("generation", 1) + (previous.get("status") in {"DONE", "INVALIDATED"})

    def fingerprints(self, round_name, stage):
        options = asdict(self.options)
        for key in ("root", "rounds", "force", "no_cascade", "lease"):
            options.pop(key)
        preceding = STAGES[:STAGES.index(stage)]
        dependencies = {s: self.result(round_name, s) for s in preceding}
        if stage == "session" and int(round_name[1:]) > 1:
            dependencies["previous_deploy"] = self.result(f"r{int(round_name[1:])-1}", "deploy")
        external = self.adapters.inputs(round_name, stage) if hasattr(self.adapters, "inputs") else {}
        base = {"options": options, "external": external}
        return digest({**base, "dependencies": dependencies}), digest(base)

    def run(self):
        with run_lock(self.run_dir / ".lock"):
            self.manifest = read_json(self.path) if self.path.exists() else {
                "schema": "r2p_run/1", "task": self.options.task, "run": self.key,
                "created": utcnow(), "rounds": {}}
            if self.manifest.get("schema") != "r2p_run/1":
                raise ValueError(f"Unsupported manifest schema: {self.path}")
            self.manifest.update(mode=self.options.mode, synthetic=self.options.mode == "dry-run",
                                 models=self.options.environment(), status="RUNNING")
            self.manifest["profile"] = {"smoke_only": self.options.mode == "dry-run", "device": self.options.device,
                                        "traces_per_round": self.options.traces, "training_steps": self.options.steps,
                                        "eval_scenes": self.options.eval_n,
                                        "eval_max_ticks": 60 if self.options.mode == "dry-run" else None,
                                        "critic": "synthetic-oracle" if self.options.mode == "dry-run" else "Gemini"}
            self.invalidate()
            self.save()
            for number in range(1, self.options.rounds + 1):
                name = f"r{number}"
                stage_rows = self.manifest["rounds"].setdefault(name, {"stages": {}})["stages"]
                blocked_by = None
                for stage in STAGES:
                    if blocked_by:
                        previous = stage_rows.get(stage, {})
                        stage_rows[stage] = {"status": "BLOCKED", "message": f"Waiting for {name}:{blocked_by}",
                                             "updated": utcnow(), "result": {}, "attempt": previous.get("attempt", 0),
                                             "generation": self.generation(previous)}
                        continue
                    fingerprint, base_fingerprint = self.fingerprints(name, stage)
                    previous = stage_rows.get(stage, {})
                    if (previous.get("status") == "DONE" and self.artifacts_valid(previous.get("result", {}))
                            and (previous.get("fingerprint") == fingerprint or
                                 (self.options.no_cascade and self.options.force and
                                  previous.get("base_fingerprint") == base_fingerprint))):
                        self.emit(f"{name}:{stage} CACHED {previous.get('message', '')}")
                        continue
                    row = {"status": "RUNNING", "started": utcnow(), "fingerprint": fingerprint,
                           "base_fingerprint": base_fingerprint,
                           "result": {}, "attempt": previous.get("attempt", 0) + 1,
                           "generation": self.generation(previous)}
                    stage_rows[stage] = row
                    self.save()
                    start = time.monotonic()
                    try:
                        result = self.adapters.run(name, stage)
                        if not isinstance(result, dict):
                            raise TypeError(f"{stage} adapter must return a result object")
                        row.update(status="DONE", result=result, message=result.get("message", ""))
                    except Exception as exc:
                        # Missing components, cache-only misses and quota exhaustion are all resumable.
                        # Persist even unexpected failures, preserving the error type for diagnosis.
                        from .stages import safe_error
                        row.update(status="BLOCKED", error_type=type(exc).__name__, message=safe_error(exc))
                        blocked_by = stage
                    except BaseException:
                        row.update(status="BLOCKED", message="Interrupted; rerun to resume")
                        self.manifest["status"] = "BLOCKED"
                        self.save()
                        raise
                    row.update(elapsed_s=round(time.monotonic() - start, 3), updated=utcnow())
                    self.save()
                    self.emit(f"{name}:{stage} {row['status']} {row.get('message', '')}")
                self.save()
                if blocked_by:
                    self.manifest.update(status="BLOCKED", blocked_at=f"{name}:{blocked_by}")
                    self.save()
                    return False
            self.manifest.update(status="DONE", blocked_at=None)
            self.save()
            return True
