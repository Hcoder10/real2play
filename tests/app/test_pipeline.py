import json
from pathlib import Path
import threading

import pytest

from r2p.cli import main, parser
from r2p.pipeline.engine import Blocked, Pipeline, RunOptions, STAGES
from r2p.pipeline.storage import atomic_json, read_json, run_lock


class FakeStages:
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail
        self.version = 0

    def run(self, round_name, stage):
        self.calls.append((round_name, stage))
        if stage == self.fail:
            raise Blocked("Dependency has not arrived")
        return {"message": f"fixture {stage}", "value": self.version if stage == "data" else stage}


def pipeline(tmp_path, fake, **options):
    return Pipeline(RunOptions(task="cube_in_bowl", root=tmp_path, mode="dry-run", **options), adapters=fake, emit=lambda _: None)


def test_atomic_replace_failure_preserves_previous(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    atomic_json(path, {"version": 1})
    def fail(*_):
        raise OSError("injected disk failure")
    monkeypatch.setattr("r2p.pipeline.storage.os.replace", fail)
    with pytest.raises(OSError):
        atomic_json(path, {"version": 2})
    assert read_json(path) == {"version": 1}
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_readers_never_see_partial_json(tmp_path):
    path = tmp_path / "manifest.json"
    atomic_json(path, {"version": 0})
    done, errors = threading.Event(), []
    def reader():
        while not done.is_set():
            try:
                assert isinstance(read_json(path)["version"], int)
            except PermissionError:
                pass  # Windows permits temporary sharing violations, but never partial JSON.
            except Exception as exc:
                errors.append(exc)
    thread = threading.Thread(target=reader)
    thread.start()
    try:
        for version in range(20):
            atomic_json(path, {"version": version, "payload": "x" * 10000})
    finally:
        done.set()
        thread.join()
    assert not errors


def test_nonfinite_json_does_not_replace(tmp_path):
    path = tmp_path / "a.json"
    atomic_json(path, {"valid": True})
    with pytest.raises(ValueError):
        atomic_json(path, {"bad": float("nan")})
    assert read_json(path) == {"valid": True}


def test_lock_released_after_exception(tmp_path):
    with pytest.raises(RuntimeError, match="test"):
        with run_lock(tmp_path / ".lock"):
            raise RuntimeError("test")
    with run_lock(tmp_path / ".lock"):
        pass


def test_stage_cache_and_resume(tmp_path):
    fake = FakeStages(fail="twin")
    first = pipeline(tmp_path, fake)
    assert not first.run()
    manifest = read_json(first.path)
    assert manifest["schema"] == "r2p_run/1"
    assert manifest["rounds"]["r1"]["stages"]["session"]["status"] == "DONE"
    assert manifest["rounds"]["r1"]["stages"]["train"]["status"] == "BLOCKED"
    assert fake.calls == [("r1", "session"), ("r1", "twin")]
    fake.fail = None
    resumed = pipeline(tmp_path, fake)
    assert resumed.run()
    assert fake.calls.count(("r1", "session")) == 1
    before = fake.calls.copy()
    assert pipeline(tmp_path, fake).run()
    assert fake.calls == before


def test_forcing_cascades_through_later_rounds(tmp_path):
    fake = FakeStages()
    assert pipeline(tmp_path, fake, rounds=2).run()
    fake.calls.clear()
    forced = pipeline(tmp_path, fake, rounds=2, force=("r1:data",))
    assert forced.run()
    assert fake.calls == [("r1", s) for s in STAGES[3:]] + [("r2", s) for s in STAGES]
    assert forced.manifest["rounds"]["r1"]["stages"]["data"]["attempt"] == 2


def test_no_cascade_keeps_downstream_even_if_forced_output_changes(tmp_path):
    fake = FakeStages()
    assert pipeline(tmp_path, fake, rounds=2).run()
    fake.calls.clear()
    fake.version = 1
    assert pipeline(tmp_path, fake, rounds=2, force=("r1:data",), no_cascade=True).run()
    assert fake.calls == [("r1", "data")]


def test_missing_artifact_invalidates_cached_stage(tmp_path):
    fake = FakeStages()
    p = pipeline(tmp_path, fake)
    artifact = tmp_path / "proof.json"
    atomic_json(artifact, {"value": 1})
    original_run = fake.run
    def run(r, stage):
        result = original_run(r, stage)
        if stage == "session":
            result["artifacts"] = [p.artifact(artifact)]
        return result
    fake.run = run
    assert p.run()
    atomic_json(artifact, {"value": 2})
    fake.calls.clear()
    assert pipeline(tmp_path, fake).run()
    assert ("r1", "session") in fake.calls
    assert ("r1", "deploy") in fake.calls


def test_round_count_extension_does_not_recompute_finished_round(tmp_path):
    fake = FakeStages()
    assert pipeline(tmp_path, fake).run()
    fake.calls.clear()
    assert pipeline(tmp_path, fake, rounds=2).run()
    assert fake.calls == [("r2", stage) for stage in STAGES]


def test_model_changes_invalidate_cache(tmp_path):
    fake = FakeStages()
    assert pipeline(tmp_path, fake).run()
    fake.calls.clear()
    assert pipeline(tmp_path, fake, main_model="test-main").run()
    assert len(fake.calls) == len(STAGES)


def test_no_cascade_does_not_ignore_changed_models(tmp_path):
    fake = FakeStages()
    assert pipeline(tmp_path, fake).run()
    fake.calls.clear()
    assert pipeline(tmp_path, fake, main_model="changed", force=("r1:data",), no_cascade=True).run()
    assert len(fake.calls) == len(STAGES)


def test_blocked_retry_preserves_generation_and_force_advances_it(tmp_path):
    fake = FakeStages(fail="deploy")
    p = pipeline(tmp_path, fake)
    assert not p.run()
    first=p.manifest["rounds"]["r1"]["stages"]["deploy"]
    assert first["generation"]==1
    fake.fail=None
    resumed=pipeline(tmp_path,fake)
    assert resumed.run()
    row=resumed.manifest["rounds"]["r1"]["stages"]["deploy"]
    assert row["generation"]==1 and row["attempt"]==2
    forced=pipeline(tmp_path,fake,force=("r1:deploy",))
    assert forced.run()
    assert forced.manifest["rounds"]["r1"]["stages"]["deploy"]["generation"]==2


@pytest.mark.parametrize("exception_name", ["CacheMiss", "QuotaError", "ModuleNotFoundError"])
def test_blocked_error_paths_redact_key(tmp_path, monkeypatch, exception_name):
    monkeypatch.setenv("GEMINI_API_KEY", "never-print-this-test-key")
    fake = FakeStages()
    def run(*_):
        raise type(exception_name, (RuntimeError,), {})("Daily 429 / cache miss: never-print-this-test-key")
    fake.run = run
    p = pipeline(tmp_path, fake)
    assert not p.run()
    raw = p.path.read_text()
    assert "never-print-this-test-key" not in raw
    assert "[REDACTED]" in raw
    assert p.manifest["blocked_at"] == "r1:session"


def test_dry_run_and_plan_manifest_are_separate(tmp_path):
    dry = pipeline(tmp_path, FakeStages())
    real = Pipeline(RunOptions(task="cube_in_bowl", root=tmp_path, mode="real"), adapters=FakeStages(), emit=lambda _:None)
    assert dry.path != real.path


def test_cli_status_and_blocked_missing_task(tmp_path, capsys):
    assert main(["run", "--task", "missing_app_fixture", "--dry-run", "--root", str(tmp_path)]) == 2
    assert main(["status", "--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "SIM DRY RUN" in out and "BLOCKED" in out
    path = tmp_path / "runs/missing_app_fixture-dry/manifest.json"
    assert read_json(path)["models"]["R2P_CACHE_ONLY"] == "1"


def test_cli_models_flags_and_force():
    args = parser().parse_args(["run", "--task", "put the cube in the bowl", "--video", "session",
                               "--main-model", "main", "--small-model", "small", "--allow-fallback",
                               "--no-cache-only", "--device", "mps", "--force", "r2:sysid", "--no-cascade"])
    assert args.cache_only is False and args.device == "mps" and args.force == ["r2:sysid"]
    with pytest.raises(ValueError):
        RunOptions(task="t", force=("r0:train",))
    with pytest.raises(ValueError):
        RunOptions(task="t", workers=5)


def test_lazy_missing_component_and_lease_block(tmp_path):
    from r2p.pipeline.stages import dependency
    with pytest.raises(Blocked, match="Missing dependency"):
        dependency("r2p.component_not_delivered")
    p = Pipeline(RunOptions(task="task", root=tmp_path, device="cuda"))
    with pytest.raises(Blocked, match="rent_apartment"):
        p.adapters.train("r1")


def test_plan_deploy_prints_exact_c1_command(tmp_path):
    p = Pipeline(RunOptions(task="task", root=tmp_path, perception_setup="setup.json"))
    p.manifest = {"rounds": {"r1": {"stages": {
        "session": {"result": {"task_file": "task.json"}},
        "gate": {"result": {"selected": "exports/incumbent"}}
    }}}}
    with pytest.raises(Blocked, match="Exact bench command") as error:
        p.adapters.deploy("r1")
    command = str(error.value)
    assert "scripts/robot/run_policy.py" in command and "--record" in command
    assert "--policy exports/incumbent" in command and "--perception-setup" in command
    assert "--out" not in command
