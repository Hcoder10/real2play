from copy import deepcopy
import json
from pathlib import Path

import pytest

from r2p import gemini
from r2p.agents import FakeAgent, ManagedAgent, safe_path, validate_bundle


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts/argon"))
    import _managed
    return _managed


def bundle():
    return {"files": {"task.json": "{}"}, "tests": [{"name": "sandbox check", "passed": True}], "report": "done"}


def test_start_poll_collect_cache_and_artifacts(tmp_path, monkeypatch):
    client = FakeAgent(bundle(), cache_dir=tmp_path)
    result = client.run("Write a file", tag="test")
    assert result["files"] == {"task.json": "{}"}
    assert client.requests_used == 2
    body = client.requests[0][1]
    assert body["agent"] == "antigravity-preview-09-2026"
    assert body["agent_config"] == {"type": "antigravity", "model": "gemini-4-argon"}
    assert body["environment"] == "remote" and body["background"]
    monkeypatch.setenv("R2P_CACHE_ONLY", "1")
    assert client.run("Write a file")["files"] == result["files"]
    assert client.requests_used == 2
    with pytest.raises(gemini.CacheMiss):
        client.run("Different prompt")
    response = {"status": "completed", "output_text": json.dumps(bundle()),
                "steps": [{"type": "model_output", "content": [{"type": "document", "uri": "remote/artifact"}]}]}
    assert client.collect(response)["artifacts"] == [{"type": "document", "uri": "remote/artifact"}]


def test_resume_started_agent_does_not_create_again(tmp_path):
    first = FakeAgent(bundle(), cache_dir=tmp_path, max_requests=1)
    with pytest.raises(gemini.RequestLimit):
        first.run("resume")
    second = FakeAgent(bundle(), cache_dir=tmp_path)
    second.run("resume")
    assert [op for op, _ in second.requests] == ["get"]


def test_secrets_are_refused_before_any_request(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_ACCESS_TOKEN", "secret-token-never-forward")
    client = FakeAgent(bundle(), cache_dir=tmp_path)
    with pytest.raises(ValueError, match="secret"):
        client.run("source: secret-token-never-forward")
    assert not client.requests
    malicious = bundle()
    malicious["files"]["task.json"] = "secret-token-never-forward"
    with pytest.raises(ValueError, match="secret"):
        validate_bundle(malicious)
    with pytest.raises(ValueError, match="secret"):
        ManagedAgent(tmp_path, source_files={"example.txt": "secret-token-never-forward"})


def test_sources_are_mounted_exactly_and_change_request_identity(tmp_path):
    from google.genai._gaos.types.interactions import Environment
    client = FakeAgent(bundle(), cache_dir=tmp_path, source_files={"tests/check.lua": "return true\n"})
    body = client._body("test")
    Environment.model_validate(body["environment"])
    assert body["environment"]["sources"] == [
        {"type": "inline", "target": "/workspace/tests/check.lua", "content": "return true\n"}]
    first = client._path("test")
    client.source_files["tests/check.lua"] = "return false\n"
    assert client._path("test") != first


@pytest.mark.parametrize("path", ["../escape", "/absolute", "C:/drive", "a\\..\\b", "a/../b", ".env",
                                   "a/.git/config", "a//b", "a/./b", "nul.json", "dir/file."])
def test_paths_cannot_escape_or_name_private_files(path):
    with pytest.raises(ValueError):
        safe_path(path)


def test_bundle_rejects_malformed_checks_and_alias_paths():
    invalid = bundle()
    invalid["tests"][0]["passed"] = "true"
    with pytest.raises(gemini.JSONError):
        validate_bundle(invalid)
    invalid = bundle()
    invalid["files"]["TASK.json"] = "{}"
    with pytest.raises(gemini.JSONError):
        validate_bundle(invalid)


def test_task_stage_runs_actual_checker_and_records_acceptance(pipeline, tmp_path):
    result = pipeline.run_stage("author_task", "fake", tmp_path, dry_run=True, goal="Place cube in bowl")
    assert result["passed"] and not result["deployment_ready"]
    assert result["verification"] == {"passed": True, "cases": 4, "windows": 2, "sampled_scenes": 10}
    assert (Path(result["workspace"]) / "tasks/managed_cube_in_bowl/task.json").exists()


def test_authored_wrong_vectors_and_trivial_goals_fail(pipeline):
    value = pipeline.fixture_task_bundle()
    name = "tasks/managed_cube_in_bowl/vectors.json"
    vectors = json.loads(value["files"][name])
    vectors["cases"][0]["ok"] = False
    value["files"][name] = json.dumps(vectors)
    with pytest.raises(ValueError):
        pipeline.verify_task(value)


def test_failed_sandbox_never_applies_files(pipeline, tmp_path):
    value = pipeline.fixture_task_bundle()
    value["tests"][0]["passed"] = False
    agent = FakeAgent(value, cache_dir=tmp_path / "cache")
    with pytest.raises(ValueError, match="sandbox"):
        pipeline.run_stage("author_task", "fake", tmp_path / "out", agent=agent, goal="test", dry_run=True)
    assert not list((tmp_path / "out").glob("candidate-*"))
    assert not json.loads((tmp_path / "out/result.json").read_text())["passed"]


def test_atomic_path_validation_before_write(pipeline, tmp_path):
    value = bundle()
    value["files"]["disallowed.json"] = "{}"
    with pytest.raises(ValueError):
        pipeline.write_bundle(value, tmp_path, lambda p: p == "task.json")
    assert not list(tmp_path.iterdir())


def test_game_dry_run_uses_real_luau_and_parity_without_touching_sources(pipeline, tmp_path):
    source = pipeline.ROOT / pipeline.GAME_PATH
    original = source.read_bytes()
    context = pipeline.game_context(pipeline.load_task("place_on_cell"),
        pipeline.read_json(pipeline.ROOT / "config/physics/so101_bench.json"),
        pipeline.read_json(pipeline.ROOT / "config/camera.json"))
    files = pipeline.game_sources(context)
    result = pipeline.run_stage("build_game", "fake", tmp_path, dry_run=True, context=context, source_files=files)
    assert result["passed"] and result["verification"]["parity"]["pass"]
    assert result["source_sha256"] == pipeline.source_hashes(files)
    assert all(c["passed"] for c in result["verification"]["checks"])
    assert source.read_bytes() == original
    workspace = Path(result["workspace"])
    assert (workspace / "roblox/default.project.json").is_file()
    assert "Teach the robot:" in (workspace / pipeline.GAME_PATH).read_text()


def test_game_config_cannot_remove_required_fields(pipeline, tmp_path):
    value = bundle()
    value["files"] = {pipeline.GAME_PATH: "{}"}
    with pytest.raises(ValueError, match="required"):
        pipeline.verify_game(value, {}, tmp_path)


def test_sandbox_cannot_replace_or_omit_oracle_hashes(pipeline, tmp_path):
    value = pipeline.fixture_task_bundle()
    files = {"tests/oracle.py": "assert actual == expected\n"}
    value["source_sha256"] = {"tests/oracle.py": "altered"}
    agent = FakeAgent(value, cache_dir=tmp_path / "cache", source_files=files)
    with pytest.raises(ValueError, match="immutable-source"):
        pipeline.run_stage("author_task", "fake", tmp_path / "out", agent=agent, goal="test",
                           dry_run=True, source_files=files)
    assert not list((tmp_path / "out").glob("candidate-*"))


def test_twin_evidence_must_match_all_inputs(pipeline, tmp_path):
    import hashlib
    task = pipeline.load_task("place_on_cell")
    context = pipeline.game_context(task, {"kp": 1}, {"K": []})
    report = tmp_path / "held-out.json"
    report.write_text(json.dumps({"passed": True, "held_out": True}), encoding="utf-8")
    evidence = {"schema": "r2p_twin_verification/1", "passed": True, "input_sha256": context["input_sha256"],
                "evidence": [{"path": str(report), "sha256": hashlib.sha256(report.read_bytes()).hexdigest()}]}
    assert pipeline.game_context(task, {"kp": 1}, {"K": []}, evidence)["verified"]
    with pytest.raises(ValueError, match="hash"):
        pipeline.game_context(task, {"kp": 2}, {"K": []}, evidence)
    report.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        pipeline.game_context(task, {"kp": 1}, {"K": []}, evidence)
