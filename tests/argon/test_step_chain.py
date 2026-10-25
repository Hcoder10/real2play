"""W's complete pass: all SSH/model/hardware seams stay offline."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import zipfile

import pytest

from r2p import gemini
from r2p.frames import ROOT
from r2p.orchestrator import Orchestrator, STEP_ORDER, WaitingForOwner
from r2p.orchestrator_client import StatelessController
from r2p.orchestrator_remote import B200, LEASE, RENTAL_END, training_recipe, unpack
from r2p.orchestrator_step import physics_from_sim, snapshot_humans


def decision(name, **args):
    return {"tool": name, "args": args, "why": "Run the next stage from measured evidence.",
            "done_when": [{"metric": "deploy.attempts", "op": "gte", "value": 1}]}


@pytest.fixture
def runner(tmp_path, monkeypatch):
    for name in ("config/physics/so101_bench.json", "config/bench.json", "config/joint_map.json",
                 "tasks/place_on_cell/task.json", "scripts/robot/K_STREAM_PROFILE.json"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    from r2p.robot import config
    monkeypatch.setattr(config, "config_path", lambda value: tmp_path / value)
    runner = Orchestrator(root=tmp_path, run="w-test", dry_run=True, approve="owner-authorized",
                          options={"step_run": "existing", "wait_seconds": 1})
    def command(argv, log, **kwargs):
        assert argv[1] == "scripts/argon/build_game.py"
        assert "--dry-run" in argv and "--allow-provisional-twin" in argv
        folder = Path(argv[argv.index("--out") + 1])
        gemini.atomic_json(folder / "result.json", {"passed": True, "record": {"agent": "fake-managed"}})
        return 0
    monkeypatch.setattr(runner.backend, "command", command)
    from r2p import orchestrator_step
    def git(argv, **kwargs):
        assert argv[:2] == ["git", "log"]
        return SimpleNamespace(stdout="" if "e8cd98b..HEAD" in argv else "e8cd98b old bench profile interface")
    monkeypatch.setattr(orchestrator_step.subprocess, "run", git)
    return runner


def through(runner, last="evaluate"):
    for name in STEP_ORDER:
        runner.execute(decision(name))
        if name == last:
            return


def test_complete_fake_ssh_mockrobot_chain(runner):
    result = runner.run()
    assert result["status"] == "DONE" and result["synthetic"]
    assert list(result["results"]) == STEP_ORDER
    deploy = result["results"]["deploy"]
    assert deploy["executor"] == "calibrated_ik" and deploy["transport"] == "MockRobot"
    assert deploy["metrics"]["attempts"] == 1 and deploy["metrics"]["success"] == 0
    assert result["results"]["collect_play"]["metrics"]["human"] == 0
    rows = [json.loads(line) for line in (runner.out / "ledger.jsonl").read_text().splitlines()]
    assert [r["stage"] for r in rows if r["kind"] == "fake_ssh"] == ["collect_play", "train", "evaluate"]
    assert (runner.root / "exports/w-test-policy/config.json").is_file()


def test_step_dependencies_and_closed_tool_set(runner):
    assert runner.order == STEP_ORDER
    with pytest.raises(ValueError, match="requires"):
        runner.execute(decision("deploy"))
    with pytest.raises(ValueError, match="Step mode"):
        runner.execute(decision("active_twin"))
    assert not runner.execute(decision("finish"))["ok"]


def test_owner_authorization_covers_every_stage(runner):
    runner.dry_run = False
    runner.prompt = lambda _: pytest.fail("Owner-authorized must not ask again")
    for name in STEP_ORDER:
        runner.approval(name, {})
    rows = [json.loads(line) for line in (runner.out / "ledger.jsonl").read_text().splitlines()]
    assert all(row["source"] == "owner-authorized" and row["approved"] for row in rows)


def test_provisional_build_cannot_publish(runner):
    runner.execute(decision("experiment_sim"))
    with pytest.raises(ValueError, match="provisional"):
        runner.execute(decision("build_game", publish=True))


def test_frozen_physics_change_blocks_downstream(runner):
    runner.execute(decision("experiment_sim"))
    runner.backend.physics_path.write_text("{}")
    with pytest.raises(ValueError, match="Frozen"):
        runner.execute(decision("build_game"))


def test_failed_rehearsal_is_valid_evaluation_for_ik(runner):
    through(runner)
    evaluation = runner.state["results"]["evaluate"]
    assert evaluation["ok"] and evaluation["metrics"]["rehearsal_goals"] == 0
    assert runner.execute(decision("deploy"))["executor"] == "calibrated_ik"
    assert runner.execute(decision("finish"))["ok"]


def test_exactly_one_deployment_even_in_dry_run(runner):
    through(runner, "deploy")
    with pytest.raises(WaitingForOwner, match="already started"):
        runner.backend.deploy()


def test_returned_export_cannot_change_after_rehearsal(runner):
    through(runner)
    (runner.backend.export / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="export changed"):
        runner.backend.deploy()
    assert not (runner.out / "deployment_started.json").exists()


@pytest.mark.parametrize("qualified", [False, True])
def test_live_routes_once_without_hardware(runner, monkeypatch, qualified):
    through(runner)
    backend = runner.backend
    backend.dry = False
    backend.live_inputs = {}
    backend.evaluation.update(policy_qualified=qualified, reason="test rehearsal")
    calls = []
    def command(argv, log, **kwargs):
        calls.append(argv)
        path = Path(argv[argv.index("--record") + 1]) / "one/trace.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("fake trace parsed by fake reader")
        return 1  # A failed real turn still must never trigger an IK retry.
    monkeypatch.setattr(backend, "command", command)
    monkeypatch.setattr("r2p.traces.read_trace", lambda _: ({"success": False, "fail_reason": "timeout"}, [{}]))
    def ik(folder):
        calls.append("ik")
        return {"metrics": {"attempts": 1, "success": 0}, "executor": "calibrated_ik"}
    monkeypatch.setattr(backend, "ik_turn", ik)
    result = backend.deploy()
    assert result["executor"] == ("learned_policy" if qualified else "calibrated_ik")
    assert len(calls) == 1
    if qualified:
        argv = calls[0]
        for flag in ("--argon-perception", "--transport", "--bench-profile", "--cap", "--real", "--allow-unverified"):
            assert flag in argv
        assert "--cell" not in argv  # Argon, not the orchestrator, chooses the cell.
    with pytest.raises(WaitingForOwner):
        backend.deploy()


def test_partial_policy_attempt_never_falls_back(runner, monkeypatch):
    through(runner)
    backend = runner.backend
    backend.dry = False
    backend.live_inputs = {}
    backend.evaluation["policy_qualified"] = True
    monkeypatch.setattr(backend, "command", lambda *a, **k: 1)
    monkeypatch.setattr(backend, "ik_turn", lambda *a: pytest.fail("No fallback after possible motion"))
    with pytest.raises(WaitingForOwner, match="no IK retry"):
        backend.deploy()
    with pytest.raises(WaitingForOwner, match="already started"):
        backend.deploy()


def test_physics_conversion_preserves_frame_and_ignores_nonfinite_patches():
    base = json.loads((ROOT / "config/physics/so101_bench.json").read_text())
    original = deepcopy(base)
    sim = {"physics": {"shoulder_lift": {"kp": 25., "kv": float("nan"), "bad": True}},
           "objects": [{"id": "cap1", "mass_kg": .003, "friction": .7}], "arm": {"xy_offset_mm": [10, 11]}}
    result = physics_from_sim(base, sim, "recorded")
    assert result["joints"]["shoulder_lift"]["kp"] == 25
    assert result["joints"]["shoulder_lift"]["kv"] == original["joints"]["shoulder_lift"]["kv"]
    assert result["object"] == {"mass_kg": .003, "friction": .7}
    assert base == original and result["gravity"] == original["gravity"]


def test_human_snapshot_excludes_synthetic_and_duplicates(tmp_path, monkeypatch):
    source, target = tmp_path / "receiver", tmp_path / "human"
    source.mkdir()
    for name, value in (("human", "real"), ("duplicate", "real"), ("bot", "bot"), ("fake", "fake")):
        (source / (name + ".jsonl")).write_text(value)
    def read(path):
        name = path.stem
        return {"source": "bot" if name == "bot" else "human", "task_name": "place_on_cell",
                "synthetic": name == "fake"}, [{}]
    monkeypatch.setattr("r2p.traces.read_trace", read)
    report = snapshot_humans(source, target, {"name": "place_on_cell"})
    assert report["human"] == 1 and len(report["rejected"]) == 3
    assert len(list(target.glob("*.jsonl"))) == 1


@pytest.mark.parametrize("member", ["../escape", "/escape", "C:/escape", "..\\escape"])
def test_remote_archive_cannot_escape(tmp_path, member):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(member, "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        unpack(archive, tmp_path / "results")


def test_p2_recipe_has_all_required_parameters():
    argv = training_recipe({"name": "w-policy"})
    for flag, value in {"--steps": "60000", "--batch-size": "256", "--width": "512", "--blocks": "3",
                        "--device": "cuda:0", "--memory-fraction": "0.949", "--eval-n": "30"}.items():
        assert argv[argv.index(flag)+1] == value
    assert argv[argv.index("--seeds")+1:argv.index("--seeds")+4] == ["0", "1", "2"]
    assert "--no-augmentation" in argv


def test_b200_launch_prefix_and_no_duplicate(tmp_path, monkeypatch):
    events, calls = [], []
    runner = SimpleNamespace(root=tmp_path, out=tmp_path, options={"wait_seconds": 60},
                             event=lambda *a, **k: events.append((a, k)))
    remote = B200(runner)
    remote.ready = True
    from r2p import orchestrator_remote as module
    monkeypatch.setattr(module.time, "time", lambda: RENTAL_END-600)
    def popen(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(poll=lambda: 0, returncode=0)
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    def pull(**kwargs):
        gemini.atomic_json(tmp_path / "remote/train.json", {"ok": True})
    monkeypatch.setattr(remote, "pull", pull)
    assert remote.run("train")["ok"]
    command = calls[0][-1]
    for key, value in LEASE.items():
        assert f"{key}={value}" in command
    assert "r2p.orchestrator_remote" in command and "--stage train" in command
    with pytest.raises(WaitingForOwner, match="already launched"):
        remote.run("train")
    assert len(calls) == 1


def test_remote_rehearsal_failure_routes_ik(tmp_path, monkeypatch):
    from r2p import orchestrator_remote as module
    out = tmp_path / "results"
    gemini.atomic_json(out / "export/config.json", {"eval": {"clean": {"n": 30, "success": .9},
                                                            "real": {"n": 30, "success": .8}}})
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(module.subprocess, "run", run)
    report = module.evaluate({"cap": 2}, tmp_path / "inputs", out)
    assert report["ok"] and not report["policy_qualified"] and report["rehearsal_n"] == 0
    assert report["eval_n"] == 30 and report["real"] == .8
    assert "--real" not in calls[0] and "--transport" in calls[0]


def test_remote_rehearsal_pass_requires_all_cells(tmp_path, monkeypatch):
    from r2p import orchestrator_remote as module
    out = tmp_path / "results"
    gemini.atomic_json(out / "export/config.json", {"eval": {"clean": {"n": 30, "success": .9},
                                                            "real": {"n": 30, "success": .8}}})
    def run(argv, **kwargs):
        gemini.atomic_json(Path(argv[argv.index("--out")+1]), {"n": 9, "successes": 9})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess, "run", run)
    assert module.evaluate({"cap": 0}, tmp_path / "inputs", out)["policy_qualified"]


def test_step_controller_is_stateless_low_thinking(monkeypatch):
    controller = StatelessController(max_requests=1)
    calls = []
    def call(parts, **kwargs):
        calls.append((parts, kwargs))
        return json.dumps(decision("experiment_sim")), {"served_by": "fake-argon"}
    monkeypatch.setattr(controller, "call", call)
    selected, _ = controller.decide({"step_run": "step_01", "order": STEP_ORDER})
    assert selected["tool"] == "experiment_sim"
    assert calls[0][1]["thinking"] == "low"
    assert "there is no lead training receipt" in calls[0][0][0]
    assert "previous_interaction_id" not in calls[0][1]


def test_remote_training_refuses_unreplayed_play(tmp_path, monkeypatch):
    from r2p import orchestrator_remote as module
    monkeypatch.setattr("r2p.pipeline.game_loop.ingest", lambda *a: {"counts": {"relabeled": 0}})
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("No training without replay"))
    with pytest.raises(ValueError, match="No play trace"):
        module.train({"task": {}, "physics": {}}, tmp_path / "inputs", tmp_path / "out")


@pytest.mark.parametrize("bench", [False, True])
def test_remote_training_uses_matching_start_and_relabels_play(tmp_path, monkeypatch, bench):
    from r2p import orchestrator_remote as module
    from r2p.tasks import load_task, sample_scene
    import numpy as np
    task = load_task("place_on_cell")
    profile = {"start_q": [0]*6, "model_limits": [[-1, 1]]*6}
    out = tmp_path / "out"
    replay_path = out / "replayed/play.jsonl"
    replay_path.parent.mkdir(parents=True)
    replay_path.write_text("source trace")
    scene = sample_scene(task, np.random.default_rng(500000))
    monkeypatch.setattr("r2p.pipeline.game_loop.ingest", lambda *a: {"counts": {"relabeled": 1}})
    seen_twins, seen_experts, recipes = [], [], []
    class Twin:
        def __init__(self, scene, **kwargs):
            seen_twins.append(kwargs)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    class Expert:
        def __init__(self, task, **kwargs):
            seen_experts.append(kwargs)
        def run(self, twin, **kwargs):
            return {"meta": {"success": True}, "ticks": []}
    monkeypatch.setattr("r2p.sim.TaskTwin", Twin)
    monkeypatch.setattr("r2p.sim.expert.Expert", Expert)
    monkeypatch.setattr("r2p.traces.read_trace", lambda _: ({"scene": scene, "seed": 500000, "source": "human"}, []))
    monkeypatch.setattr("r2p.traces.write_trace", lambda path, meta, ticks: Path(path).write_text(json.dumps(meta)))
    monkeypatch.setattr(module.subprocess, "run", lambda argv, **kwargs: recipes.append(argv))
    request = {"name": "test", "task": task, "physics": {}, "profile": profile,
               "bench_expert_committed": bench, "experts": 2, "expert_commit": "fixture commit"}
    report = module.train(request, tmp_path / "inputs", out)
    assert report["ok"] and report["replayed"] == 1
    assert len(seen_experts) == (3 if bench else 2)
    assert all(row["deployment"] == (profile if bench else None) for row in seen_experts)
    assert all(row["joint_limits"] == (profile["model_limits"] if bench else None) for row in seen_twins)
    if bench:
        relabeled = json.loads((out / "mixed/play_play.jsonl").read_text())
        assert relabeled["play_lineage"]["source"] == "human"
        assert "re-demonstration" in relabeled["play_lineage"]["method"]
    assert recipes == [training_recipe(request)]
