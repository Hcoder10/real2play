"""Lab chain boundaries; all model, receipt and arm endpoints are offline fakes."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from r2p import gemini
from r2p.frames import ROOT
from r2p.orchestrator import LAB_ORDER, Orchestrator, WaitingForOwner, digest
from r2p.orchestrator_client import RobustGemini, StatelessController, first_object
from r2p.orchestrator_lab import LabTools, file_hash, lab_revision


def decision(name, **args):
    return {"tool": name, "args": args, "why": "Run the next evidence-gated stage.",
            "done_when": [{"metric": "deploy_real.attempts", "op": "gte", "value": 1}]}


@pytest.fixture
def runner(tmp_path):
    for name in ("config/camera.json", "config/joint_map.json", "config/physics/so101_bench.json"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    lab = tmp_path / "runs/lab/fake"
    (lab / "sim").mkdir(parents=True)
    shutil.copyfile(ROOT / "config/physics/so101_bench.json", lab / "sim/physics.json")
    (lab / "sim/scene.xml").write_text("<mujoco/>", encoding="utf-8")
    held = {"synthetic": True, "episodes": ["held-a", "held-b"], "result": {
        "sealed": True, "improved": True, "revision": lab_revision(lab / "sim"),
        "baseline": {"joint_rms_deg": 2., "jaw_path_rms_mm": 5.},
        "current": {"joint_rms_deg": 1., "jaw_path_rms_mm": 3.}}}
    gemini.atomic_json(lab / "heldout.json", held)
    return Orchestrator(root=tmp_path, run="chain", dry_run=True, approve="owner-authorized",
                        options={"lab_run": "fake", "wait_seconds": 0})


def test_lab_order_and_evidence_without_old_experiments(runner):
    assert runner.order == LAB_ORDER
    result = runner.execute(decision("twin_gate"))
    assert result["metrics"]["passed"] == 1
    assert result["metrics"]["current_jaw_path_rms_mm"] == 3
    assert runner.backend.frozen["physics"] == file_hash(runner.backend.lab / "sim/physics.json")
    assert runner.backend.frozen["scene"] == file_hash(runner.backend.lab / "sim/scene.xml")
    with pytest.raises(ValueError, match="disabled"):
        runner.execute(decision("active_twin"))


@pytest.mark.parametrize("key,value", [("sealed", False), ("improved", False), ("improved", "true")])
def test_lab_gate_requires_actual_booleans(runner, key, value):
    path = runner.backend.lab / "heldout.json"
    held = json.loads(path.read_text())
    held["result"][key] = value
    gemini.atomic_json(path, held)
    assert runner.execute(decision("twin_gate"))["metrics"]["passed"] == 0


def test_modified_scoring_revision_cannot_pass(runner):
    (runner.backend.lab / "sim/scene.xml").write_text("<mujoco model='changed'/>")
    assert runner.execute(decision("twin_gate"))["metrics"]["passed"] == 0


@pytest.mark.parametrize("snapshot", [False, True])
def test_changed_lab_or_snapshot_refuses_downstream(runner, snapshot):
    runner.execute(decision("twin_gate"))
    directory = Path(runner.backend.frozen["snapshot"]) if snapshot else runner.backend.lab / "sim"
    (directory / "scene.xml").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        runner.backend.check_frozen()


def test_synthetic_lab_never_authorizes_live_run(runner):
    runner.backend.dry = runner.dry_run = False
    with pytest.raises(ValueError, match="Synthetic"):
        runner.execute(decision("twin_gate"))


def test_owner_authorized_is_logged_without_prompt(runner):
    runner.dry_run = False
    runner.prompt = lambda _: pytest.fail("Owner authorization must not prompt")
    runner.approval("build_game", {})
    rows = [json.loads(s) for s in (runner.out / "ledger.jsonl").read_text().splitlines()]
    assert rows[-1]["source"] == "owner-authorized" and rows[-1]["approved"]
    assert "session" in rows[-1]["authorization"]


def test_provisional_build_allowed_but_deploy_blocked(runner, monkeypatch):
    runner.state["results"]["twin_gate"] = {"ok": True, "metrics": {"passed": 0}}
    calls = []
    monkeypatch.setattr(runner.backend, "execute", lambda name, args: calls.append(name) or {"metrics": {"passed": 1}})
    runner.execute(decision("build_game"))
    assert calls == ["build_game"]
    runner.state["results"]["replay_and_train"] = {"ok": True, "metrics": {"policy_qualified": 0}}
    with pytest.raises(ValueError, match="sealed, improved"):
        runner.execute(decision("deploy_real"))


def test_provisional_game_has_no_verification_claim(runner, monkeypatch):
    from r2p import orchestrator_lab
    from scripts.argon import _managed
    path = runner.backend.lab / "heldout.json"
    held = json.loads(path.read_text())
    held["result"]["improved"] = False
    gemini.atomic_json(path, held)
    runner.execute(decision("twin_gate"))
    contexts = []
    def run_stage(*a, **kw):
        contexts.append(kw["context"])
        return {"passed": True, "verification": {"checks": [True], "parity": {"pass": True}},
                "record": {"interaction_id": "fake", "agent": "fake", "latency": 0}}
    helper = SimpleNamespace(game_context=_managed.game_context, game_sources=lambda _: {},
        game_prompt=lambda *a: "fake", run_stage=run_stage)
    monkeypatch.setattr(orchestrator_lab, "managed_helpers", lambda _: helper)
    result = runner.execute(decision("build_game"))
    assert not contexts[0]["verified"] and contexts[0]["verification"] is None
    assert result["metrics"]["provisional"] == 1 and "PROVISIONAL" in result["label"]


def test_first_object_and_stateless_error_feedback(tmp_path):
    requests = []
    responses = ["not JSON", "```json\n" + json.dumps(decision("twin_gate")) + "\n``` explanation"]
    def transport(**request):
        requests.append(request)
        return responses.pop(0)
    client = StatelessController(tmp_path, transport=transport, retries=0, sleep=lambda _: None)
    result, record = client.decide({"order": LAB_ORDER}, {"ok": False, "error": "previous stage failed"})
    assert result["tool"] == "twin_gate" and record["stateless"]
    assert len(requests) == 2
    assert all("previous_interaction_id" not in r["config"] for r in requests)
    assert "Previous reply failed" in str(requests[1]["parts"])
    assert "previous stage failed" in str(requests[1]["parts"])
    with pytest.raises(gemini.JSONError):
        first_object('{"x": NaN}')


def test_stateless_cache_only_and_quota_are_not_retried(tmp_path, monkeypatch):
    value = decision("twin_gate")
    client = StatelessController(tmp_path, transport=lambda **kw: json.dumps(value), retries=0)
    state = {"order": LAB_ORDER}
    assert client.decide(state)[0] == value
    monkeypatch.setenv("R2P_CACHE_ONLY", "1")
    cached = StatelessController(tmp_path, transport=lambda **kw: pytest.fail("network"))
    assert cached.decide(state)[1]["cached"]
    with pytest.raises(gemini.CacheMiss):
        cached.decide({"order": LAB_ORDER, "changed": True})
    monkeypatch.delenv("R2P_CACHE_ONLY")
    class Quota(Exception):
        status_code = 429
    calls = []
    def transport(**kw):
        calls.append(kw)
        raise Quota("per day limit 0")
    quota = RobustGemini(tmp_path / "quota", transport=transport, retries=0, sleep=lambda _: pytest.fail("retry"))
    with pytest.raises(gemini.QuotaError):
        quota.call(["x"], tag="orchestrator:plan", schema={"type": "object"})
    assert len(calls) == 1


def test_filter_block_retry_uses_plain_language_preface(tmp_path):
    calls = []
    def transport(**request):
        calls.append(request)
        if len(calls) == 1:
            raise RuntimeError("blocked by filter")
        return '{"ok": true} trailing prose'
    client = RobustGemini(tmp_path, transport=transport, retries=0, sleep=lambda _: None)
    assert json.loads(client.call(["x"], tag="orchestrator:plan", schema={"type": "object"})[0])["ok"]
    assert len(calls) == 2 and "Benign robotics" in str(calls[1]["parts"])


def test_training_request_has_recipe_paths_and_never_launches_gpu(runner, monkeypatch):
    from r2p.pipeline import game_loop, collection
    tool = runner.backend
    runner.execute(decision("twin_gate"))
    tool.train_plan()
    tool.dry = runner.dry_run = False
    tool.play = runner.out / "play"
    def ingest(source, output, task, physics):
        assert physics == tool.physics
        output.mkdir(parents=True)
        (output / "game.jsonl").write_text("game")
        return {"counts": {"relabeled": 1}, "traces": []}
    def collect(task, n, output, **kw):
        assert kw["workers"] == 1 and kw["physics"] == tool.physics
        output.mkdir()
        (output / "expert.jsonl").write_text("expert")
    monkeypatch.setattr(game_loop, "ingest", ingest)
    monkeypatch.setattr(collection, "collect", collect)
    with pytest.raises(WaitingForOwner):
        tool.replay_and_train()
    request = json.loads((runner.out / "train_request.json").read_text())
    assert request["physics_path"] == str(tool.physics_path)
    assert len(request["trace_dirs"]) == 2 and len(request["traces"]) == 2
    assert request["recipe"][1:3] == ["-m", "r2p.policy.train"]
    assert "--no-augmentation" in request["recipe"]
    assert request["recipe"][request["recipe"].index("--device")+1] == "cuda:0"
    assert "LANDLORD_MEMORY_FRACTION" in request["instructions"]
    with pytest.raises(WaitingForOwner, match="refusing to submit a replacement"):
        tool.replay_and_train()
    assert json.loads((runner.out / "train_request.json").read_text()) == request


@pytest.mark.parametrize("qualified,preflight_passed", [(False, False), (True, False), (True, True)])
def test_live_deploy_routes_one_argon_turn_and_never_retries(runner, monkeypatch, qualified, preflight_passed):
    from r2p import ttt
    from r2p import orchestrator_lab
    from r2p.orchestrator_tools import LocalTools
    tool = runner.backend
    runner.execute(decision("twin_gate"))
    tool.dry = runner.dry_run = False
    runner.state["results"]["replay_and_train"] = {"ok": True, "metrics": {"policy_qualified": int(qualified)}}
    calls = []
    frame = object()
    monkeypatch.setattr(tool, "read_frame", lambda: frame)
    chosen = {"board": ["X"]+[""]*8, "move": "r1c1", "source": "gemini", "why": "Take center"}
    monkeypatch.setattr(ttt, "decide", lambda *a, **kw: deepcopy(chosen))
    monkeypatch.setattr(tool, "supply_caps", lambda *a: [[.17, -.12]])
    monkeypatch.setattr(tool, "policy_preflight", lambda *a: {"passed": preflight_passed, "motion": False})
    script = SimpleNamespace(place=lambda *a: calls.append(("ik", a)) or {"grasped": True})
    monkeypatch.setattr(orchestrator_lab, "load_script", lambda *a: script)
    monkeypatch.setattr(LocalTools, "deploy_real", lambda self, **kw: calls.append(("policy", kw)) or {
        "metrics": {"attempts": 1, "success": 1}, "outcome": "fake"})
    tool.export_hashes = {}
    setup = runner.out / "setup.json"
    setup.write_text("{}")
    runner.options["perception_setup"] = setup
    result = tool.deploy_real(cell="r2c2")
    use_policy = qualified and preflight_passed
    assert result["executor"] == ("learned_policy" if use_policy else "calibrated_ik")
    assert calls[0][0] == ("policy" if use_policy else "ik")
    assert "r1c1" in str(calls) and "r2c2" not in str(calls)
    if not use_policy:
        assert result["metrics"]["success_verified"] == 0
    with pytest.raises(WaitingForOwner, match="already started"):
        tool.deploy_real()
    assert len(calls) == 1


def test_missing_policy_setup_fails_preflight_without_subprocess(runner, monkeypatch):
    from r2p import orchestrator_lab
    monkeypatch.setattr(orchestrator_lab.subprocess, "run", lambda *a, **kw: pytest.fail("No process needed"))
    assert not runner.backend.policy_preflight("r1c1", runner.out)["passed"]


def test_live_finish_records_ik_turn_without_claiming_policy_success(runner):
    runner.dry_run = False
    runner.state["results"] = {name: {"ok": True, "metrics": {"passed": 1}} for name in LAB_ORDER}
    runner.state["results"]["collect_play"]["metrics"].update(traces=1, human=0, headless=1)
    runner.state["results"]["replay_and_train"]["metrics"].update(replayed=1, eval_n=12, clean=0, real=0, improved=0)
    runner.state["results"]["deploy_real"].update(executor="calibrated_ik", metrics={"attempts": 1, "success": 0})
    assert runner.numeric_gates()[0]
    runner.state["results"]["deploy_real"]["executor"] = "learned_policy"
    assert not runner.numeric_gates()[0]


def test_collection_keeps_humans_and_headless_separate(runner, monkeypatch):
    from r2p import traces
    from r2p.pipeline import game_player
    runner.execute(decision("twin_gate"))
    tool = runner.backend
    tool.dry = False
    source = runner.root / "roblox/out/traces"
    source.mkdir(parents=True)
    for name in ("human", "bot", "fake_human", "fixture", "broken"):
        (source / (name + ".jsonl")).write_text(name)
    def read(path):
        name = Path(path).stem
        if name == "broken":
            raise ValueError("partial upload")
        return {"source": "bot" if name == "bot" else "human", "task_name": runner.task["name"],
                "headless": name == "fake_human", "test_only": name == "fixture"}, [{"tick": 1}]
    monkeypatch.setattr(traces, "read_trace", read)
    def collect(task, path, **kw):
        assert kw["physics"] == tool.physics
        path.mkdir(parents=True)
        result = path / "headless.jsonl"
        result.write_text("headless")
        return [result]
    monkeypatch.setattr(game_player, "collect_players", collect)
    result = tool.collect_play()
    assert result["metrics"] == {"traces": 2, "human": 1, "headless": 1, "rejected": 4}
    sources = json.loads(Path(result["sources"]).read_text())
    assert "SkillOperator" in sources["simulation"]


@pytest.mark.parametrize("incumbent,successes,expected", [(False, 12, 0), (True, 3, 0), (True, 12, 1)])
def test_receipt_is_evaluated_on_frozen_physics_before_policy_route(runner, monkeypatch, incumbent, successes, expected):
    from r2p import policy
    # Load modules that bind policy.load before replacing it, so the fake cannot
    # leak through a lazy module-level import into a later CPU integration test.
    from r2p.policy import eval as policy_eval  # noqa: F401
    from r2p.pipeline import evaluation
    from r2p.pipeline.storage import digest as plan_digest
    tool = runner.backend
    runner.execute(decision("twin_gate"))
    tool.training = {"eval_n": 12}
    folder = runner.out / "evaluation"
    folder.mkdir()
    export, before = folder / "export", folder / "incumbent"
    config = {"task": tool.task["name"], "task_spec": tool.task, "physics": tool.physics,
              "action": "absolute_q", "horizon": 15, "hz": 30, "obs_frame": "spec_footprint",
              "start_q": [0.]*6, "obs_layout": [], "obs_const": [], "eval": {"clean": {"n": 12}}}
    for path in (export, before):
        gemini.atomic_json(path / "config.json", config)
    monkeypatch.setattr(policy, "load", lambda path: SimpleNamespace(config=config))
    plans = []
    def evaluate_pair(a, b, task, plan, **kwargs):
        plans.append(plan)
        def report(path, count):
            return {"policy": str(path), "plan_hash": plan_digest(plan), "results": {
                noise: [{"scene_id": row["id"], "noise_seed": row["noise_seed"], "success": i < count}
                        for i, row in enumerate(plan["episodes"])] for noise in ("clean", "real")}}
        return report(a, successes), report(b, 0)
    monkeypatch.setattr(evaluation, "evaluate_pair", evaluate_pair)
    receipt = {"export": str(export), **({"incumbent": str(before)} if incumbent else {})}
    result = tool.evaluate_receipt(receipt, folder)
    assert result["policy_qualified"] == expected
    assert result["eval_n"] == 12 and plans[0]["physics"] == tool.physics
    assert len(plans[0]["episodes"]) == 12
