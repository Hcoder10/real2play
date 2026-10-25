"""Offline acceptance for the real step-loop demo wiring."""
import json
import os

import pytest

from dashboard.demo import DemoData
from tests.app.test_demo import adapter, lab_run, write


def step_run(root, name="step_01", count=8, complete=True):
    run = root / "runs/argon_step" / name
    run.mkdir(parents=True)
    rows = []
    for turn in range(1, count + 1):
        action = {"done": True} if complete and turn == count else {"gripper": 0} if turn == 2 else {"move": [.1, -.02, .03]}
        rows.append({"turn": turn, "t": turn, "obs": {}, "reply": {"thought": f"Thought {turn}", "action": action}})
        (run / f"t{turn:02}_real_vs_sim.jpg").write_bytes(b"comparison")
        (run / f"t{turn:02}_wrist.jpg").write_bytes(b"wrist")
    (run / "ledger.jsonl").write_text("\n".join(map(json.dumps, rows)) + '\n{"partial":', encoding="utf-8")
    write(run / "sim.json", {"objects": [{"id": "cap"}], "arm": {
        "z_offset_mm": -1.5, "xy_offset_mm": [10, 11], "notes": "Successfully picked cap2 and placed it in r1c1."}})
    return run


def policy(root, **overrides):
    value = {"clean": {"success": 29 / 30, "n": 30}, "realx0.5": {"success": .8, "n": 30},
             "real": {"success": 13 / 30, "n": 30}, **overrides}
    write(root / "exports/wide_v3/config.json", {"eval": value})


def test_step_selection_latest_thought_six_actions_numeric_slides_and_sim(tmp_path):
    old = lab_run(tmp_path)
    os.utime(old / "status.json", (1, 1))
    run = step_run(tmp_path, count=12)
    result = DemoData(adapter(tmp_path)).lab()
    assert result["kind"] == "argon_step" and result["turn_count"] == 12
    assert result["thought"] == "Thought 12" and result["state"] == "done"
    assert [r["turn"] for r in result["slides"]] == list(range(1, 13))
    assert [r["turn"] for r in result["actions"]] == list(range(7, 13))
    assert result["actions"][-1]["label"] == "done"
    assert result["frames"]["comparison"] == result["slides"][-1]["comparison"]
    assert result["frames"]["wrist"] == result["slides"][-1]["wrist"]
    assert result["sim"]["objects"] == 1 and result["sim"]["z_mm"] == -1.5
    assert result["sim"]["xy_mm"] == [10, 11]
    assert result["placement"] == "Real pick-and-place into r1c1"
    assert not result["synthetic"]
    # Newer traditional labs still win; nested test fixtures never win discovery.
    os.utime(run / "ledger.jsonl", (1, 1))
    lab_run(tmp_path, "newer")
    step_run(tmp_path, "acceptance/nested")
    assert DemoData(adapter(tmp_path)).lab()["run"] == "newer"


def test_incomplete_step_missing_sim_and_stale_activity_are_not_done(tmp_path):
    run = step_run(tmp_path, count=3, complete=False)
    result = DemoData(adapter(tmp_path)).lab()
    assert result["state"] == "running" and result["placement"] is None
    assert result["actions"][1]["label"] == "gripper 0"
    (run / "sim.json").unlink()
    os.utime(run / "ledger.jsonl", (1, 1))
    result = DemoData(adapter(tmp_path)).snapshot()
    assert result["lab"]["state"] == "pending"
    assert result["lab"]["sim"]["objects"] is None
    assert next(s for s in result["chain"]["stages"] if s["name"] == "Twin")["state"] == "pending"


def test_real_artifact_chain_keeps_blocked_play_separate_from_existing_twin_eval(tmp_path):
    step_run(tmp_path)
    policy(tmp_path)
    write(tmp_path / "runs/after_step/step_01/game/build/result.json", {"passed": True, "published": False})
    write(tmp_path / "runs/after_step/step_01/loop/game_loop.json", {"status": "BLOCKED", "stage": "ingest", "error": "Out of memory"})
    write(tmp_path / "runs/argon_orchestrator/fake/status.json", {"synthetic": True, "status": "DONE"})
    write(tmp_path / "out/ttt/status.json", {"synthetic": True, "physical_placement": True, "status": "done"})
    result = DemoData(adapter(tmp_path)).snapshot()
    stages = {s["name"]: s for s in result["chain"]["stages"]}
    assert stages["Real"]["state"] == stages["Twin"]["state"] == stages["Game"]["state"] == "done"
    assert stages["Play"]["state"] == "pending" and "Out of memory" in stages["Play"]["detail"]
    assert stages["Replay"]["state"] == stages["Train"]["state"] == "done"
    assert "twin" in stages["Train"]["provenance"]
    assert stages["Deploy"]["state"] == "pending"
    assert result["policy"]["clean"] == pytest.approx(29 / 30)
    assert result["policy"]["half"] == .8 and result["policy"]["real"] == pytest.approx(13 / 30)
    assert result["policy"]["n"] == 30
    assert result["policy"]["message"] == "state policy, twin eval; real-arm deployment pending (bench-profile expert)"


def test_missing_eval_is_unknown_and_unpassed_build_is_pending(tmp_path):
    policy(tmp_path, clean={"success": None}, real={"success": 1.2}, **{"realx0.5": {"success": True}})
    write(tmp_path / "runs/after_step/step_01/game/result.json", {"passed": False})
    result = DemoData(adapter(tmp_path)).snapshot()
    assert not result["policy"]["available"]
    assert result["policy"]["clean"] is result["policy"]["half"] is result["policy"]["real"] is None
    assert next(s for s in result["chain"]["stages"] if s["name"] == "Game")["state"] == "pending"


def test_after_step_does_not_mix_runs(tmp_path):
    old = tmp_path / "runs/after_step/old/game/result.json"
    write(old, {"passed": True})
    os.utime(old, (1, 1))
    write(tmp_path / "runs/after_step/new/loop/game_loop.json", {"status": "RUNNING", "stage": "ingest"})
    result = DemoData(adapter(tmp_path)).snapshot()
    assert result["game"]["after_step"]["run"] == "new"
    stages = {s["name"]: s for s in result["chain"]["stages"]}
    assert stages["Game"]["state"] == "pending" and stages["Play"]["state"] == "running"


def test_deploy_requires_real_physical_placement(tmp_path):
    write(tmp_path / "out/ttt/status.json", {"source": "real", "status": "done", "physical_placement": True})
    result = DemoData(adapter(tmp_path)).snapshot()
    assert next(s for s in result["chain"]["stages"] if s["name"] == "Deploy")["state"] == "done"
