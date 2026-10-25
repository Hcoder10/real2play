"""Controller boundaries, approvals and genuine offline end-to-end evidence."""
import json
import time
from pathlib import Path

import pytest

from r2p import gemini
from r2p.orchestrator import (FakeController, InteractionsController, Orchestrator, ORDER,
                              validate_decision)
from r2p.orchestrator_tools import gate_metrics


def decision(name="measure_scene", args=None):
    return {"tool": name, "args": args or {}, "why": "Measure before choosing an action.",
            "done_when": [{"metric": "deploy_real.attempts", "op": "gte", "value": 1}]}


class Backend:
    def __init__(self):
        self.calls = []

    def action(self, name, args):
        return f"EXACT {name} {args}"

    def execute(self, name, args):
        self.calls.append((name, args))
        return {"metrics": {"passed": 1, "traces": 1, "replayed": 1, "eval_n": 1, "attempts": 1}}


@pytest.mark.parametrize("value", [decision("shell"), decision("active_twin", {"budget": 0}),
    decision("active_twin", {"budget": True}), decision("build_game", {"publish": "yes"}),
    decision("measure_scene", {"command": "anything"}), {**decision(), "extra": 1},
    {**decision(), "done_when": []}, {**decision(), "done_when": [{"metric": "x", "op": "gte", "value": float("nan")}]}])
def test_reject_model_arguments(value):
    with pytest.raises(ValueError):
        validate_decision(value)


def test_live_auto_refused_before_creation(tmp_path):
    with pytest.raises(ValueError, match="only"):
        Orchestrator(root=tmp_path, approve="auto", backend=Backend())
    assert not (tmp_path / "runs").exists()


def test_decline_records_exact_action_without_execution(tmp_path):
    backend = Backend()
    runner = Orchestrator(root=tmp_path, backend=backend, prompt=lambda question: "no")
    with pytest.raises(PermissionError):
        runner.execute(decision("author_task"))
    assert not backend.calls
    ledger = [json.loads(line) for line in (runner.out / "ledger.jsonl").read_text().splitlines()]
    approval = next(row for row in ledger if row["kind"] == "approval")
    assert approval["action"] == "EXACT author_task {}" and approval["approved"] is False


def test_dependency_and_finish_cannot_be_bypassed(tmp_path):
    runner = Orchestrator(root=tmp_path, dry_run=True, backend=Backend())
    with pytest.raises(ValueError, match="requires"):
        runner.execute(decision("deploy_real"))
    assert runner.execute(decision("finish"))["ok"] is False
    assert runner.stop_met([{"metric": "absent", "op": "gte", "value": 0}]) is False


def test_rerunning_upstream_invalidates_descendant_evidence(tmp_path):
    runner = Orchestrator(root=tmp_path, dry_run=True, backend=Backend())
    for name in ORDER:
        runner.execute(decision(name))
    assert runner.numeric_gates()[0]
    runner.execute(decision())
    assert list(runner.state["results"]) == ["measure_scene"]
    assert not runner.numeric_gates()[0]


def test_failed_retry_also_invalidates_old_pass(tmp_path):
    backend = Backend()
    runner = Orchestrator(root=tmp_path, dry_run=True, backend=backend)
    for name in ORDER:
        runner.execute(decision(name))
    backend.execute = lambda *args: (_ for _ in ()).throw(ValueError("new measurement failed"))
    with pytest.raises(ValueError, match="measurement failed"):
        runner.execute(decision())
    assert not runner.state["results"]


def test_stateful_function_call_and_cache_only(tmp_path, monkeypatch):
    requests = []
    def transport(**request):
        requests.append(request)
        return {"id": f"interaction-{len(requests)}", "model": "gemini-4-argon", "steps": [{
            "type": "function_call", "id": f"call-{len(requests)}", "name": "measure_scene",
            "arguments": {k: v for k, v in decision().items() if k != "tool"}}]}
    controller = InteractionsController(tmp_path, transport=transport, retries=0, max_requests=2)
    first, record = controller.decide({"goal": "test"})
    controller.decide({"goal": "test"}, {"ok": True, "metrics": {"caps": 3}})
    assert requests[1]["previous_interaction_id"] == "interaction-1"
    assert requests[1]["input"][0]["call_id"] == "call-1"
    assert json.loads(requests[1]["input"][0]["result"])["metrics"]["caps"] == 3
    assert requests[0]["generation_config"]["thinking_level"] == "high"
    monkeypatch.setenv("R2P_CACHE_ONLY", "1")
    cached = InteractionsController(tmp_path, transport=lambda **kw: pytest.fail("network"))
    assert cached.decide({"goal": "test"})[0] == first
    assert cached.previous_id == "interaction-1"
    with pytest.raises(gemini.CacheMiss):
        cached.decide({"goal": "uncached"})


def test_quota_stops_without_fallback(tmp_path):
    class DailyQuota(Exception):
        status_code = 429
    controller = InteractionsController(tmp_path, transport=lambda **kw: (_ for _ in ()).throw(
        DailyQuota("per day quota limit 0")), retries=0)
    with pytest.raises(gemini.QuotaError):
        controller.decide({})


def test_plan_only_calls_no_tools(tmp_path):
    backend = Backend()
    runner = Orchestrator(root=tmp_path, dry_run=True, controller=FakeController(), backend=backend)
    assert runner.run(plan_only=True)["status"] == "PLANNED"
    assert backend.calls == []
    assert (runner.out / "plan.json").is_file()


def test_fail_closed_heldout_numeric_gate():
    row = {"split": "heldout", "joint_deg": .1, "jaw_mm": 1, "object_mm": 2, "iou": .9}
    metrics = {**row, "rows": [row], "objects": {"cap": {"iou": .9}}}
    assert gate_metrics(metrics)[0]
    assert not gate_metrics({**metrics, "jaw_mm": float("nan")})[0]
    assert not gate_metrics({**metrics, "rows": []})[0]
    assert not gate_metrics({**metrics, "rows": [{**row, "object_mm": 10}]})[0]
    assert not gate_metrics({**metrics, "objects": {}})[0]
    assert not gate_metrics({**metrics, "objects": {"cap": {"iou": .79}}})[0]
    assert not gate_metrics(metrics, {"objects": {"cap": {"iou": .99}}})[0]


def test_managed_bundles_reject_scope_and_false_tests(tmp_path):
    from r2p.orchestrator_managed import run_candidate, hashes
    from r2p.agents import FakeAgent
    files = {"input.json": "{}"}
    bundle = {"files": {"unauthorized.py": "print('bad')"}, "tests": [{"name": "test", "passed": True}],
              "report": "", "source_sha256": hashes(files)}
    agent = FakeAgent(bundle, cache_dir=tmp_path / "cache")
    with pytest.raises(ValueError, match="out-of-scope"):
        run_candidate("train_plan", files, "training.json", {}, tmp_path, dry=True,
                      instructions="", verify=lambda value: {}, agent=agent)


def test_offline_end_to_end_under_three_minutes(tmp_path, monkeypatch):
    from r2p.frames import ROOT
    from r2p.orchestrator_tools import LocalTools
    # Outputs are isolated; source assets/config are the shared, read-only repository.
    runner = Orchestrator(root=ROOT, run="n-test-" + tmp_path.name, dry_run=True, approve="auto")
    original = runner.out
    target = tmp_path / "run"
    original.rename(target)
    runner.out = target
    runner.backend = LocalTools(runner)
    monkeypatch.setattr(InteractionsController, "_client", lambda self: pytest.fail("No Google client in dry run"))
    from r2p.robot.io import RealRobot
    monkeypatch.setattr(RealRobot, "__init__", lambda *args, **kwargs: pytest.fail("No real robot"))
    started = time.monotonic()
    result = runner.run()
    assert time.monotonic() - started < 180
    assert result["status"] == "DONE" and result["synthetic"]
    assert result["metrics"]["collect_play.headless"] >= 1
    assert result["metrics"]["collect_play.human"] == 0
    assert result["metrics"]["replay_and_train.replayed"] >= 1
    assert result["metrics"]["deploy_real.ticks"] > 0
    assert {r["tool"] for r in result["managed"]} == {"analyze_experiments", "build_game", "train_plan"}
    assert all(r["synthetic"] for r in result["managed"])


def test_dashboard_reads_status_and_partial_ledger(tmp_path):
    from dashboard.argon import snapshot
    path = tmp_path / "runs/argon_orchestrator/a"
    path.mkdir(parents=True)
    gemini.atomic_json(path / "status.json", {"schema": "r2p_argon_orchestrator/1", "stage": "twin_gate", "synthetic": True})
    (path / "ledger.jsonl").write_text('{"kind":"decision"}\n{"incomplete":')
    result = snapshot(tmp_path)
    assert result["available"] and result["stage"] == "twin_gate"
    assert result["ledger"] == [{"kind": "decision"}]


def test_mcp_tool_retry_does_not_execute_twice(tmp_path):
    from scripts.argon.tools_mcp import ToolService
    backend = Backend()
    runner = Orchestrator(root=tmp_path, dry_run=True, backend=backend)
    service = ToolService(runner)
    value = {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {
        "name": "measure_scene", "arguments": {k: v for k, v in decision().items() if k != "tool"}}}
    assert service.dispatch(value) == service.dispatch(value)
    assert len(backend.calls) == 1
    assert "error" in service.dispatch({**value, "method": "ping"})
    tools = service.method("tools/list", {})["tools"]
    assert {row["name"] for row in tools} >= set(ORDER)


def test_mcp_http_auth_origin_and_schema(tmp_path):
    from scripts.argon.tools_mcp import make_server
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError
    import threading
    runner = Orchestrator(root=tmp_path, dry_run=True, backend=Backend())
    token = "n-test-token-" + "x" * 32
    server = make_server(runner, token, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/mcp"
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}).encode()
    try:
        with pytest.raises(HTTPError) as error:
            urlopen(Request(url, data=body), timeout=3)
        assert error.value.code == 401
        with pytest.raises(HTTPError) as error:
            urlopen(Request(url, data=body, headers={"Authorization": "Bearer " + token, "Origin": "https://untrusted.invalid"}), timeout=3)
        assert error.value.code == 403
        with urlopen(Request(url, data=body, headers={"Authorization": "Bearer " + token}), timeout=3) as response:
            assert json.load(response)["result"]["protocolVersion"] == "2025-03-26"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_managed_controller_requires_existing_tunnel_and_approval(tmp_path):
    from scripts.argon.tools_mcp import run_managed
    runner = Orchestrator(root=tmp_path, dry_run=True, backend=Backend(), prompt=lambda _: "no")
    with pytest.raises(ValueError, match="owner-provided HTTPS"):
        run_managed(runner, "http://localhost/mcp")
    with pytest.raises(PermissionError, match="declined"):
        run_managed(runner, "https://owner.example/mcp")


def test_receipt_requires_exact_request_hash(tmp_path):
    from types import SimpleNamespace
    from r2p.orchestrator_tools import LocalTools
    tool = LocalTools.__new__(LocalTools)
    tool.r = SimpleNamespace(options={"wait_seconds": 0}, publish=lambda **kwargs: None)
    path = tmp_path / "receipt.json"
    with pytest.raises(RuntimeError, match="no GPU/publish"):
        tool.wait_receipt(path, "expected")
    gemini.atomic_json(path, {"request_sha256": "wrong", "export": "x"})
    with pytest.raises(ValueError, match="exact request"):
        tool.wait_receipt(path, "expected")
    gemini.atomic_json(path, {"request_sha256": "expected", "export": "x"})
    assert tool.wait_receipt(path, "expected")["export"] == "x"


def test_owner_wait_stops_controller_without_resubmission(tmp_path):
    from r2p.orchestrator import WaitingForOwner
    backend = Backend()
    calls = []
    def execute(name, args):
        calls.append(name)
        raise WaitingForOwner("Persisted request is still pending")
    backend.execute = execute
    runner = Orchestrator(root=tmp_path, dry_run=True, backend=backend)
    with pytest.raises(WaitingForOwner, match="still pending"):
        runner.run()
    assert runner.state["status"] == "BLOCKED"
    assert calls == ["measure_scene"]


def test_live_collection_requires_publication_receipt(tmp_path):
    backend = Backend()
    runner = Orchestrator(root=tmp_path, backend=backend, prompt=lambda _: "yes")
    runner.state["results"]["build_game"] = {"ok": True, "metrics": {"passed": 1, "published": 0}}
    with pytest.raises(ValueError, match="publication receipt"):
        runner.execute(decision("collect_play"))
    assert not backend.calls
