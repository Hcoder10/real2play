import json
from pathlib import Path
import threading
import urllib.request

import numpy as np

from r2p.pipeline.engine import RunOptions
from r2p.pipeline.storage import atomic_json


def test_cpu_smoke_defaults_and_explicit_overrides():
    smoke = RunOptions(task="place_on_cell", mode="dry-run")
    assert (smoke.traces, smoke.steps, smoke.eval_n, smoke.device) == (2, 5, 2, "cpu")
    assert RunOptions(task="x", mode="dry-run", traces=9).traces == 9
    assert RunOptions(task="x", mode="real").steps == 10000


def test_cpu_worker_needs_no_gpu_lease(monkeypatch):
    import sys
    from r2p.pipeline import train_worker
    monkeypatch.delenv("LANDLORD_LEASE_ID", raising=False)
    monkeypatch.delenv("R2P_LEASE_VRAM_GB", raising=False)
    monkeypatch.setattr(sys, "argv", ["worker", "--device", "cpu"])
    called = []
    monkeypatch.setattr(train_worker.runpy, "run_module", lambda *a, **k: called.append(a))
    train_worker.main()
    assert called == [("r2p.policy.train",)]


def test_publisher_preserves_mock_provenance_and_rejects_posts(tmp_path):
    from r2p.robot.publisher import Publisher, make_server
    Publisher(tmp_path).update({"source": "bot", "synthetic": True, "status": "running"}, np.zeros((20, 30, 3), np.uint8))
    server = make_server(tmp_path, "http://unused", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        status = json.load(urllib.request.urlopen(base + "/status.json"))
        assert status["synthetic"] and status["source"] == "bot"
        assert urllib.request.urlopen(base + "/frame.jpg").read().startswith(b"\xff\xd8")
        import pytest
        from urllib.error import HTTPError
        with pytest.raises(HTTPError) as error:
            urllib.request.urlopen(urllib.request.Request(base + "/cmd", data=b"{}"))
        assert error.value.code == 501
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_auto_discovery_prefers_real_twin_and_labels_smoke_policy(tmp_path):
    from dashboard.adapters import DashboardData
    atomic_json(tmp_path / "out/live/real/run.json", {"schema": "r2s_live/1", "session_kind": "bench", "iters": []})
    atomic_json(tmp_path / "runs/task-dry/r1/twin/attempt_001/run.json", {"schema": "r2s_live/1", "synthetic": True, "iters": []})
    atomic_json(tmp_path / "exports/smoke/config.json", {"name": "smoke", "synthetic": True, "eval": {"n": 1, "clean": 0., "real": 0.}})
    adapter = DashboardData(tmp_path, {"auto_discover": True})
    assert adapter.twin()["run"] == str(Path("out/live/real"))
    assert not adapter.twin()["synthetic"]
    assert adapter.policy()["evidence"] == "SMOKE EVALUATION"


def test_crashed_recording_does_not_hide_successful_retry(tmp_path):
    from r2p.pipeline.engine import Pipeline, Blocked
    import pytest
    adapter = Pipeline(RunOptions(task="cube_in_bowl", root=tmp_path)).adapters
    for name, error in (("attempt_1", "inference crashed"), ("attempt_2", None)):
        directory = tmp_path / "attempts" / name
        (directory / "frames").mkdir(parents=True)
        (directory / "frames/0000.jpg").write_bytes(b"fixture")
        (directory / "trace.jsonl").write_text(json.dumps({"meta": {"schema": "r2p_trace/1", "summary": {"error": error}}}) + "\n")
        if error:
            with pytest.raises(Blocked, match="inference crashed"):
                adapter.recorded_session(tmp_path / "attempts")
    assert adapter.recorded_session(tmp_path / "attempts").name == "attempt_2"
