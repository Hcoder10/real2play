"""Offline contracts for the read-only judges' view; no model or robot access."""
import json
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

import pytest

from dashboard.adapters import DashboardData
from dashboard.demo import DemoData, GAME_URL
from dashboard.server import make_server, qr_png


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def adapter(root):
    return DashboardData(root, {"auto_discover": True, "traces": ["roblox/out/traces"]},
                         fetch_json=lambda _: {"ok": True, "traces": 4})


def lab_run(root, name="live", **status):
    run = root / "runs/lab" / name
    write(run / "status.json", {"schema": "r2p_lab/1", "real_mock": False, "turn": 2,
                                "status": "running", "model": "gemini-4-argon", **status})
    return run


def test_lab_reads_latest_thought_tools_edits_and_all_residual_shapes(tmp_path):
    older = lab_run(tmp_path, "old")
    os.utime(older / "status.json", (1, 1))
    run = lab_run(tmp_path, "new")
    rows = [
        {"turn": 1, "response": {"thought": "# Thought: Observe first.", "calls": []},
         "results": [{"tool": "read_sim", "result": {"path": "physics.json", "content": "one\ntwo"}}]},
        {"turn": 2, "response": {"thought": "# Thought: Fix the servo.", "calls": [
            {"tool": "edit_sim", "args": {"path": "physics.json", "content": "one\nthree"}},
            {"tool": "replay", "args": {"episode": "1"}}, {"tool": "score", "args": {}}]},
         "results": [{"tool": "edit_sim", "result": {"edited": ["physics.json"], "validated": True}},
                     {"tool": "replay", "result": {"residuals": {"joint_rms_deg": 2.5, "jaw_path_rms_mm": 6,
                                                               "object_final_pose_error_mm": None, "image_similarity": .8}}},
                     {"tool": "score", "result": {"current": {"joint_rms_deg": 1.2, "jaw_path_rms_mm": 3,
                                                               "object_final_pose_error_mm": 4, "image_similarity": .9}}}]}]
    (run / "ledger.jsonl").write_text("\n".join(map(json.dumps, rows)) + '\n{"unfinished":', encoding="utf-8")
    frame = run / "turns/2/scene.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"image")
    result = DemoData(adapter(tmp_path)).lab()
    assert result["run"] == "new" and result["label"] == "Real bench"
    assert result["thought"] == "# Thought: Fix the servo."
    assert result["history"][-1]["jaw_mm"] == 3
    assert result["history"][0]["object_mm"] is None
    assert result["edits"] == [{"path": "physics.json", "edits": 1, "added": 1, "removed": 1, "diff_known": True}]
    assert result["frames"]["scene"].startswith("/media/")
    assert result["calls"][-1]["tool"] == "score"


def test_missing_outputs_and_simulated_reference_are_not_real_success(tmp_path):
    demo = DemoData(adapter(tmp_path))
    empty = demo.snapshot()
    assert not empty["lab"]["available"] and not empty["deploy"]["available"]
    assert next(s for s in empty["chain"]["stages"] if s["name"] == "Deploy")["state"] == "pending"
    lab_run(tmp_path, real_mock=True, status="incomplete", complete=False)
    write(tmp_path / "out/ttt/status.json", {"board": [""] * 9, "synthetic": True,
                                            "physical_placement": False, "status": "mock_complete"})
    result = demo.snapshot()
    assert result["lab"]["synthetic"] and result["lab"]["state"] == "blocked"
    assert result["deploy"]["synthetic"]
    assert next(s for s in result["chain"]["stages"] if s["name"] == "Deploy")["state"] == "pending"


def test_refused_real_tool_is_not_marked_success(tmp_path):
    run = lab_run(tmp_path)
    (run / "ledger.jsonl").write_text(json.dumps({"turn": 1, "results": [
        {"tool": "real_plan", "result": {"status": "refused", "error": "IK error"}}]}))
    call = DemoData(adapter(tmp_path)).lab()["calls"][0]
    assert call["ok"] is False and call["detail"] == "IK error"


def test_game_sources_are_disjoint_and_exclude_test_traces(tmp_path):
    directory = tmp_path / "roblox/out/traces"
    directory.mkdir(parents=True)
    entries = [{"source": "human"}, {"source": "bot", "headless": True}, {"source": "bot"},
               {"source": "expert"}, {"source": "human", "synthetic": True}, {"source": "human", "test_only": True}]
    for index, fields in enumerate(entries):
        (directory / f"{index}.jsonl").write_text(json.dumps({"meta": {"schema": "r2p_trace/1", **fields}}) + "\n")
    (directory / "partial.jsonl").write_text('{"meta":')
    result = DemoData(adapter(tmp_path)).game()
    assert result["groups"] == {"human": 1, "headless": 1, "bot": 1, "expert": 1, "unknown": 1}
    assert result["invalid"] == 1 and result["receiver_online"] and result["receiver_traces"] == 4


def test_synthetic_orchestrator_never_overrides_recorded_chain(tmp_path):
    run = tmp_path / "runs/argon_orchestrator/latest"
    write(run / "status.json", {"synthetic": True, "status": "DONE",
                                "results": {"deploy_real": {"ok": True, "metrics": {"attempts": 8}}}})
    result = DemoData(adapter(tmp_path)).snapshot()["chain"]
    assert not result["synthetic"]
    assert all(s["state"] == "pending" for s in result["stages"])
    assert all("orchestrator" not in s["provenance"].lower() for s in result["stages"])


def test_nested_board_decision_and_latency(tmp_path):
    write(tmp_path / "out/ttt/status.json", {"decision": {"board": [[1, 0, 2], [0, 0, 0], [0, 0, 0]],
          "move": "r1c1", "reason": "Take the centre", "latency_ms": 1250, "model": "gemini-4-argon"}})
    result = DemoData(adapter(tmp_path)).deploy()
    assert result["board"][:3] == ["X", "", "O"] and result["latency_s"] == 1.25
    assert result["why"] == "Take the centre"


@pytest.fixture
def http(tmp_path):
    data = adapter(tmp_path)
    server = make_server(data, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield data, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def request(url, range_value=None):
    req = urllib.request.Request(url, headers={"Range": range_value} if range_value else {})
    return urllib.request.urlopen(req, timeout=4)


def test_demo_http_and_safe_streaming_ranges(http, tmp_path):
    data, base = http
    video = tmp_path / "clip.mp4"
    video.write_bytes(bytes(range(100)))
    url = base + data.asset(str(video), tmp_path)
    with request(url, "bytes=10-19") as response:
        assert response.status == 206 and response.headers["Content-Range"] == "bytes 10-19/100"
        assert response.headers["Accept-Ranges"] == "bytes" and response.read() == bytes(range(10, 20))
    with request(url, "bytes=-7") as response:
        assert response.read() == bytes(range(93, 100))
    with request(url, "bytes=90-") as response:
        assert response.read() == bytes(range(90, 100))
    with request(url) as response:
        assert response.status == 200 and len(response.read()) == 100
    for invalid in ("bytes=999-", "bytes=-0", "bytes=40-20", "bytes=0-1,5-9", "nonsense"):
        with pytest.raises(urllib.error.HTTPError) as exc:
            request(url, invalid)
        assert exc.value.code == 416 and exc.value.headers["Content-Range"] == "bytes */100"
    with request(base + "/demo") as response:
        assert b"Argon Lab" in response.read()
    with request(base + "/api/demo") as response:
        assert len(json.load(response)["chain"]["stages"]) == 7
    for path in ("/media/not-registered", "/demo/camera/unknown.jpg", "/static/%2e%2e/config.json"):
        with pytest.raises(urllib.error.HTTPError) as exc:
            request(base + path)
        assert exc.value.code == 404


def test_actual_published_game_qr_decodes_locally():
    import cv2
    import numpy as np
    frame = cv2.imdecode(np.frombuffer(qr_png(GAME_URL), np.uint8), cv2.IMREAD_GRAYSCALE)
    assert cv2.QRCodeDetector().detectAndDecode(frame)[0] == GAME_URL


def test_browser_video_cache_preserves_original_and_encodes_h264(tmp_path):
    import shutil
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is an optional projector video dependency")
    import cv2
    import numpy as np
    from dashboard.video import BrowserVideos
    original = tmp_path / "runs/lab/offline/replays/one/side_by_side.mp4"
    original.parent.mkdir(parents=True)
    writer = cv2.VideoWriter(str(original), cv2.VideoWriter_fourcc(*"mp4v"), 10, (80, 60))
    for i in range(12):
        writer.write(np.full((60, 80, 3), i * 20, dtype=np.uint8))
    writer.release()
    os.utime(original, (time.time() - 3, time.time() - 3))
    evidence = original.read_bytes()
    cache = BrowserVideos(tmp_path)
    deadline = time.monotonic() + 15
    output, message = cache.get(original)
    while not output and time.monotonic() < deadline:
        time.sleep(.05)
        output, message = cache.get(original)
    assert output, message
    assert original.read_bytes() == evidence
    assert output.is_relative_to(tmp_path / "dashboard/.cache") and b"avc1" in output.read_bytes()
    assert cache.get(original) == (output, None)
