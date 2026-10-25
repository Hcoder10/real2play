import importlib.util
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

PATH = Path(__file__).resolve().parents[2] / "roblox/tools/trace_server.py"
spec = importlib.util.spec_from_file_location("trace_server", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def server(tmp_path):
    server = module.make_server("127.0.0.1", 0, tmp_path, "test-only-key")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, tmp_path
    server.shutdown(); server.server_close(); thread.join()


def post(server, body, key="test-only-key"):
    req = Request(f"http://127.0.0.1:{server.server_port}/trace", json.dumps(body, ensure_ascii=False).encode(),
                  {"Content-Type": "application/json", "X-Trace-Key": key})
    try:
        with urlopen(req, timeout=5) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


def chunk(part, start, n, final=False):
    return dict(trace_id="human_utf8_test", part=part, final=final,
                meta=dict(schema="r2p_trace/1", source="human", task="cube_in_bowl", task_name="cube_in_bowl", player="机器人 🦾"),
                ticks=[dict(t=i/30, q=[0]*6, q_target=[0.1]*6, poses={"cube": {"xyz": [.2, 0, .01], "yaw_deg": 0}},
                            grip_closed=False, command="release", phase="release") for i in range(start, start+n)],
                meta_final=dict(success=True, fail_reason="", commands=[[0, (start+n)/30, "release"]]))


def test_chunk_roundtrip_out_of_order_retry_utf8(server):
    srv, out = server
    status, data = post(srv, chunk(1, 600, 3, True))
    assert status == 200 and not data["complete"]
    assert not list(out.glob("*.jsonl"))
    assert post(srv, chunk(0, 0, 600))[1]["complete"]
    assert post(srv, chunk(0, 0, 600))[0] == 200
    rows = [json.loads(line) for line in (out / "human_utf8_test.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 604 and rows[0]["meta"]["player"] == "机器人 🦾"
    assert rows[-1]["q_target"] != rows[-1]["q"]
    assert rows[0]["meta"]["success"]


def test_auth_limits_traversal_and_conflicts(server):
    srv, out = server
    assert post(srv, chunk(0, 0, 1, True), "wrong")[0] == 401
    assert post(srv, chunk(0, 0, 601, True))[0] == 400
    bad = chunk(0, 0, 1, True); bad["trace_id"] = "../escape"
    assert post(srv, bad)[0] == 400
    assert post(srv, chunk(0, 0, 1, True))[0] == 200
    bad = chunk(0, 0, 1, True); bad["ticks"][0]["q_target"][0] = 2
    assert post(srv, bad)[0] == 400
    assert len(list(out.glob("*.jsonl"))) == 1


def test_reject_bad_time_and_nan(server):
    srv, out = server
    bad = chunk(0, 0, 2, True); bad["ticks"][1]["t"] = .1
    assert post(srv, bad)[0] == 400
    bad = chunk(0, 0, 1, True); bad["ticks"][0]["q"][0] = float("nan")
    assert post(srv, bad)[0] == 400
    assert not list(out.glob("*.jsonl"))
    with urlopen(f"http://127.0.0.1:{srv.server_port}/health") as response:
        assert json.load(response)["ok"]


def test_stats_require_key_and_exclude_test_fixtures(server):
    srv, out = server
    assert post(srv, chunk(0, 0, 1, True))[0] == 200
    fixture = chunk(0, 0, 1, True)
    fixture["trace_id"] = "fixture"
    fixture["meta"]["test_only"] = True
    assert post(srv, fixture)[0] == 200
    with pytest.raises(HTTPError) as error:
        urlopen(f"http://127.0.0.1:{srv.server_port}/stats")
    assert error.value.code == 401
    req = Request(f"http://127.0.0.1:{srv.server_port}/stats", headers={"X-Trace-Key":"test-only-key"})
    with urlopen(req) as response:
        stats = json.load(response)
    assert stats["rounds"] == 1
    assert stats["tasks"]["cube_in_bowl"]["success_rate"] == 1


def test_final_part_cannot_relabel_provenance(server):
    srv,out=server
    for field in ("source","level","physics_sha256"):
        bad=chunk(0,0,1,True); bad["trace_id"]="relabel_"+field
        bad["meta"].update(level="cube_in_bowl",physics_sha256="original")
        bad["meta_final"][field]="changed"
        assert post(srv,bad)[0]==400
    assert not list(out.glob("*.jsonl"))
