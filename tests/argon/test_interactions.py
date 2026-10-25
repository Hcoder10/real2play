"""SDK-boundary tests: these exercise Interactions, never the live service."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from r2p import gemini

REAL_REQUEST = gemini.Gemini._request


def sdk_client(monkeypatch, tmp_path, replies, **options):
    requests = []
    def send(operation, **kwargs):
        requests.append((operation, kwargs))
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply
    service = SimpleNamespace(create=lambda **kw: send("create", **kw), get=lambda **kw: send("get", **kw))
    client = gemini.Gemini(tmp_path, sleep=lambda _: None, **options)
    monkeypatch.setattr(client, "_client", lambda: nullcontext(SimpleNamespace(interactions=service)))
    monkeypatch.setattr(client, "_request", REAL_REQUEST.__get__(client))
    return client, requests


def done(text="ok"):
    return SimpleNamespace(id="interaction-1", status="completed", output_text=text,
                           model="gemini-4-argon", usage={"total_tokens": 22})


def test_interactions_typed_media_schema_and_usage(monkeypatch, tmp_path):
    client, requests = sdk_client(monkeypatch, tmp_path, [done('{"ok":true}')])
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    parts = ["look", {"inline_data": {"mime_type": "image/jpeg", "data": b"jpeg"}},
             {"inline_data": {"mime_type": "video/mp4", "data": b"movie"}}]
    _, record = client.call(parts, tag="active_patch", schema=schema)
    body = requests[0][1]
    assert [p["type"] for p in body["input"]] == ["text", "image", "video"]
    assert body["input"][1]["data"] == "anBlZw=="
    assert body["response_format"] == {"type": "text", "mime_type": "application/json", "schema": schema}
    assert body["generation_config"] == {"thinking_level": "high", "max_output_tokens": 8192}
    assert body["timeout"] == 1200
    assert not {"temperature", "top_p", "top_k", "thinking_budget", "cached_content"} & body.keys()
    assert record["usage"] == {"total_tokens": 22} and record["interaction_id"] == "interaction-1"
    monkeypatch.setenv("R2P_CACHE_ONLY", "1")
    assert client.call(parts, tag="active_patch", schema=schema)[1]["usage"] == record["usage"]
    assert len(requests) == 1


def test_background_polls_same_id_and_budgets_every_wire_call(monkeypatch, tmp_path):
    pending = SimpleNamespace(id="interaction-1", status="in_progress")
    error = type("Busy", (RuntimeError,), {"status_code": 503})("busy")
    client, requests = sdk_client(monkeypatch, tmp_path, [pending, error, done()], max_requests=3)
    assert client.call("work", tag="background", background=True)[0] == "ok"
    assert [op for op, _ in requests] == ["create", "get", "get"]
    assert requests[1][1]["id"] == requests[2][1]["id"] == "interaction-1"
    assert client.requests_used == 3


@pytest.mark.parametrize("status", ["failed", "cancelled", "incomplete", "budget_exceeded", "requires_action"])
def test_terminal_failure_does_not_cache_text_or_relaunch(monkeypatch, tmp_path, status):
    client, requests = sdk_client(monkeypatch, tmp_path, [SimpleNamespace(id="i", status=status, output_text="partial")])
    with pytest.raises(gemini.GeminiError, match=status):
        client.call("x", tag="terminal", background=True)
    assert len(requests) == 1 and not list(tmp_path.glob("*.json"))


def test_poll_budget_exhaustion_cannot_duplicate_create(monkeypatch, tmp_path):
    client, requests = sdk_client(monkeypatch, tmp_path, [SimpleNamespace(id="i", status="in_progress")], max_requests=1)
    with pytest.raises(gemini.RequestLimit):
        client.call("x", tag="limited", background=True)
    assert len(requests) == 1


def test_poll_timeout_names_resumable_id(monkeypatch, tmp_path):
    now = [0.0]
    client, requests = sdk_client(monkeypatch, tmp_path, [SimpleNamespace(id="resume-me", status="in_progress")],
                                  timeout_s=1, clock=lambda: now[0])
    client.sleep = lambda seconds: now.__setitem__(0, now[0] + seconds)
    with pytest.raises(gemini.GeminiError, match="resume-me"):
        client.call("x", tag="deadline", background=True)
    assert len(requests) == 1


def test_poll_backoff_cannot_send_after_deadline(monkeypatch, tmp_path):
    now = [0.0]
    error = type("Busy", (RuntimeError,), {"code": 503})("busy")
    client, requests = sdk_client(monkeypatch, tmp_path, [SimpleNamespace(id="i", status="in_progress"), error],
                                  timeout_s=1, poll_interval_s=.1, clock=lambda: now[0])
    client.sleep = lambda seconds: now.__setitem__(0, now[0] + seconds)
    with pytest.raises(gemini.GeminiError, match="timed out"):
        client.call("x", tag="deadline", background=True)
    assert [op for op, _ in requests] == ["create", "get"]


def test_flash_requires_approval_and_logs_site(monkeypatch, caplog):
    client = gemini.FakeGemini(["argon", "flash"])
    client.call("x", role="small", tag="perception_boxes")
    assert client.requests[0]["model"] == "gemini-4-argon"
    monkeypatch.setenv("R2P_ALLOW_FLASH", "1")
    client.call("x", role="small", tag="perception_boxes")
    assert client.requests[1]["model"] == "gemini-3.8-flash"
    assert "perception_boxes" in caplog.text


def test_fallback_flag_alone_does_not_authorize_flash(monkeypatch):
    monkeypatch.setenv("R2P_ALLOW_FALLBACK", "1")
    error = type("Limited", (RuntimeError,), {"code": 429})("PerDay limit: 0")
    client = gemini.FakeGemini([error])
    with pytest.raises(gemini.QuotaError):
        client.call("x", tag="quota")
    assert len(client.requests) == 1


def test_small_role_cannot_bypass_flash_task_restriction(monkeypatch):
    monkeypatch.setenv("R2P_ALLOW_FLASH", "1")
    client = gemini.FakeGemini([])
    with pytest.raises(ValueError, match="low-effort"):
        client.call("choose an experiment", role="small", tag="active_choose")
    assert not client.requests


def test_flash_approval_does_not_move_main_reasoning_to_flash(monkeypatch):
    monkeypatch.setenv("R2P_ALLOW_FALLBACK", "1")
    monkeypatch.setenv("R2P_ALLOW_FLASH", "1")
    error = type("Limited", (RuntimeError,), {"code": 429})("PerDay limit: 0")
    client = gemini.FakeGemini([error])
    with pytest.raises(gemini.QuotaError):
        client.call("choose an experiment", tag="active_choose")
    assert len(client.requests) == 1


@pytest.mark.parametrize("tag,expected", [("visual:point", "low"), ("real2sim:detect", "low"),
    ("active:cap_pose", "low"),
    ("perception_boxes", "low"), ("task_from_video:0", "low"), ("labeling:refine", "low"),
    ("ttt:decide:0", "medium"), ("real2sim:critique:1", "medium"),
    ("active_choose", "high"), ("active_patch", "high"), ("build_game", "high"), ("author_task", "high")])
def test_call_site_levels(tag, expected):
    assert gemini.thinking_level(tag) == expected
