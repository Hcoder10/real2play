import json

import numpy as np
import pytest

from r2p.gemini import (CacheMiss, FakeGemini, Gemini, GeminiError, JSONError, QuotaError,
                        frames_to_video, image_part, model_for, video_part)


class Limited(RuntimeError):
    code = 429


def test_cache_content_addressed_and_roles(tmp_path, monkeypatch):
    fake = FakeGemini([("answer", "actual-main-version"), "small answer", "different budget"], cache_dir=tmp_path)
    text, first = fake.call(["question"], tag="test")
    assert text == "answer" and first["served_by"] == "actual-main-version" and not first["cached"]
    cached, record = fake.call(["question"], tag="another_tag")
    assert cached == text and record["cached"] and len(fake.requests) == 1
    fake.call(["question"], role="small", tag="test")
    fake.call(["question"], thinking=128, tag="test")
    assert len(fake.requests) == 3
    assert fake.requests[0]["config"]["thinking_config"]["thinking_budget"] == 2048
    monkeypatch.setenv("R2P_CACHE_ONLY", "1")
    fake.call(["question"], tag="cache_only")
    with pytest.raises(CacheMiss):
        fake.call(["never-seen"], tag="missing")
    assert len(fake.requests) == 3
    assert len((tmp_path / "calls.jsonl").read_text().splitlines()) == 6


@pytest.mark.parametrize("message", ["PerDay requests quota", "RequestsPerDayPerProjectPerModel", "limit: 0", '"quotaValue": "0"'])
def test_daily_quota_is_not_retried_or_silently_fallback(message):
    fake = FakeGemini([Limited(message)])
    with pytest.raises(QuotaError):
        fake.call("hello", tag="quota")
    assert len(fake.requests) == 1


def test_rpm_error_with_billing_retries_and_backoff():
    sleeps = []
    fake = FakeGemini([Limited("PerMinute limit: 20. See billing help"), Limited("429 RPM quota billing"), "ok"], sleep=sleeps.append)
    assert fake.call("x", tag="rpm")[0] == "ok"
    assert sleeps == [1, 2]
    assert len(fake.requests) == 3


def test_rate_limit_retry_exhaustion_is_not_daily():
    fake = FakeGemini([Limited("RPM billing") for _ in range(4)])
    with pytest.raises(GeminiError) as error:
        fake.call("x", tag="rpm")
    assert not isinstance(error.value, QuotaError)
    assert len(fake.requests) == 4


def test_explicit_fallback_reports_actual_model(monkeypatch):
    monkeypatch.setenv("R2P_ALLOW_FALLBACK", "1")
    fake = FakeGemini([Limited("PerDay"), ("ok", "flash-real-version")])
    _, record = fake.call("x", tag="fallback")
    assert record["served_by"] == "flash-real-version"
    assert record["fallback_from"] == "gemini-4-argon"
    assert [request["model"] for request in fake.requests] == ["gemini-4-argon", "gemini-3.8-flash"]


def test_json_repair_reuses_original_cache(tmp_path, monkeypatch):
    schema = {"type": "object", "required": ["label"], "properties": {"label": {"type": "string"}}}
    fake = FakeGemini(['{"name":"cube"}', {"label": "cube"}], cache_dir=tmp_path)
    text, record = fake.call(["label image"], tag="repair", schema=schema)
    assert json.loads(text) == {"label": "cube"}
    assert record["served_by"] == "gemini-4-argon"
    assert record["repair"]["served_by"] == "gemini-3.8-flash"
    assert '"name":"cube"' in fake.requests[1]["parts"][0]
    monkeypatch.setenv("R2P_CACHE_ONLY", "1")
    again, cached = fake.call(["label image"], tag="repair", schema=schema)
    assert again == text and cached["cached"] and cached["repair"]["cached"]
    assert len(fake.requests) == 2


def test_failed_json_repair_is_bounded():
    fake = FakeGemini(["not JSON", "also not JSON"])
    with pytest.raises(JSONError):
        fake.call("x", tag="bad_json", schema={"type": "object"})
    assert len(fake.requests) == 2


def test_cache_only_does_not_construct_transport(tmp_path, monkeypatch):
    monkeypatch.setenv("R2P_CACHE_ONLY", "1")
    with pytest.raises(CacheMiss):
        Gemini(tmp_path).call("x", tag="offline")


def test_keys_never_enter_logs_or_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-for-this-test")
    fake = FakeGemini(["secret-for-this-test echoed"], cache_dir=tmp_path)
    text, _ = fake.call("hello", tag="secret-for-this-test")
    assert "secret-for-this-test" not in text
    for file in tmp_path.glob("*"):
        assert "secret-for-this-test" not in file.read_text()


def test_model_and_budget_overrides(monkeypatch):
    monkeypatch.setenv("R2P_MODEL_MAIN", "test-main")
    monkeypatch.setenv("R2P_THINKING", "1024")
    fake = FakeGemini(["ok"])
    fake.call("x", tag="config")
    assert fake.requests[0]["model"] == "test-main"
    assert fake.requests[0]["config"]["thinking_config"]["thinking_budget"] == 1024
    with pytest.raises(ValueError):
        model_for("wrong")
    with pytest.raises(ValueError):
        fake.call("x", tag="config", thinking=-1)


def test_image_and_whole_video_parts(tmp_path):
    frames = [np.full((32, 48, 3), i * 30, np.uint8) for i in range(5)]
    part = image_part(frames[0])
    assert part["inline_data"]["mime_type"] == "image/jpeg"
    path = frames_to_video(frames, tmp_path / "episode.mp4", 5)
    part = video_part(path)
    assert part["inline_data"]["mime_type"] == "video/mp4"
    assert part["inline_data"]["data"] == path.read_bytes()
