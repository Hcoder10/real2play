import json

import cv2
import numpy as np
import pytest

from r2p import gemini, ttt
from r2p.perception import CameraModel

REAL_REQUEST = gemini.Gemini._request


@pytest.mark.parametrize("error", [TimeoutError("timed out"), ConnectionError("reset"),
                                     type("ServerError", (RuntimeError,), {"code": 503})("busy")])
def test_transient_attempts_are_counted_and_logged(error):
    sleeps = []
    client = gemini.FakeGemini([error, "ok"], sleep=sleeps.append)
    _, record = client.call("retry", tag="retry")
    assert record["attempts"] == 2 and sleeps == [1]
    assert len(record["attempt_details"]) == 2
    assert record["attempt_details"][0]["served_by"] is None
    assert record["served_by"] == "gemini-4-argon"


def test_request_budget_includes_retries_and_json_repairs():
    client = gemini.FakeGemini([TimeoutError("slow"), "not json", "{}"])
    client.max_requests = 2
    with pytest.raises(gemini.RequestLimit):
        client.call("repair", tag="test", schema={"type": "object"})
    assert len(client.requests) == client.requests_used == 2
    assert client.records[-1]["error"] == "RequestLimit"
    assert client.records[-1]["attempts"] == 0


def test_sdk_has_finite_timeout_and_no_hidden_retries(monkeypatch):
    from google import genai
    seen = {}
    class Client:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            self.models = self
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def generate_content(self, **kwargs):
            return "ok"
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(genai, "Client", Client)
    REAL_REQUEST(gemini.Gemini(timeout_s=7), "fake", ["hi"], {})
    assert seen["http_options"].timeout == 7000
    assert seen["http_options"].retry_options.attempts == 1


def test_empty_board_waits_and_does_not_use_minimax():
    camera = CameraModel.look_at()
    frame = np.zeros((480, 640, 3), np.uint8)
    answer = {"board": [""] * 9, "move": None, "why": "Human starts"}
    client = gemini.FakeGemini([answer])
    result = ttt.decide(frame, camera, [], ask=client)
    assert result["move"] is None and result["status"] == "waiting_for_human"
    assert len(client.requests) == 1


def test_rotated_measured_board_remap_and_resized_camera():
    camera = CameraModel.look_at(image_size=(960, 540))
    angle = .35
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    cells = {key: (np.array([.22, .01]) + np.array([i // 3 - 1, i % 3 - 1]) @ rot * .035).tolist()
             for i, key in enumerate(ttt.CELL_IDS)}
    task = {"board": {"center_xy": [.22, .01], "cell_m": .035, "cells": cells}}
    frame = np.zeros((270, 480, 3), np.uint8)
    small = camera.resized((480, 270))
    for i, xy in enumerate(cells.values()):
        p = np.rint(small.project([*xy, 0])).astype(int)
        cv2.circle(frame, tuple(p), 3, (20 * (i + 1), 255, 0), -1)
    _, top = ttt.board_images(frame, camera, task)
    for i in range(9):
        sample = top[i // 3 * 150 + 75, i % 3 * 150 + 75]
        assert sample[1] > 200
        assert abs(int(sample[0]) - 20 * (i + 1)) < 3
