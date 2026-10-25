import json

import numpy as np
import pytest

from r2p import ttt
from r2p.gemini import FakeGemini, QuotaError


def camera():
    from r2p.perception import CameraModel
    return CameraModel.look_at()


def test_geometry_comes_from_task():
    from r2p.tasks import load_task
    task = load_task("place_on_cell")
    assert ttt.geometry()["cells"] == task["board"]["cells"]
    assert np.allclose(ttt.geometry()["cells"]["r2c2"], [0.205, 0.045])


@pytest.mark.parametrize("robot", ["X", "O"])
def test_exhaustive_perfect_minimax_never_loses(robot):
    visited = set()
    human = "O" if robot == "X" else "X"

    def walk(board, turn):
        if tuple(board) in visited:
            return
        visited.add(tuple(board))
        if ttt.terminal(board):
            assert ttt.winner(board) != human
            return
        moves = [ttt.minimax(board, robot)] if turn == robot else [i for i, mark in enumerate(board) if not mark]
        for i in moves:
            child = board.copy()
            child[i] = turn
            walk(child, "O" if turn == "X" else "X")
    walk([""] * 9, "X")
    assert len(visited) > 50


def test_reachable_validation_and_history():
    with pytest.raises(ValueError, match="reachable"):
        ttt.validate_board(["O"] + [""] * 8)
    with pytest.raises(ValueError, match="earlier mark"):
        ttt.consistent_board(["", "X", "O", "X", "", "", "", "", ""], ["X", "", "O", "", "", "", "", "", ""], "O")
    with pytest.raises(ValueError, match="more than one"):
        ttt.consistent_board(["X", "X", "X", "", "O", "", "", "", ""], ["X", "", "", "", "O", "", "", "", ""], "O")


def test_decide_one_main_request_with_two_distinct_views():
    board = ["X"] + [""] * 8
    fake = FakeGemini([{"board": board, "move": "r1c1", "why": "Take the centre"}])
    frame = np.full((480, 640, 3), 120, np.uint8)
    result = ttt.decide(frame, camera(), [], ask=fake)
    assert result["source"] == "gemini" and result["move"] == "r1c1"
    assert len(fake.requests) == 1 and fake.requests[0]["model"] == "gemini-4-argon"
    parts = fake.requests[0]["parts"]
    assert len(parts) == 3 and parts[1] != parts[2]


def test_exact_error_reask_then_fallback_on_consistent_board():
    board = ["X"] + [""] * 8
    reply = {"board": board, "move": "r0c0", "why": "oops"}
    fake = FakeGemini([reply, reply])
    result = ttt.decide(np.zeros((480, 640, 3), np.uint8), camera(), [], ask=fake)
    assert result["source"] == "minimax" and result["move"] == "r1c1"
    assert len(fake.requests) == 2
    assert "move r0c0 is occupied by X" in fake.requests[1]["parts"][-1]


def test_quota_requires_human_confirmation_on_stale_board():
    def unavailable(*args, **kwargs):
        raise QuotaError("offline")
    frame = np.zeros((480, 640, 3), np.uint8)
    with pytest.raises(ttt.BoardUnavailable):
        ttt.decide(frame, camera(), [], ask=unavailable)
    confirmed = {"board": ["X"] + [""] * 8}
    assert ttt.decide(frame, camera(), confirmed, ask=unavailable)["source"] == "minimax"


def test_model_cannot_make_a_losing_move():
    history = [{"cell": "r0c0", "mark": "X"}, {"cell": "r1c1", "mark": "O"}]
    board = ["X", "X", "", "", "O", "", "", "", ""]
    reply = {"board": board, "move": "r2c2", "why": "Ignore the immediate threat"}
    result = ttt.decide(np.zeros((480, 640, 3), np.uint8), camera(), history, ask=FakeGemini([reply, reply]))
    assert result["source"] == "minimax" and result["move"] == "r0c2"


def test_terminal_board_is_not_played():
    history = [{"cell": "r0c0", "mark": "X"}, {"cell": "r1c0", "mark": "O"},
               {"cell": "r0c1", "mark": "X"}, {"cell": "r1c1", "mark": "O"}]
    board = ["X", "X", "X", "O", "O", "", "", "", ""]
    result = ttt.decide(np.zeros((480, 640, 3), np.uint8), camera(), history,
                        ask=FakeGemini([{"board": board, "move": None, "why": "Human won"}]))
    assert result["move"] is None


def test_placement_uses_measured_supply():
    from r2p.tasks import validate, sample_scene
    task = ttt.placement_task("r2c1", [0.18, -0.12, 0.016])
    assert validate(task) == []
    scene = sample_scene(task, np.random.default_rng(3))
    objects = {o["id"]: o for o in scene["objects"]}
    piece = next(o for o in objects.values() if o["role"] == "manipulated")
    assert np.allclose(piece["xyz"], [0.18, -0.12, 0.016])
    assert np.allclose(objects["cell"]["xyz"][:2], [0.205, 0])
