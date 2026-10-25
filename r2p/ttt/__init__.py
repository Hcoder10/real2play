"""Tic-tac-toe geometry, perfect play and a validated Argon board reader."""
from __future__ import annotations

import copy
import json
from functools import lru_cache

import numpy as np

from r2p import gemini

LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))
CELL_IDS = tuple(f"r{i}c{j}" for i in range(3) for j in range(3))
REPLY_SCHEMA = {"type": "object", "required": ["board", "move", "why"], "properties": {
    "board": {"type": "array", "minItems": 9, "maxItems": 9, "items": {"type": "string", "enum": ["", "X", "O"]}},
    "move": {"anyOf": [{"type": "string", "enum": list(CELL_IDS)}, {"type": "null"}]},
    "why": {"type": "string"}}}


class BoardUnavailable(ValueError):
    """No reliable current board; the operator must confirm it before moving."""


def geometry(task=None):
    """Read the shared task board, with the binding interface as bootstrap fallback."""
    if task is None:
        try:
            from r2p.tasks import load_task
            task = load_task("place_on_cell")
        except (ImportError, FileNotFoundError):
            task = {"board": {"center_xy": [0.245, 0], "cell_m": 0.045}}
    board = task["board"]
    pitch = float(board["cell_m"])
    centre = np.asarray(board["center_xy"], dtype=float)
    cells = board.get("cells") or {key: (centre + [(i // 3 - 1) * pitch, (i % 3 - 1) * pitch]).tolist()
                                   for i, key in enumerate(CELL_IDS)}
    return {"cell_m": pitch, "center_xy": centre.tolist(), "cells": {key: list(cells[key]) for key in CELL_IDS}}


def cell_index(cell):
    if isinstance(cell, str) and cell in CELL_IDS:
        return CELL_IDS.index(cell)
    if isinstance(cell, int) and not isinstance(cell, bool) and 0 <= cell < 9:
        return cell
    raise ValueError(f"Unknown cell {cell!r}; use r0c0 through r2c2")


def normalize_board(board):
    if not isinstance(board, (list, tuple)) or len(board) != 9:
        raise ValueError("board must contain exactly nine marks")
    values = []
    for mark in board:
        if mark in (None, "", ".", " ", 0):
            values.append("")
        elif mark in ("X", "O"):
            values.append(mark)
        else:
            raise ValueError(f"Unknown board mark {mark!r}")
    return values


def winner(board):
    for a, b, c in LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    return None


def terminal(board):
    return winner(board) is not None or all(board)


@lru_cache(maxsize=1)
def _reachable():
    states = set()

    def visit(board, mark):
        if board in states:
            return
        states.add(board)
        if terminal(board):
            return
        for i, value in enumerate(board):
            if not value:
                visit(board[:i] + (mark,) + board[i + 1:], "O" if mark == "X" else "X")
    visit(("",) * 9, "X")
    return states


def validate_board(board):
    board = normalize_board(board)
    if tuple(board) not in _reachable():
        raise ValueError("board is not reachable by alternating legal play with X first")
    return board


def history_board(history):
    if isinstance(history, dict) and "board" in history:
        return validate_board(history["board"])
    if len(history) == 9 and all(not isinstance(item, dict) for item in history):
        return validate_board(history)
    board = [""] * 9
    for entry in history:
        if terminal(board):
            raise ValueError("history contains a move after game end")
        i = cell_index(entry.get("cell", entry.get("move")))
        mark = entry["mark"]
        expected = "X" if board.count("X") == board.count("O") else "O"
        if board[i] or mark != expected:
            raise ValueError("history contains an occupied cell or out-of-turn mark")
        board[i] = mark
    return validate_board(board)


def consistent_board(board, previous, robot_mark):
    if robot_mark not in ("X", "O"):
        raise ValueError("robot_mark must be X or O")
    board = normalize_board(board)
    previous = validate_board(previous)
    human = "O" if robot_mark == "X" else "X"
    additions = []
    for i, (old, new) in enumerate(zip(previous, board)):
        if old and old != new:
            raise ValueError(f"earlier mark at {CELL_IDS[i]} changed from {old} to {new or 'empty'}")
        if not old and new:
            additions.append((i, new))
    if any(mark != human for _, mark in additions):
        raise ValueError("an unrecorded robot mark appeared")
    if len(additions) > 1:
        raise ValueError("more than one new human mark appeared")
    if terminal(previous) and additions:
        raise ValueError("a new mark appeared after game end")
    validate_board(board)
    if not terminal(board):
        turn = "X" if board.count("X") == board.count("O") else "O"
        if turn != robot_mark:
            raise ValueError("the human has not yet made a confirmed move")
    return board


def validate_decision(answer, previous, robot_mark="O"):
    if not isinstance(answer, dict):
        raise ValueError("answer must be a JSON object")
    board = consistent_board(answer.get("board"), previous, robot_mark)
    if terminal(board):
        if answer.get("move") is not None:
            raise ValueError("move must be null because the game is over")
        return board, None
    i = cell_index(answer.get("move"))
    if board[i]:
        raise ValueError(f"move {CELL_IDS[i]} is occupied by {board[i]}")
    return board, i


@lru_cache(maxsize=None)
def _value(board, turn, robot):
    won = winner(board)
    if won:
        return 1 if won == robot else -1
    if all(board):
        return 0
    outcomes = [_value(board[:i] + (turn,) + board[i + 1:], "O" if turn == "X" else "X", robot)
                for i, mark in enumerate(board) if not mark]
    return max(outcomes) if turn == robot else min(outcomes)


def minimax(board, robot_mark="O"):
    board = validate_board(board)
    if terminal(board):
        return None
    turn = "X" if board.count("X") == board.count("O") else "O"
    if robot_mark != turn:
        raise ValueError("minimax called out of turn")
    # Deterministic preference: centre, corners, then sides among equally good moves.
    order = (4, 0, 2, 6, 8, 1, 3, 5, 7)
    return max((i for i in order if not board[i]),
               key=lambda i: _value(tuple(board[:i] + [robot_mark] + board[i + 1:]),
                                    "O" if robot_mark == "X" else "X", robot_mark))


def board_images(frame, cam_model, task=None):
    """Full camera frame with named projected cells, plus a genuine perspective warp."""
    import cv2

    grid = geometry(task)
    marked = np.asarray(frame).copy()
    pitch = grid["cell_m"]
    for key, xy in grid["cells"].items():
        xy = np.asarray(xy)
        corners = xy + np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * pitch / 2
        points = np.asarray(cam_model.project(np.c_[corners, np.zeros(4)]), float)
        if not np.isfinite(points).all():
            raise ValueError("Grid projects outside the calibrated camera")
        cv2.polylines(marked, [points.round().astype(np.int32)], True, (0, 255, 255), 1, cv2.LINE_AA)
        centre = np.asarray(cam_model.project(np.array([[*xy, 0.0]]))).reshape(-1, 2)[0]
        cv2.putText(marked, key, tuple(centre.round().astype(int)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(marked, key, tuple(centre.round().astype(int)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 255), 1, cv2.LINE_AA)
    x, y = grid["center_xy"]
    half = 1.5 * pitch
    # Rows are x, columns y. The warped image uses that exact row/column convention.
    corners = np.array([[x-half, y-half, 0], [x-half, y+half, 0],
                        [x+half, y+half, 0], [x+half, y-half, 0]])
    src = np.asarray(cam_model.project(corners), np.float32).reshape(4, 2)
    dst = np.array([[0, 0], [449, 0], [449, 449], [0, 449]], np.float32)
    if abs(cv2.contourArea(src)) < 1:
        raise ValueError("Grid has a degenerate camera projection")
    top = cv2.warpPerspective(frame, cv2.getPerspectiveTransform(src, dst), (450, 450))
    for i, key in enumerate(CELL_IDS):
        cv2.putText(top, key, (i % 3 * 150 + 6, i // 3 * 150 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return marked, top


def decide(frame, cam_model, history, robot_mark="O", robot_desc="brown cubes", human_desc="blue discs", *, ask=None):
    previous = history_board(history)
    marked, top = board_images(frame, cam_model)
    prompt = ("Read the physical tic-tac-toe board and select a perfect move. X starts. "
              "Board is row-major r0c0..r2c2; empty is an empty string. Return board[9], move (cell id or null "
              "when terminal), why. Preserve all earlier marks. At most one new human mark is possible.\n"
              f"Robot is {robot_mark}: {robot_desc}. Human: {human_desc}.\n"
              f"Confirmed history: {json.dumps(history)}\nLast confirmed board: {json.dumps(previous)}\n"
              "Image 1: projected cells on the real frame. Image 2: top-down grid, r0 at top, c0 at left.")
    parts = [prompt, gemini.image_part(marked), gemini.image_part(top)]
    ask = ask or gemini.call
    consistent = None
    errors = []
    model = None
    for attempt in range(2):
        text = "No parseable reply"
        try:
            text, record = ask(parts, role="main", tag=f"ttt:decide:{attempt}", schema=REPLY_SCHEMA)
            model = record.get("served_by", "unknown")
            print(f"Tic-tac-toe answered by {model}")
            answer = gemini.parse_json(text)
            # Retain a trustworthy board even if the move itself is invalid.
            consistent = consistent_board(answer.get("board"), previous, robot_mark)
            board, move = validate_decision(answer, previous, robot_mark)
            best = minimax(board, robot_mark)
            if move is not None:
                opponent = "X" if robot_mark == "O" else "O"
                played = tuple(board[:move] + [robot_mark] + board[move + 1:])
                optimal = tuple(board[:best] + [robot_mark] + board[best + 1:])
                if _value(played, opponent, robot_mark) < _value(optimal, opponent, robot_mark):
                    raise ValueError(f"move {CELL_IDS[move]} gives a worse minimax outcome than {CELL_IDS[best]}")
            return {"board": board, "move": CELL_IDS[move] if move is not None else None,
                    "why": answer.get("why", ""), "model": model, "source": "gemini", "record": record}
        except gemini.JSONError as exc:
            errors.append(str(exc))
            parts = parts[:3] + ["Exact validation error: " + str(exc) + "\nReturn the requested board, move and why JSON."]
        except gemini.GeminiError as exc:
            errors.append(str(exc))
            break
        except (ValueError, TypeError, AttributeError) as exc:
            errors.append(str(exc))
            parts = parts[:3] + ["Your last reply was: " + text + "\nExact validation error: " + str(exc)
                                + "\nCorrect the reply using the same two images."]
    if consistent is None:
        # Only use history if it already includes a confirmed human move / initial robot turn.
        try:
            consistent = consistent_board(previous, previous, robot_mark)
        except ValueError:
            raise BoardUnavailable("Operator must confirm the current human move. " + "; ".join(errors)) from None
    move = minimax(consistent, robot_mark)
    return {"board": consistent, "move": CELL_IDS[move] if move is not None else None,
            "why": "Local minimax: " + "; ".join(errors), "model": model, "source": "minimax"}


def placement_task(cell, supply_xyz, *, piece_size=0.032, board=None, robot_mark="O"):
    """One placement, with a measured supply pose and the board's fixed target pose."""
    from r2p.tasks import load_task

    task = copy.deepcopy(load_task("place_on_cell"))
    xy = geometry(task)["cells"][CELL_IDS[cell_index(cell)]]
    manipulated = next(obj for obj in task["objects"] if obj["role"] == "manipulated")
    target = next(obj for obj in task["objects"] if obj["role"] == "target")
    manipulated.update(size_m=[piece_size] * 3, xyz=list(supply_xyz))
    manipulated.pop("xy", None)
    target["xyz"] = [*xy, 0.0005]
    task["randomize"] = {manipulated["id"]: {"x": [supply_xyz[0]] * 2, "y": [supply_xyz[1]] * 2},
                         target["id"]: {"x": [xy[0]] * 2, "y": [xy[1]] * 2}, "min_sep_m": 0}
    # The policy slots hold the four most recent blockers, as specified by the shared task.
    occupied = [(i, mark) for i, mark in enumerate(board or [""] * 9) if mark]
    blockers = [o for o in task["objects"] if o["role"] == "obstacle"]
    for obj, entry in zip(blockers, occupied[-len(blockers):]):
        pos = geometry(task)["cells"][CELL_IDS[entry[0]]]
        obj.update(xyz=[*pos, piece_size / 2], fixed=True)
        task["randomize"][obj["id"]] = {"xy_choice": [pos], "p_present": 1}
    for obj in blockers[len(occupied):]:
        task["randomize"][obj["id"]] = {"p_present": 0}
        obj.update(optional=True, p_present=0)
    task["target_cell"] = CELL_IDS[cell_index(cell)]
    return task
