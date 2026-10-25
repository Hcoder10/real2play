"""Build deterministic board QA cases by projecting photo crops onto a real bench frame.

Coin inputs are face-on photos; cube input is a segmented BGRA bench crop with
its base midpoint given in sprite pixels. Outputs are composites, never real trials.
"""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from r2p import gemini, ttt
from r2p.perception import CameraModel


def coin_crop(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot read coin image {path}")
    h, w = image.shape[:2]
    # Source photos are tightly framed round coins, with black/transparent corners.
    y, x = np.indices((h, w))
    mask = (((x - (w - 1) / 2) / (w * .49))**2 + ((y - (h - 1) / 2) / (h * .49))**2 <= 1).astype(np.uint8) * 255
    if image.shape[2] == 4:
        mask = np.minimum(mask, image[..., 3])
    return cv2.resize(np.dstack((image[..., :3], mask)), (256, 256), interpolation=cv2.INTER_AREA)


def blend(frame, sprite, transform):
    h, w = frame.shape[:2]
    # Premultiply alpha to avoid black halos in bilinear interpolation.
    alpha = sprite[..., 3].astype(np.float32) / 255
    pixels = sprite[..., :3].astype(np.float32) * alpha[..., None]
    layer = cv2.warpPerspective(pixels, transform, (w, h))
    coverage = cv2.warpPerspective(alpha, transform, (w, h))
    return np.clip(frame * (1 - coverage[..., None]) + layer, 0, 255).astype(np.uint8)


def compose(frame, camera, board, coins, cube, *, cube_base, cube_width_m=.038, piece_m=.025, seed=0):
    rng = np.random.default_rng(seed)
    cells = ttt.geometry()["cells"]
    result = frame.copy()
    # Draw far cells first so nearer raised blocks can occlude farther pieces.
    for i in range(9):
        mark = board[i]
        if not mark:
            continue
        xy = np.asarray(cells[ttt.CELL_IDS[i]]) + rng.uniform(-.002, .002, 2)
        if mark == "X":
            which = (i + seed) % len(coins)
            sprite = coins[which]
            diameter = .01905 if which == 0 else .02121
            xyz = np.c_[xy + np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]]) * diameter / 2,
                        np.full(4, .0015)]
            target = camera.project(xyz).astype(np.float32)
            source = np.array([[0, 0], [255, 0], [255, 255], [0, 255]], np.float32)
            result = blend(result, sprite, cv2.getPerspectiveTransform(source, target))
        else:
            h, w = cube.shape[:2]
            projected = camera.project([[*xy, 0], [xy[0], xy[1] + piece_m, 0]])
            # Calibrated width and base point; sprite retains the photographed
            # perspective/texture. It is a 2D crop composite, not new geometry.
            width = float(np.linalg.norm(projected[1] - projected[0]))
            scale = width / w
            base = camera.project([*xy, 0])
            matrix = np.array([[scale, 0, base[0] - scale * cube_base[0]],
                               [0, scale, base[1] - scale * cube_base[1]], [0, 0, 1]], float)
            result = blend(result, cube, matrix)
    return result


def cases(count=20, seed=149):
    # Empty + every cell occupied by exactly one coin + independent mixed boards.
    result = [{"board": [""] * 9, "history": []}]
    for i in range(9):
        board = [""] * 9
        board[i] = "X"
        result.append({"board": board, "history": []})
    rng = np.random.default_rng(seed)
    seen = {tuple(r["board"]) for r in result}
    while len(result) < count:
        history, board = [], [""] * 9
        length = int(rng.choice([3, 5, 7, 9]))
        for move in range(length):
            i = int(rng.choice([j for j, value in enumerate(board) if not value]))
            mark = "X" if move % 2 == 0 else "O"
            board[i] = mark
            history.append({"cell": ttt.CELL_IDS[i], "mark": mark})
            if ttt.terminal(board):
                break
        if history[-1]["mark"] != "X" or tuple(board) in seen:
            continue
        seen.add(tuple(board))
        result.append({"board": board, "history": history[:-1]})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", required=True, type=Path)
    parser.add_argument("--camera", default="config/camera.json", type=Path)
    parser.add_argument("--penny", required=True, type=Path)
    parser.add_argument("--nickel", required=True, type=Path)
    parser.add_argument("--cube", required=True, type=Path)
    parser.add_argument("--cube-base", type=float, nargs=2, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    camera = CameraModel.load(args.camera)
    frame = cv2.imread(str(args.frame))
    if frame is None:
        parser.error("Cannot read frame")
    camera = camera.resized(frame.shape[1::-1])
    cube = cv2.imread(str(args.cube), cv2.IMREAD_UNCHANGED)
    if cube is None or cube.shape[2] != 4:
        parser.error("Cube must be a segmented BGRA crop")
    coins = [coin_crop(path) for path in (args.penny, args.nickel)]
    args.out.mkdir(parents=True, exist_ok=True)
    trials, thumbs = cases(args.count), []
    for i, case in enumerate(trials):
        image = compose(frame, camera, case["board"], coins, cube, cube_base=args.cube_base, seed=i)
        path = args.out / f"board_{i:02d}.jpg"
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        case.update(image=str(path), kind="photo_crop_composite")
        overlay, top = ttt.board_images(image, camera)
        cv2.imwrite(str(args.out / f"top_{i:02d}.jpg"), top)
        thumb = cv2.resize(top, (225, 225))
        cv2.putText(thumb, str(i), (5, 220), 0, .6, (0, 0, 255), 2)
        thumbs.append(thumb)
    while len(thumbs) % 5:
        thumbs.append(np.zeros_like(thumbs[0]))
    cv2.imwrite(str(args.out / "contact.jpg"), np.vstack([np.hstack(thumbs[i:i+5]) for i in range(0, len(thumbs), 5)]))
    gemini.atomic_json(args.out / "manifest.json", {"schema": "r2p_board_qa/1", "cases": trials,
        "sources": {"frame": str(args.frame), "penny": str(args.penny), "nickel": str(args.nickel), "cube": str(args.cube)},
        "limitations": "Synthetic occupancy on real background; 2D crop perspective, no new physical interaction."})
    print(args.out / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
