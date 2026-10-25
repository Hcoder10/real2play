"""Build the timestamp-synchronized, CPU-only 15 s Argon highlight reel.

Run with .venv/Scripts/python.exe scripts/lead/make_demo_clip.py.
Only selected action windows are decoded; source footage is never modified.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FFMPEG = Path.home() / "AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1-full_build/bin/ffmpeg.exe"
# Exact, contiguous excerpts of the recorded thoughts, checked before rendering.
SHOTS = [
    (3, "Let's test descending over `cap2`"),
    (7, "Let's close the gripper to 6 and see how it grasps `cap2`!"),
    (13, "the jaws are high above `cap2`"),
    (14, "the two jaws will straddle `cap2` cleanly on both sides"),
    (17, "Let's close the gripper to 6 to grasp `cap2` firmly."),
    (18, "let's first lift and move part-way"),
    (19, "`cap2` is held cleanly above the grid"),
    (20, "will lower the cap right onto the center cell `r1c1`!"),
    (21, "Let's open the gripper to 20 to release `cap2` cleanly into `r1c1`."),
    (22, "`cap2` is released cleanly in cell `r1c1`!"),
]


class Camera:
    """Map wall time to written-frame ordinals, ignoring misleading MP4 FPS."""

    def __init__(self, folder: Path, name: str):
        self.folder, self.name = folder, name
        self.times, self.frames = [], []
        counts = {}
        for line in (folder / f"{name}_ts.txt").read_text().splitlines():
            part, stamp = line.split()
            part = int(part)
            ordinal = counts.get(part, 0)
            self.times.append(float(stamp))
            self.frames.append((part, ordinal))
            counts[part] = ordinal + 1
        self.cap, self.part, self.next_frame = None, None, 0

    def at(self, timestamp: float):
        if not self.times[0] <= timestamp <= self.times[-1]:
            raise ValueError(f"{self.name}: timestamp outside recording")
        i = min(bisect_left(self.times, timestamp), len(self.times) - 1)
        if i and timestamp - self.times[i - 1] < self.times[i] - timestamp:
            i -= 1
        part, ordinal = self.frames[i]
        if part != self.part:
            self.close()
            self.cap = cv2.VideoCapture(str(self.folder / f"{self.name}_{part:03}.mp4"))
            self.part, self.next_frame = part, 0
        if ordinal < self.next_frame or ordinal - self.next_frame > 100:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, ordinal)
            self.next_frame = ordinal
        while self.next_frame < ordinal:
            if not self.cap.grab():
                raise RuntimeError(f"Cannot skip {self.name} part {part}")
            self.next_frame += 1
        ok, frame = self.cap.read()
        self.next_frame += 1
        if not ok:
            raise RuntimeError(f"Missing {self.name} part {part}, frame {ordinal}")
        return frame

    def close(self):
        if self.cap is not None:
            self.cap.release()


def text(frame, message, xy, scale=1.0, color=(245, 245, 245), thickness=2):
    cv2.putText(frame, message, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def compose(scene, wrist, row, snippet, progress, end=False):
    frame = cv2.resize(scene, (1920, 1080))
    cv2.rectangle(frame, (0, 0), (1920, 108), (20, 22, 27), -1)
    text(frame, "Gemini 4 Argon - experimenting on the real arm", (48, 67), 1.35)
    cv2.rectangle(frame, (1338, 138), (1880, 490), (20, 22, 27), -1)
    frame[186:479, 1349:1869] = cv2.resize(wrist, (520, 293))
    text(frame, "WRIST CAMERA", (1355, 172), .72, (120, 230, 210))
    text(frame, "SCENE CAMERA", (48, 154), .7)
    cv2.rectangle(frame, (0, 865), (1920, 1080), (20, 22, 27), -1)
    text(frame, f"TURN {row['turn']:02d} / 23", (48, 913), .85, (120, 230, 210))
    text(frame, "ACTION HIGHLIGHTS  /  5.4x SPEED", (1230, 913), .8)
    if end:
        text(frame, "Real pick-and-place learned by experimenting", (48, 977), 1.35)
        text(frame, "23 turns  /  white bottle cap placed in the centre cell", (48, 1035), .9, (120, 230, 210))
    else:
        display = '"' + snippet.replace('`', '') + '"'
        text(frame, display, (48, 973), 1.08)
        action = row['reply']['action']
        if 'move' in action:
            label = "move  [" + ", ".join(f"{v:.4f}" for v in action['move']) + "] m"
        else:
            label = f"gripper  {action['gripper']}"
        text(frame, label, (48, 1030), .85, (175, 183, 195))
    cv2.rectangle(frame, (0, 1072), (int(1920 * progress), 1079), (120, 230, 210), -1)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'runs/bench/video/20261025_133414')
    parser.add_argument('--ledger', type=Path, default=ROOT / 'runs/argon_step/step_01/ledger.jsonl')
    parser.add_argument('--output', type=Path, default=ROOT / 'out/demo/argon_experiments_15s.mp4')
    parser.add_argument('--ffmpeg', default=shutil.which('ffmpeg') or str(FFMPEG))
    args = parser.parse_args()
    rows = {r['turn']: r for r in map(json.loads, args.ledger.read_text().splitlines())}
    for turn, snippet in SHOTS:
        if snippet not in rows[turn]['reply']['thought']:
            raise ValueError(f"Thought excerpt does not match turn {turn}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scene, wrist = Camera(args.source, 'scene'), Camera(args.source, 'wrist')
    command = [args.ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
               '-pix_fmt', 'bgr24', '-s', '1920x1080', '-r', '30', '-i', 'pipe:0',
               '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '19', '-threads', '2',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(args.output)]
    contact, number = [], 0
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for turn, snippet in SHOTS:
            row = rows[turn]
            print(f"Rendering turn {turn}", flush=True)
            # Start just before the logged command; retain seven seconds of motion.
            for timestamp in np.linspace(row['t'] - .25, row['t'] + 6.75, 39):
                frame = compose(scene.at(timestamp), wrist.at(timestamp), row, snippet, (number + 1) / 450)
                process.stdin.write(frame.tobytes())
                if number in (19, 214):
                    contact.append(cv2.resize(frame, (640, 360)))
                number += 1
        # Freeze the observed final result behind the two-second end card.
        final_scene, final_wrist = scene.at(rows[23]['t']), wrist.at(rows[23]['t'])
        for _ in range(60):
            frame = compose(final_scene, final_wrist, rows[23], '', (number + 1) / 450, end=True)
            process.stdin.write(frame.tobytes())
            number += 1
        contact.append(cv2.resize(frame, (640, 360)))
    finally:
        process.stdin.close()
        scene.close()
        wrist.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg encoding failed')
    sheet = args.output.with_name(args.output.stem + '_contact_sheet.jpg')
    if not cv2.imwrite(str(sheet), np.hstack(contact)):
        raise RuntimeError('Contact sheet write failed')
    probe = str(Path(args.ffmpeg).with_name('ffprobe.exe' if Path(args.ffmpeg).suffix == '.exe' else 'ffprobe'))
    info = json.loads(subprocess.check_output([probe, '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(args.output)]))
    stream = info['streams'][0]
    assert len(info['streams']) == 1 and stream['codec_name'] == 'h264'
    assert (stream['width'], stream['height'], stream['r_frame_rate']) == (1920, 1080, '30/1')
    assert int(stream['nb_frames']) == 450 and float(info['format']['duration']) <= 15
    print(json.dumps({'video': str(args.output), 'contact_sheet': str(sheet), 'frames': number, 'duration': info['format']['duration']}, indent=2))


if __name__ == '__main__':
    main()
