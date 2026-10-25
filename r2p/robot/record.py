"""Paired frame/tick recording. The control loop never waits on JPEG or disk."""
from copy import deepcopy
import json
from pathlib import Path
import queue
import threading

import numpy as np


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


class TraceWriter:
    def __init__(self, directory, meta, *, capacity=32):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "frames").mkdir()
        self.meta = {**deepcopy(meta), "schema": "r2p_trace/1"}
        self.queue = queue.Queue(maxsize=capacity)
        self.dropped = 0
        self.written = 0
        self.error = None
        self.closed = False
        self.thread = threading.Thread(target=self._run, name="r2p-trace-writer", daemon=True)
        self.thread.start()

    def submit(self, tick, frame=None):
        if self.closed:
            raise RuntimeError("writer closed")
        if self.error:
            raise RuntimeError("recording failed") from self.error
        item = (deepcopy(tick), None if frame is None else frame.copy())
        try:
            self.queue.put_nowait(item)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _run(self):
        try:
            with (self.directory / "ticks.tmp").open("w", encoding="utf-8") as stream:
                while (item := self.queue.get()) is not None:
                    tick, frame = item
                    if frame is not None:
                        import cv2
                        name = f"frames/{self.written:04d}.jpg"
                        if not cv2.imwrite(str(self.directory / name), frame):
                            raise OSError("JPEG write failed")
                        tick["frame"] = name
                    stream.write(json.dumps(tick, default=json_default) + "\n")
                    self.written += 1
        except BaseException as exc:
            self.error = exc

    def close(self, summary=None):
        if self.closed:
            return
        while self.thread.is_alive():
            try:
                self.queue.put(None, timeout=0.1)
                break
            except queue.Full:
                continue
        self.thread.join()
        self.closed = True
        if self.error:
            raise RuntimeError("recording failed") from self.error
        if summary:
            self.meta.update(success=summary["success"], fail_reason=None if summary["success"] else summary["outcome"], summary=summary)
        self.meta.update(dropped=self.dropped, ticks=self.written)
        dest = self.directory / "trace.jsonl"
        with dest.open("w", encoding="utf-8") as out, (self.directory / "ticks.tmp").open(encoding="utf-8") as source:
            out.write(json.dumps({"meta": self.meta}, default=json_default) + "\n")
            for line in source:
                out.write(line)
        (self.directory / "ticks.tmp").unlink()
        (self.directory / "meta.json").write_text(json.dumps(self.meta, indent=2, default=json_default), encoding="utf-8")
