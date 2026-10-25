"""Make browser-playable copies of lab MP4s without changing experiment evidence."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time


class BrowserVideos:
    def __init__(self, root):
        self.root = Path(root)
        self.directory = self.root / "dashboard/.cache/videos"
        self.lock = threading.Lock()
        self.pending = set()
        self.errors = {}

    def get(self, source):
        if source is None:
            return None, None
        source = Path(source).resolve()
        if not source.is_relative_to((self.root / "runs/lab").resolve()):
            return None, "Video must belong to a lab run"
        stat = source.stat()
        key = hashlib.sha256(f"{source}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:24]
        output = self.directory / f"{key}.mp4"
        if output.is_file():
            return output, None
        # Do not read a file while the recorder is still closing its MP4 index.
        if time.time() - stat.st_mtime < 1:
            return None, "Waiting for the replay to finish writing"
        with self.lock:
            if key in self.errors:
                return None, self.errors[key]
            if not self.pending:
                self.pending.add(key)
                threading.Thread(target=self._encode, args=(source, output, key), daemon=True).start()
        return None, "Preparing browser replay"

    def _encode(self, source, output, key):
        temporary = output.with_suffix(".part.mp4")
        try:
            executable = shutil.which("ffmpeg")
            if not executable:
                raise RuntimeError("Install ffmpeg to play this lab's MPEG-4 videos in a browser")
            self.directory.mkdir(parents=True, exist_ok=True)
            command = [executable, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-threads", "1",
                       "-i", str(source), "-map", "0:v:0", "-an", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                       "-c:v", "libx264", "-threads", "1", "-preset", "veryfast", "-crf", "22",
                       "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)]
            result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.PIPE, timeout=120,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            if result.returncode:
                raise RuntimeError("Replay conversion failed; the original MP4 may still be incomplete")
            temporary.replace(output)
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            with self.lock:
                self.errors[key] = str(exc)
            temporary.unlink(missing_ok=True)
        finally:
            with self.lock:
                self.pending.discard(key)
