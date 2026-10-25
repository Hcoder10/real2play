"""Crash-safe JSON and a cross-platform, process-lifetime run lock."""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def atomic_json(path, value):
    """Replace only after a complete, flushed serialization; preserve the old file on failure."""
    path = Path(path)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # On Windows, a dashboard reader may briefly deny FILE_SHARE_DELETE.
        # Retry the rename only: never fall back to truncating the destination.
        for attempt in range(50):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if os.name != "nt" or attempt == 49:
                    raise
                time.sleep(0.01)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def run_lock(path):
    """OS releases the advisory lock even when a process crashes. Never delete lock inodes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    if stream.tell() == 0:
        stream.write(b"0")
        stream.flush()
    stream.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        stream.close()
        raise RuntimeError(f"Another pipeline owns {path.parent}") from exc
    try:
        yield
    finally:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()
