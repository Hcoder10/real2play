"""Nonblocking operator input, on Windows and POSIX, plus a sequenced file."""
import json
import os
from pathlib import Path
import select
import sys
import time


class Operator:
    KEY_COMMANDS = {"s": "stop", " ": "stop", "p": "pause", "g": "go", "y": "yes", "n": "no", "q": "quit", "T": "torque_off"}
    ALIASES = {"resume": "go", "start": "go", "y": "yes", "n": "no", "s": "stop", "p": "pause", "g": "go", "q": "quit", "T": "torque_off", "SPACE": "stop", " ": "stop"}

    def __init__(self, control_file="runs/control.json", *, keyboard=True, clock=time.monotonic):
        self.path = Path(control_file) if control_file else None
        self.clock = clock
        self.keyboard = keyboard and sys.stdin.isatty()
        self.last_seq = -1
        self.armed_at = None
        self.saved_terminal = None
        self.score = None
        self.paused = False
        # Stale commands from previous processes must not fire on a new session.
        if self.path and self.path.exists():
            try:
                self.last_seq = int(json.loads(self.path.read_text())["seq"])
            except (ValueError, KeyError, OSError, TypeError):
                pass

    def __enter__(self):
        if self.keyboard and os.name != "nt":
            import termios
            import tty
            self.saved_terminal = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
        return self

    def close(self):
        if self.saved_terminal is not None:
            import termios
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.saved_terminal)
            self.saved_terminal = None

    def __exit__(self, *_):
        self.close()

    def _keys(self):
        if not self.keyboard:
            return []
        if os.name == "nt":
            import msvcrt
            keys = []
            while msvcrt.kbhit():
                keys.append(msvcrt.getwch())
            return keys
        keys = []
        while select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            if not key:
                break
            keys.append(key)
        return keys

    def poll(self):
        commands = [self.KEY_COMMANDS[k] for k in self._keys() if k in self.KEY_COMMANDS]
        if self.path:
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                seq = data["seq"]
                if isinstance(seq, int) and not isinstance(seq, bool) and seq > self.last_seq:
                    self.last_seq = seq
                    cmd = data["cmd"]
                    commands.append(self.ALIASES.get(cmd, cmd))
            except (OSError, ValueError, KeyError, TypeError):
                pass  # includes partially-written control files
        results = [result for cmd in commands if (result := self.feed(cmd)) is not None]
        if "torque_off_armed" in results:
            print("Torque off armed: support the arm at rest and repeat T within 3 seconds.", file=sys.stderr, flush=True)
        return results

    def feed(self, cmd):
        cmd = self.ALIASES.get(cmd, cmd)
        if cmd == "torque_off":
            now = self.clock()
            if self.armed_at is not None and 0 <= now - self.armed_at <= 3:
                self.armed_at = None
                return "torque_off"
            self.armed_at = now
            return "torque_off_armed"
        if cmd in ("yes", "no"):
            self.score = cmd == "yes"
        elif cmd == "pause":
            self.paused = True
        elif cmd == "go":
            self.paused = False
        elif cmd not in ("stop", "quit"):
            return None
        return cmd
