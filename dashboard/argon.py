"""Read-only orchestrator strip, kept separate from other dashboard publishers."""
import json
from collections import deque


def snapshot(root):
    paths = sorted((root / "runs/argon_orchestrator").glob("*/status.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for path in paths:
        try:
            status = json.loads(path.read_text(encoding="utf-8"))
            if status.get("schema") != "r2p_argon_orchestrator/1":
                continue
            ledger = path.with_name("ledger.jsonl")
            rows = []
            if ledger.is_file():
                with ledger.open(encoding="utf-8") as stream:
                    for line in deque(stream, maxlen=12):
                        try:
                            rows.append(json.loads(line))
                        except ValueError:
                            pass  # A writer may currently be appending the final line.
            keys = ("run", "status", "stage", "last_tool", "why", "metrics", "next", "synthetic", "updated", "managed", "error")
            return {"available": True, **{k: status.get(k) for k in keys}, "ledger": rows}
        except (OSError, ValueError, TypeError):
            continue
    return {"available": False, "stage": "No orchestrator run", "managed": [], "ledger": []}
