"""Twin -> game: turn an Argon Lab run's sealed held-out result into the game builder's verification evidence and build
the Roblox level from Argon's OWN sim (runs/lab/<run>/sim/physics.json) via the Antigravity managed agent.

  python scripts/lead/lab_to_game.py --run lab-real-07 [--dry-run]
Refuses (no verified claim) unless the Lab's held-out gate improved; --provisional builds a labelled rehearsal instead.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "argon"))
from r2p import gemini  # noqa: E402
from r2p.tasks import load_task  # noqa: E402
from _managed import digest, read_json  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--task", default="place_on_cell")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--provisional", action="store_true")
    a = ap.parse_args()
    lab = ROOT / "runs" / "lab" / a.run
    physics_path = lab / "sim" / "physics.json"
    held = read_json(lab / "heldout.json") if (lab / "heldout.json").exists() else {}
    result = held.get("result") or {}
    passed = bool(result.get("improved")) and bool(result.get("sealed"))
    out = ROOT / "runs" / "lab_to_game" / a.run
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "scripts" / "argon" / "build_game.py"), "--task", a.task, "--physics", str(physics_path),
           "--out", str(out / "game")]
    if passed:
        report = {"passed": True, "held_out": True, "lab_run": a.run, "result": result,
                  "episodes": held.get("episodes"), "by": "Argon Lab sealed held-out gate"}
        report_path = out / "heldout_gate.json"
        gemini.atomic_json(report_path, report)
        task = load_task(a.task)
        inputs = {"task": task, "physics": read_json(physics_path), "board": task.get("board", {}),
                  "camera": read_json(ROOT / "config" / "camera.json")}
        verification = {"schema": "r2p_twin_verification/1", "passed": True,
                        "input_sha256": {k: digest(v) for k, v in inputs.items()},
                        "evidence": [{"path": str(report_path),
                                      "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest()}]}
        gemini.atomic_json(out / "verification.json", verification)
        cmd += ["--verification", str(out / "verification.json")]
    elif a.provisional or a.dry_run:
        cmd += ["--allow-provisional-twin"]
    else:
        sys.exit(f"Lab run {a.run} has no improved sealed held-out result yet: {json.dumps(result)[:300]}")
    if a.dry_run:
        cmd += ["--dry-run"]
    print("twin verified by held-out gate:" if passed else "PROVISIONAL twin:", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd))
