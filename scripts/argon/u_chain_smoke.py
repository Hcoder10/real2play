"""Copy a Lab sim into an explicitly synthetic fixture and exercise the CPU chain."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from r2p import gemini  # noqa: E402
from r2p.orchestrator_lab import lab_revision  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab-run", required=True, help="Lab name or directory to copy, read-only")
    parser.add_argument("--run", default=time.strftime("u-smoke-%Y%m%d-%H%M%S"))
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,90}", args.run):
        parser.error("Unsafe run name")
    source = Path(args.lab_run)
    source = source if source.is_absolute() else ROOT / "runs/lab" / source
    fake = ROOT / "runs/lab" / (args.run + "-fake")
    revision = lab_revision(source / "sim")
    shutil.copytree(source / "sim", fake / "sim")
    if lab_revision(fake / "sim") != revision or lab_revision(source / "sim") != revision:
        raise RuntimeError("Source Lab changed while copying; retry with a new smoke name")
    held = {"synthetic": True, "sealed": True, "episodes": ["synthetic-a", "synthetic-b"],
        "result": {"sealed": True, "improved": True, "revision": revision,
            "baseline": {"joint_rms_deg": 1.1, "jaw_path_rms_mm": 5., "object_final_pose_error_mm": 4.,
                         "image_similarity": .7, "episodes": 2},
            "current": {"joint_rms_deg": .8, "jaw_path_rms_mm": 3., "object_final_pose_error_mm": 3.,
                        "image_similarity": .8, "episodes": 2}}}
    gemini.atomic_json(fake / "heldout.json", held)
    gemini.atomic_json(fake / "status.json", {"real_mock": True, "synthetic": True,
        "source": str(source), "note": "Fabricated gate metrics for workflow smoke ONLY; no real fidelity claim"})
    command = [sys.executable, "-m", "r2p", "argon", "--task", "place_on_cell", "--lab-run", fake.name,
               "--dry-run", "--approve", "owner-authorized", "--run", args.run]
    print("CPU smoke: " + subprocess.list2cmdline(command), flush=True)
    start = time.monotonic()
    log = []
    with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace") as process:
        for line in process.stdout:
            log.append(line)
            print(line, end="", flush=True)
        code = process.wait()
    elapsed = time.monotonic()-start
    out = ROOT / "runs/argon_orchestrator" / args.run
    (out / "smoke_transcript.log").write_text("".join(log), encoding="utf-8")
    report = {"command": command, "fake_lab": str(fake), "elapsed_s": elapsed,
              "exit_code": code, "under_three_minutes": elapsed < 180, "synthetic": True}
    gemini.atomic_json(out / "smoke.json", report)
    print(json.dumps(report, indent=2))
    return code if code else (0 if elapsed < 180 else 1)


if __name__ == "__main__":
    raise SystemExit(main())
