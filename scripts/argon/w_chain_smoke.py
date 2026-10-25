"""Repeatable <3 minute W acceptance: fake Argon, fake SSH, real MockRobot."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="w-smoke-" + uuid.uuid4().hex[:10])
    args = parser.parse_args(argv)
    command = [sys.executable, "-m", "r2p", "argon", "--task", "place_on_cell", "--step-run", "step_01",
               "--approve", "owner-authorized", "--dry-run", "--run", args.run]
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT,
                            text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=175)
    elapsed = time.perf_counter()-started
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    folder = ROOT / "runs/argon_orchestrator" / args.run
    if not folder.is_dir():
        return result.returncode or 1
    (folder / "smoke_transcript.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (folder / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    fake_stages = [r["stage"] for r in rows if r["kind"] == "fake_ssh"]
    deploy = status.get("results", {}).get("deploy", {})
    passed = (result.returncode == 0 and elapsed < 180 and status["status"] == "DONE" and status["synthetic"]
              and fake_stages == ["collect_play", "train", "evaluate"] and deploy.get("transport") == "MockRobot"
              and deploy.get("metrics", {}).get("attempts") == 1)
    report = {"passed": passed, "seconds": elapsed, "command": command, "synthetic": True,
              "fake_argon": True, "fake_ssh_stages": fake_stages, "robot": deploy.get("transport"),
              "live_requests": 0, "real_motion": False, "gpu_work": False, "status": status["status"]}
    (folder / "smoke.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
