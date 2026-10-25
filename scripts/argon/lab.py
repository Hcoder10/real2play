"""Argon Lab entry point. Defaults to per-call owner approval on real hardware."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from r2p.lab.agent import LabAgent, StatefulGemini, fake_client  # noqa: E402
from r2p.tasks import load_task  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="task JSON path or registered task name")
    parser.add_argument("--run", required=True)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--approve", choices=("ask", "auto-sim"), default="ask")
    parser.add_argument("--real-mock", action="store_true", help="Private hidden-truth MuJoCo backend; no hardware")
    parser.add_argument("--fake-argon", action="store_true", help="Offline one-turn smoke response; requires --real-mock")
    parser.add_argument("--request-budget", type=int, default=40)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--serve-only", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.run):
        parser.error("run must be a simple name without directory separators")
    if args.fake_argon and not args.real_mock:
        parser.error("--fake-argon requires --real-mock")
    if args.fake_argon and args.max_turns != 1:
        parser.error("--fake-argon smoke supports --max-turns 1; integration tests exercise multiple turns")
    run = ROOT/"runs/lab"/args.run
    if (run/"ledger.jsonl").exists():
        parser.error("Run already has a ledger; choose a new run name to preserve evidence")
    task = load_task(args.task)
    client = fake_client(run/"fake-cache") if args.fake_argon else StatefulGemini(
        max_requests=args.request_budget, timeout_s=args.timeout, retries=0)
    agent = LabAgent(run, task, client=client, real_mock=args.real_mock, approve=args.approve)
    try:
        address = agent.serve(args.port)
        print(f"Argon Lab HTTP http://{address[0]}:{address[1]} | {'MOCK' if args.real_mock else 'REAL'}", flush=True)
        if args.serve_only:
            agent.server_thread.join()
            return 0
        status = agent.run_loop(args.max_turns)
        print(json.dumps(status, indent=2))
        return 0 if status.get("complete") or args.real_mock and status["status"] == "incomplete" else 2
    finally:
        agent.close()


if __name__ == "__main__":
    raise SystemExit(main())
