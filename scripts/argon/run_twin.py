"""Fit a calibrated recorded session, publishing atomic r2s_live/1 artifacts."""
import _bootstrap  # noqa: F401
import argparse
import json


def main(argv=None):
    from r2p.real2sim import run_twin

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session")
    parser.add_argument("--task")
    parser.add_argument("--camera")
    parser.add_argument("--out")
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--anchors", choices=("auto", "hard", "soft"), default="auto")
    parser.add_argument("--max-calls", type=int, default=12)
    parser.add_argument("--budget-s", type=float, default=1800)
    args = parser.parse_args(argv)
    result = run_twin(args.session, task=args.task, camera=args.camera, out_dir=args.out,
                      iterations=args.iters, patience=args.patience, anchors=args.anchors,
                      max_calls=args.max_calls, budget_s=args.budget_s)
    print(json.dumps({key: result[key] for key in ("run_dir", "best_iter", "stop_reason", "metrics")}))
    return 1 if result["stop_reason"] in ("quota", "cache_miss", "request_failed", "invalid_reply") else 0


if __name__ == "__main__":
    raise SystemExit(main())
