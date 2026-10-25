"""Plan experiments or identify physics from disjoint recorded attempts. Never runs hardware."""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path


def main(argv=None):
    from r2p.sysid import Parameter, fit_parameters, plan_experiments

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("residuals", help="JSON residual values, tolerances and object/joint ids")
    fit = sub.add_parser("fit")
    fit.add_argument("traces", nargs="+")
    fit.add_argument("--heldout", nargs="+", required=True)
    fit.add_argument("--physics", default="so101_bench")
    fit.add_argument("--parameters", help="JSON [{path,low,high}] for a focused identifiable experiment")
    fit.add_argument("--output", help="new physics version; written only after the gate passes")
    fit.add_argument("--max-nfev", type=int, default=100)
    args = parser.parse_args(argv)
    if args.command == "plan":
        result = plan_experiments(json.loads(Path(args.residuals).read_text(encoding="utf-8")))
        print(json.dumps(result))
        return 0
    parameters = [Parameter(**item) for item in json.loads(Path(args.parameters).read_text())] if args.parameters else None
    result = fit_parameters(args.traces, args.physics, heldout=args.heldout, parameters=parameters,
                             max_nfev=args.max_nfev, output=args.output)
    print(json.dumps(result))
    return 0 if result["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
