"""Label an r2p_trace/1 episode using task commands and observed motion/gripper phases."""
import _bootstrap  # noqa: F401
import argparse
import json


def main(argv=None):
    from r2p.labeling import label_trace

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace")
    parser.add_argument("output")
    parser.add_argument("--task")
    parser.add_argument("--refine", action="store_true", help="opt in to one MAIN-role refinement")
    args = parser.parse_args(argv)
    print(json.dumps(label_trace(args.trace, args.output, args.task, refine=args.refine)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
