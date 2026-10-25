"""Read-only preflight. Never calibrates, configures, moves or toggles torque."""
import _bootstrap  # noqa: F401
import argparse
from r2p.robot.config import load_config
from r2p.robot.preflight import run_preflight


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--policy")
    parser.add_argument("--tests", action="store_true", help="tests run by default")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args(argv)
    policy = None
    if args.policy == "stub":
        from r2p.model import Arm
        from r2p.tasks import load_task
        from r2p.robot.dryrun import ScriptedPolicy
        policy = ScriptedPolicy(load_task("cube_in_bowl"), Arm())
    elif args.policy:
        from r2p.robot.policy import load_policy, policy_config
        try:
            policy = load_policy(args.policy)
            policy.config = policy_config(policy, args.policy)
        except Exception as exc:
            print(f"FAIL policy: {exc}")
            return 1
    try:
        report = run_preflight(load_config(args.config), policy=policy, tests=not args.skip_tests,
                               allow_unverified=args.allow_unverified, mock=args.mock)
    except Exception as exc:
        print(f"FAIL preflight: {exc}")
        return 1
    report.emit()
    return int(report.failed)


if __name__ == "__main__":
    raise SystemExit(main())
