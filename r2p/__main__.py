import sys

from .cli import main

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "argon":
        from .orchestrator import main as argon_main
        raise SystemExit(argon_main(sys.argv[2:]))
    raise SystemExit(main())
