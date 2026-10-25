#!/bin/zsh
# Open in Terminal on macOS, then grant Terminal camera access when prompted.
set -eu
cd "$(dirname "$0")/../.."
exec .venv/bin/python scripts/perception/camserver.py "$@"
