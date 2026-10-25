"""Run pure Luau acceptance tests with an explicit runner, never lune."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[2]

def runner():
    configured=os.environ.get("LUAU_BIN")
    candidates=[configured,shutil.which("luau"),str(Path.home()/"g4studio/backend/bin/luau.exe")]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():return candidate
    raise SystemExit("Set LUAU_BIN to the pure Luau executable")

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--luau",default=runner());args=parser.parse_args()
    tests=sorted((ROOT/"tests/roblox").glob("*_test.luau"))
    for test in tests:
        subprocess.run([args.luau,str(test)],cwd=ROOT,check=True)
    print(f"All {len(tests)} Luau suites passed")

if __name__=="__main__":main()
