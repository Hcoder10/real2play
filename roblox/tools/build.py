"""Local publish gate: fresh presets, Studio engine parity, then Rojo build.

Publishing itself remains an owner action in Studio. This command never closes
Studio, never uploads a place, and refuses to build after a failed parity gate.
"""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"roblox/real2play.rbxl")
    args=parser.parse_args()
    def call(*command):subprocess.run(command,cwd=ROOT,check=True)
    call(sys.executable,str(ROOT/"roblox/tools/gen_presets.py"),"--check")
    call("node",str(ROOT/"roblox/tools/sync.mjs"))
    call(sys.executable,str(ROOT/"roblox/tools/parity.py"),"--studio","--out",str(ROOT/"roblox/out/publish_parity.json"))
    call("rojo","build",str(ROOT/"roblox/default.project.json"),"-o",str(args.output.resolve()))

if __name__=="__main__":main()
