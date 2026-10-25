"""After Argon's step loop: Argon's sim -> game -> play data -> twin replay (+ CPU check). The rig retrain and the real
deploy are launched by the lead right after (commands printed at the end).

  python scripts/lead/after_step.py --run step_01
"""
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable


def sh(cmd, log):
    print(">>", " ".join(map(str, cmd)), flush=True)
    with open(log, "w") as f:
        rc = subprocess.call(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    print("   exit", rc, "->", log, flush=True)
    return rc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--headless", type=int, default=32)
    a = ap.parse_args()
    step = ROOT / "runs" / "argon_step" / a.run
    out = ROOT / "runs" / "after_step" / a.run
    out.mkdir(parents=True, exist_ok=True)
    sim = json.loads((step / "sim.json").read_text())

    # 1. Argon's sim -> physics config for the game and the twin (base physics + Argon's joint/object patches).
    base = json.loads((ROOT / "config/physics/so101_bench.json").read_text())
    physics = copy.deepcopy(base)
    for name, patch in (sim.get("physics") or {}).items():
        if name in physics.get("joints", {}) and isinstance(patch, dict):
            physics["joints"][name].update({k: v for k, v in patch.items() if isinstance(v, (int, float))})
        elif name in physics and isinstance(patch, (int, float, dict)):
            physics[name] = patch if not isinstance(patch, dict) else {**physics[name], **patch}
    caps = [o for o in sim.get("objects", []) if "cap" in str(o.get("id", "")).lower()]
    if caps and caps[0].get("mass_kg"):
        physics.setdefault("object", {})["mass_kg"] = caps[0]["mass_kg"]
    if caps and caps[0].get("friction"):
        physics.setdefault("object", {})["friction"] = caps[0]["friction"]
    physics["notes"] = f"Argon step-loop calibrated ({a.run}): {json.dumps(sim.get('arm', {}))[:400]}"
    phys_path = ROOT / "config" / "physics" / f"so101_bench.argon_{a.run}.json"
    phys_path.write_text(json.dumps(physics, indent=1))
    (out / "argon_sim.json").write_text(json.dumps(sim, indent=1))
    print("physics ->", phys_path, flush=True)

    # 2. Sim -> game: the Antigravity managed agent builds the level from Argon's sim (local Luau + parity gates).
    sh([PY, "scripts/argon/build_game.py", "--task", "place_on_cell", "--physics", phys_path,
        "--out", out / "game", "--allow-provisional-twin"], out / "build_game.log")

    # 3. Game -> data -> twin replay/relabel: real players from the receiver + labelled headless players.
    sh([PY, "-m", "r2p", "loop", "--from-game", "roblox/out/traces", "--collect-headless", a.headless,
        "--task", "place_on_cell", "--output", out / "loop", "--steps", 4000, "--eval-n", 12], out / "loop.log")
    print(json.dumps({"physics": str(phys_path), "game": str(out / "game"), "loop": str(out / "loop")}, indent=1))
