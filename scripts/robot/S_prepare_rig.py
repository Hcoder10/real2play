"""Offline dependency/weight staging in the agent's private RAM-disk workspace."""
from pathlib import Path
import argparse
import json
import shutil
import zipfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--root", type=Path, required=True)
p.add_argument("--wheels", type=Path, required=True)
p.add_argument("--cache", type=Path, required=True, help="HF hub cache directory")
a = p.parse_args()
root = a.root
deps = root/"deps"
deps.mkdir(parents=True, exist_ok=True)
wheels = list(a.wheels.glob("*.whl"))
packages = [w.name.split("-")[0] for w in wheels]
if len(packages) != len(set(packages)):
    raise ValueError("wheel directory contains conflicting versions; use the pinned VLA dependencies")
for wheel in wheels:
    with zipfile.ZipFile(wheel) as z:
        z.extractall(deps)
cache = a.cache
assets = root/"assets_vla"
assets.mkdir(exist_ok=True)
receipt = {}
for repo, folder, revision in [
    ("lerobot/smolvla_base", "base", "c83c3163b8ca9b7e67c509fffd9121e66cb96205"),
    ("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "backbone", "7b375e1b73b11138ff12fe22c8f2822d8fe03467"),
]:
    source = cache/("models--"+repo.replace("/", "--"))/"snapshots"/revision
    destination = assets/folder
    destination.mkdir(exist_ok=True)
    for item in source.iterdir():
        if item.is_file() and (item.suffix in (".json", ".txt", ".model", ".jinja") or (folder == "base" and item.name == "model.safetensors")):
            shutil.copy2(item, destination/item.name)
    receipt[folder] = {"repo": repo, "revision": revision, "source": "pre-existing upstream HF cache; no finetuned weights"}
(assets/"provenance.json").write_text(json.dumps(receipt, indent=2))
print("staged", assets, flush=True)
