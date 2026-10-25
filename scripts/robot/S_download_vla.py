"""Stage pinned pretrained weights and tokenizer on a connected workstation."""
from pathlib import Path
import json
import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import HfApi, snapshot_download

root = Path("runs/S_vla_assets")
root.mkdir(parents=True, exist_ok=True)
receipt = {}
for repo, folder, patterns in [
    ("lerobot/smolvla_base", "base", ["*.json", "*.safetensors"]),
    ("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "backbone", ["*.json", "*.txt", "*.model", "*.jinja"]),
]:
    revision = HfApi().model_info(repo).sha
    print(repo, revision, flush=True)
    snapshot_download(repo, revision=revision, local_dir=root/folder, allow_patterns=patterns)
    receipt[folder] = {"repo": repo, "revision": revision}
(root/"provenance.json").write_text(json.dumps(receipt, indent=2))
print("Downloaded", root, flush=True)
