"""Compute-only Antigravity stages with immutable inputs and local verification.

The returned patch is bounded JSON, never executable code. Held-out observations
are kept off the managed sandbox and are scored only by the local evaluator.
"""
from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import zipfile

from r2p import gemini
from r2p.agents import FakeAgent, ManagedAgent, validate_bundle
from r2p.orchestrator import digest, object_schema


def hashes(files):
    return {name: hashlib.sha256(text.encode()).hexdigest() for name, text in files.items()}


def run_candidate(stage, files, candidate_name, fixture, folder, *, dry, instructions, verify, agent=None):
    """Only a single declarative file can cross the sandbox boundary."""
    folder.mkdir(parents=True, exist_ok=True)
    expected = hashes(files)
    prompt = ("Use Gemini 4 Argon with high reasoning effort. Work in /workspace. All supplied sources are immutable. "
              "Do not access hardware, publish, obtain credentials, or start GPU work. Return ONLY JSON "
              "{files:{relative_path:complete_UTF8_contents},tests:[{name,passed,output}],report,source_sha256}. "
              "Return exactly " + candidate_name + ". Read the actual immutable files to compute their SHA256; "
              "report failed tests honestly. Never modify test or source files. Run python verify_candidate.py "
              "on your candidate; fix it until it passes. " + instructions)
    fixture_bundle = {"files": {candidate_name: json.dumps(fixture)}, "tests": [
        {"name": "offline fixture; independent local verification required", "passed": True}],
        "source_sha256": expected, "report": "Synthetic managed-agent fixture; no Google request."}
    agent = agent or (FakeAgent(fixture_bundle, cache_dir=folder / "cache", source_files=files) if dry else
                      ManagedAgent(folder / "cache", source_files=files, max_requests=15, poll_interval_s=10, timeout_s=600))
    bundle = agent.run(prompt, tag=stage)
    validate_bundle(bundle)
    gemini.atomic_json(folder / "bundle.json", bundle)
    if set(bundle["files"]) != {candidate_name}:
        raise ValueError("Managed candidate attempted an out-of-scope file patch")
    if bundle.get("source_sha256") != expected:
        raise ValueError("Managed source integrity check failed")
    if not all(row["passed"] for row in bundle["tests"]):
        raise ValueError("Managed sandbox tests failed")
    candidate = gemini.parse_json(bundle["files"][candidate_name])
    checked = verify(candidate)
    gemini.atomic_json(folder / candidate_name, candidate)
    report = {"stage": stage, "passed": True, "candidate": candidate_name, "verification": checked,
              "record": bundle["record"], "synthetic": dry, "source_sha256": expected,
              "sandbox_tests": bundle["tests"]}
    gemini.atomic_json(folder / "result.json", report)
    managed = {"agent": bundle["record"]["agent"], "agent_id": bundle["record"]["interaction_id"],
               "duration_s": bundle["record"]["latency"], "tests_passed": len(bundle["tests"]) + 1, "synthetic": dry}
    return candidate, checked, managed


def parameter_schema():
    from r2p.real2sim.active import PARAMETERS
    return object_schema({name: {"type": "number", "minimum": lo, "maximum": hi}
                          for name, (lo, hi) in PARAMETERS.items()}, PARAMETERS)


def analyze(backend, folder):
    from r2p.real2sim.active import initial_parameters, materialize, score_observations, keep_heldout
    training = [o for o in backend.observations if o.split == "train"]
    heldout = [o for o in backend.observations if o.split == "heldout"]
    if not training or not heldout or {o.session.path for o in training} & {o.session.path for o in heldout}:
        raise ValueError("Need disjoint training and held-out experiments")
    physics = deepcopy(backend.physics)
    physics.pop("active_twin", None)  # Metadata embeds holdout scores; do not send it to the optimizer.
    scene = deepcopy(backend.scene)
    schema = parameter_schema()
    baseline_training = score_observations(training, physics, scene, previews=folder / "training_real_sim")
    files = {"physics.json": json.dumps(physics), "scene.json": json.dumps(scene),
             "residuals.json": json.dumps(baseline_training), "parameter_schema.json": json.dumps(schema),
             "image_summary.json": json.dumps({"source": "real|sim training overlays only",
                 "frames": len(list((folder / "training_real_sim").glob("*.jpg"))),
                 "per_object": baseline_training["objects"], "rows": baseline_training["rows"]})}
    # Recorded commands and physical sources allow sandbox replay, with no access
    # to held-out frames/targets, local paths, credentials, or arbitrary execution.
    for i, observation in enumerate(training):
        files[f"experiments/train_{i}.json"] = json.dumps(observation.session.ticks)
    for path in (backend.root / "r2p").rglob("*.py"):
        if path.name == "agents.py" or path.name.startswith("orchestrator"):
            continue
        files[path.relative_to(backend.root).as_posix()] = path.read_text(encoding="utf-8")
    files["r2p/__init__.py"] = ""
    if not backend.dry:
        # Inline Sources accept text. The archive is generated here from trusted
        # repository assets and restored by the immutable validator, never by a returned script.
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
            for path in (backend.root / "assets/so101").rglob("*"):
                if path.is_file():
                    zipped.writestr(path.relative_to(backend.root).as_posix(), path.read_bytes())
        files["robot_assets.zip.base64"] = base64.b64encode(archive.getvalue()).decode()
    files["verify_candidate.py"] = '''import base64, io, json, pathlib, zipfile
import numpy as np
from r2p import gemini
from r2p.sim.twin import TaskTwin
from r2p.real2sim.active import materialize
root = pathlib.Path.cwd()
archive = root / 'robot_assets.zip.base64'
if archive.exists():
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(archive.read_bytes()))) as source:
        for name in source.namelist():
            assert (root/name).resolve().is_relative_to((root/'assets/so101').resolve())
        source.extractall(root)
read = lambda name: json.loads(pathlib.Path(name).read_text())
candidate = read('parameters.json')
assert not gemini.schema_errors(candidate, read('parameter_schema.json'))
physics, scene = materialize(read('physics.json'), read('scene.json'), candidate)
errors = []
for path in sorted(pathlib.Path('experiments').glob('*.json')):
    ticks = read(path)
    with TaskTwin(scene, physics, q0=ticks[0]['q']) as twin:
        for tick in ticks:
            errors.extend((np.array(tick['q'])-twin.q())[:5].tolist())
            twin.step(tick['q_target'])
assert errors and np.isfinite(errors).all()
print(json.dumps({'ticks':len(errors)//5, 'joint_rms_rad':float(np.sqrt(np.mean(np.square(errors))))}))
'''
    selected = {}
    def verify(candidate):
        errors = gemini.schema_errors(candidate, schema)
        if errors:
            raise ValueError("Physics parameter patch rejected: " + "; ".join(errors))
        proposed_physics, proposed_scene = materialize(physics, scene, candidate)
        before = score_observations(heldout, physics, scene)
        after = score_observations(heldout, proposed_physics, proposed_scene, previews=folder / "candidate_heldout")
        verdict = keep_heldout(after, before)
        selected.update(physics=proposed_physics if verdict["kept"] else physics,
                        scene=proposed_scene if verdict["kept"] else scene)
        return {"before": before, "after": after, "verdict": verdict, "local_tests_passed": 1}
    candidate, checked, managed = run_candidate("analyze_experiments", files, "parameters.json",
        initial_parameters(scene), folder, dry=backend.dry,
        instructions="Install numpy scipy opencv-python-headless mujoco. Propose bounded relative stiffness/damping and "
                     "absolute cap/table material values using residuals and the recorded training experiments. "
                     "Inspect r2p/sim/twin.py and physics.json; run replay tests in verify_candidate.py. "
                     "No Python source edits are accepted; physics parameter changes are the patch surface.", verify=verify)
    backend.physics, backend.scene = selected["physics"], selected["scene"]
    gemini.atomic_json(folder / "physics.json", backend.physics)
    gemini.atomic_json(folder / "scene.json", backend.scene)
    return {"metrics": {"passed": 1, "kept": int(checked["verdict"]["kept"]),
                        "before_error": checked["before"]["error"], "candidate_error": checked["after"]["error"]},
            "managed": managed, "files": [str(folder / "physics.json"), str(folder / "scene.json")],
            "activation": "run-local only after held-out improvement; global physics unchanged"}


def training_plan(backend, folder):
    count = backend.r.state["results"]["collect_play"]["metrics"]["traces"]
    fixture = {"experts": 1 if backend.dry else 200, "steps": 2 if backend.dry else 10000,
               "eval_n": 1 if backend.dry else 12, "game_traces": count, "hz": 30, "horizon": 15,
               "action": "absolute_q", "task_sha256": digest(backend.task), "physics_sha256": digest(backend.physics),
               "device": "cpu" if backend.dry else "lead_leased_gpu"}
    schema = object_schema({
        "experts": {"type": "integer", "minimum": 1, "maximum": 2 if backend.dry else 1000},
        "steps": {"type": "integer", "minimum": 1, "maximum": 5 if backend.dry else 100000},
        "eval_n": {"type": "integer", "minimum": 1 if backend.dry else 12, "maximum": 2 if backend.dry else 100},
        **{k: {"enum": [v]} for k, v in fixture.items() if k not in ("experts", "steps", "eval_n")}}, fixture)
    files = {"task.json": json.dumps(backend.task), "physics.json": json.dumps(backend.physics),
             "data.json": json.dumps({"game_traces": count, "synthetic": backend.dry}),
             "training_schema.json": json.dumps(schema), "r2p/__init__.py": "",
             "r2p/gemini.py": (backend.root / "r2p/gemini.py").read_text(encoding="utf-8"),
             "verify_candidate.py": "import json\nfrom r2p.gemini import schema_errors\n"
                "read=lambda p:json.load(open(p))\nerrors=schema_errors(read('training.json'),read('training_schema.json'))\n"
                "assert not errors, errors\nprint('Training config and frozen data contract passed')\n"}
    def verify(candidate):
        errors = gemini.schema_errors(candidate, schema)
        if errors:
            raise ValueError("Training config rejected: " + "; ".join(errors))
        return {"passed": True, "tests": 1, "game_traces": count, "experts": candidate["experts"]}
    candidate, checked, managed = run_candidate("train_plan", files, "training.json", fixture, folder,
        dry=backend.dry, instructions="Author a command-conditioned training config within training_schema.json. "
        "Mix all accepted game traces with the requested number of experts, preserving commanded-q observations "
        "and absolute 15x6 targets. Tiny CPU smoke in dry mode; production GPU execution belongs to the lead. "
        "Use disjoint held-out scenes with clean and real noise. Validate the config; do not launch training.", verify=verify)
    backend.training = candidate
    return {"metrics": {"passed": 1, "experts": candidate["experts"], "game_traces": count, "steps": candidate["steps"]},
            "managed": managed, "config": str(folder / "training.json")}
