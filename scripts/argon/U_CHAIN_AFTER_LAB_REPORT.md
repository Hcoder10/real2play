# Agent U: chain after Argon Lab

Implemented in **ecf5527** (`feat(argon): chain sealed Lab evidence through game and policy handoff`).
Completed the requested fake-Lab CPU chain and live Argon plan-only proof before the 15:10 deadline.
No arm movement, GPU job, game publication, or live managed game build was started by U.

Run this after `lab-real-08` has produced its sealed, improved held-out result:

```powershell
.venv/Scripts/python.exe -m r2p argon --task place_on_cell --lab-run lab-real-08 --approve owner-authorized --run u-live-after-lab-01 --request-budget 24 --wait-seconds 3600
```

Use a fresh `--run` on each invocation. `--lab-run` also accepts an explicit Lab directory.
The command has not been executed live: it would eventually move the arm. A missing/failed Lab gate
permits a clearly labelled provisional game build but cannot authorize real deployment or successful finish.
`--approve ask` remains the default. `owner-authorized` skips prompts and records the session authorization
in the ledger, including exact deployment executor, cell, supply position, decision and preflight result.

The Lab path is `twin_gate -> build_game -> collect_play -> train_plan -> replay_and_train -> deploy_real`.
It disables the old scene/calibration/active-experiment tools. The gate reads `heldout.json`, preserves
baseline/current numbers and replay residuals, checks the scored revision when supplied, and copies the
sim into the orchestration run. Both `physics.json` and `scene.xml`, all sim assets, held-out evidence,
task, camera, joint map and bench physics are hashed. Downstream operations reject changed inputs.
Synthetic Lab evidence is refused in a live chain.

Game construction uses the verification/context/Antigravity builder used by `scripts/lead/lab_to_game.py`,
including the actual local Luau and numerical parity checks. Publication is an optional separate receipt.
Collection snapshots only real `source=human` receiver traces from `roblox/out/traces`, excludes fixtures
and synthetic/headless claims, and separately generates labelled CPU SkillOperator players. Zero human
traces is reported as zero. L's physical replay/relabel path consumes the frozen Lab physics, followed by
expert collection with one worker. The game/player/replay path uses task-generated scenes with Lab
`physics.json`; arbitrary visual geometry from Lab `scene.xml` is retained and hashed, not transcompiled.

Live training writes `runs/argon_orchestrator/u-live-after-lab-01/train_request.json`, then waits.
It contains `name`, `trace_dirs`, `mixed_dir`, `physics_path`, `scene_path`, task and data hashes, and
an argument-vector `recipe` for the committed P2 trainer: 60,000 updates, batch 256, seeds 0/1/2,
width 512, three blocks, no augmentation, CUDA and 30 recorded evaluation scenes. The lead must copy
and rebase paths on the rig, rent the GPU through landlord, and apply its environment and memory fraction.
The orchestrator never launches that command and refuses to overwrite an existing training request.

Keep the orchestration process alive and atomically supply this receipt beside the request:

```json
{
  "request_sha256": "sha256(r2p.gemini._canonical(request).encode())",
  "export": "exports/the-returned-candidate",
  "incumbent": "exports/the-incumbent-for-this-task-revision"
}
```

`export` and optional `incumbent` must be accessible from this checkout. The request hash is printed in
the waiting status and ledger. The candidate must match the frozen task/physics and declare the 30 Hz,
absolute 15x6, footprint-frame contract, start pose, observation layout/constants and recorded eval.
The orchestrator independently evaluates clean and real-noise scenes, rejects training/evaluation overlap,
and freezes the evaluated export. Learned deployment requires >=0.8 in each condition on >=12 scenes,
improvement over the incumbent, and K's no-motion bench audit. Without an incumbent, the result is HOLD
and the executor is calibrated IK. Missing perception setup, incompatible bench physics, an unverified
map or a refused K audit also select IK before any attempt begins. A qualified learned attempt needs
`--perception-setup <saved-setup.json>` and, when applicable, `--bench-profile <current-profile.json>`.

Deployment asks Argon to read the actual board and select a legal turn, locates a fresh supply cap, and
uses either K's guarded streaming policy runner or the existing calibrated `ttt_armd.place()`.
`--board <json>` can supply the previous confirmed board; otherwise `out/ttt/status.json` is used.
The ledger records `learned_policy` or `calibrated_ik`. A persistent deployment-start marker prevents
retrying a possibly partially executed turn. There is no switch to IK after learned motion starts.
IK stall evidence is **not** labelled task success; its outcome remains independently unverified.
DONE in Lab mode means the chain and one turn completed, not that a learned policy transferred successfully.

The default controller is a stateless `gemini.Gemini` subclass with the Lab plain-language preface,
bounded filter/malformed-JSON/transient retries, first-complete-object parsing, and tool-error feedback.
It preserves disk cache, cache-only and daily-quota failures. The media helper encodes H.264 with >=2 s
padding, or uses a frame strip when ffmpeg is unavailable. Stateful Interactions remains explicit opt-in.
The legacy dry adapter also handles the newer active-patch schema with a fake client; no live-twin source changed.

The repeatable acceptance command creates a new, explicitly synthetic Lab directory by copying the named sim:

```powershell
.venv/Scripts/python.exe scripts/argon/u_chain_smoke.py --lab-run lab-real-08 --run u-next-smoke
```

Final acceptance `u-dry-final-01` copied `lab-real-08/sim` to `runs/lab/u-dry-final-01-fake/sim`, added
synthetic baseline/current held-out values, and invoked the actual `python -m r2p argon` CLI. It exited
zero in **16.875 s**, status **DONE**, synthetic=true. Fake Argon/managed responses, real CPU MuJoCo,
Luau checks, physical replay, two tiny CPU fits and MockRobot were used. The earlier independent CLI
run `u-dry-acceptance-01` also completed in **14.439 s**.

```text
ARGON twin_gate: passed=1, sealed=1, improved=1, episodes=2
ARGON build_game: passed=1, local_checks=4, parity_passed=1, published=0
HEADLESS 1/1: success=True, ticks=365
ARGON collect_play: traces=1, human=0, headless=1, rejected=0
ARGON train_plan: passed=1, experts=1, steps=2
ARGON replay_and_train: replayed=1, eval_n=1, clean=0, real=0, improved=0, policy_qualified=0
ARGON deploy_real: attempts=1, ticks=60, success=0
ARGON finish: gates_passed=1, stop_met=1
```

The smoke policy failed its task, as expected for two updates. The workflow pass is not policy qualification.
Full transcript and timing: [smoke transcript](../../runs/argon_orchestrator/u-dry-final-01/smoke_transcript.log),
[smoke result](../../runs/argon_orchestrator/u-dry-final-01/smoke.json),
[stage ledger](../../runs/argon_orchestrator/u-dry-final-01/ledger.jsonl).

The final live planning proof was:

```powershell
.venv/Scripts/python.exe -m r2p argon --task place_on_cell --lab-run lab-real-08 --approve owner-authorized --plan-only --request-budget 4 --run u-live-plan-02
```

It used **one uncached gemini-4-argon request, 11.344 s**, stateless=true, and executed no tool or motion.
Argon selected `twin_gate`, with `done_when` requiring `twin_gate.passed`, `.sealed` and `.improved` >=1.
Its reason: "Verify the Lab twin calibration results on the heldout set for lab-real-08 before proceeding to build_game."
The context was recorded bench files and current Lab status; this is not a freshly measured physical board.
Evidence: [plan.json](../../runs/argon_orchestrator/u-live-plan-02/plan.json).
An earlier one-request planning probe also chose `twin_gate`; its obsolete metric names prompted the
Lab-specific tool-description fix before the final proof.

Validation:

```powershell
.venv/Scripts/python.exe -m pytest -q tests/argon/test_lab_chain.py tests/argon/test_orchestrator.py tests/app/test_game_loop.py
# 68 passed in 100.07 s, including the legacy CPU end-to-end test.

.venv/Scripts/python.exe -m pytest -q tests/argon/test_lab_chain.py tests/argon/test_orchestrator.py tests/app/test_game_loop.py -k 'not offline_end_to_end'
# Final revision: 69 passed, 1 deselected in 8.35 s, including two added preflight-routing cases.
```

The final Lab CLI smoke independently covers the new complete path. Tests use fake model/arm endpoints;
the two planning probes were separate explicitly live CLI calls. Python compile checks and staged diff
whitespace checks passed. No repository-wide acceptance claim is made.

Files: `r2p/orchestrator.py`, `r2p/orchestrator_tools.py`, new `r2p/orchestrator_client.py`,
new `r2p/orchestrator_lab.py`, `scripts/argon/u_chain_smoke.py`, `tests/argon/test_lab_chain.py`, and this report.
The active Lab, jaw controller, policy/VLA and dashboard files were not edited. Pulled first; committed explicit paths only.

Remaining live work: Lab must actually seal and improve; lead supplies the leased training/export receipt;
human gameplay, paid game generation/publication and a physical turn remain unexecuted by U. A receipt wait
timeout preserves the request and stops. Resuming a terminated orchestration process is not implemented;
keep it alive while the lead trains. The new chain has been proved offline and planned live, not executed on hardware.
