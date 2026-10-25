# W: the full Argon pass, including B200 training and one turn

Implemented `--step-run` with six Argon-selected tools:
`experiment_sim -> build_game -> collect_play -> train -> evaluate -> deploy`.
The controller is the existing hardened stateless client, with low thinking in
step mode, a plain-language preface, bounded retries, cache/quota preservation,
and tool-error feedback. `owner-authorized` records authorization without prompts.
The previous Lab and legacy orchestration paths remain available.

## Exact live command — lead only, owner watching

```powershell
.venv/Scripts/python.exe -m r2p argon --task place_on_cell --step-run step_01 --approve owner-authorized --run w-live-step-01 --request-budget 32 --wait-seconds 5400 --headless 32 --experts 256 --cap 0
```

Use a fresh `--run`. `--step-run` accepts an existing step directory or its name;
`--step-run new` instead runs `scripts/lead/argon_step.py` first (40 turns by
default, configurable with `--experiment-turns`). W **did not execute this live
command**, make a real Gemini request, connect to the B200, start a GPU job,
publish a game, or move the arm.

The supplied rental/lease is specific to this session and expires at **17:30 PDT
on October 25**. The transport refuses to launch after that deadline. It uses
the supplied SSH host/port/key, lease `apt_8fc32d1f`, B200 device 0, and memory
fraction **0.949**. It does not extend the lease or rental, acquire another GPU,
or kill another job. Keep this orchestration process running while it retrieves
results. A timeout or failed launch is recorded and never automatically relaunched.

## What each stage actually does

- **Experiment:** copies `sim.json` and `ledger.jsonl` into an immutable run
  snapshot, preserves measured arm offsets, and converts physics using the
  `after_step.py` convention. World-frame offsets are not added elsewhere.
- **Game:** invokes `scripts/argon/build_game.py`, including its Antigravity
  managed build and local Luau/parity checks. The result is explicitly
  **provisional and unpublished** because step observations are not a held-out
  fidelity gate. This path does not manufacture a verification receipt.
- **Play:** snapshots and deduplicates genuine receiver `source=human` traces,
  rejecting synthetic/bot/test evidence. Zero humans remains zero. Headless
  players run on the B200 as the labelled Python SkillOperator port in MuJoCo;
  they are not described as humans or Roblox-engine execution.
- **Train:** physically replays play, collects expert traces, and launches P2's
  actual recipe on the B200: 60,000 steps, batch 256, seeds 0/1/2, width 512,
  three blocks, no augmentation, CUDA and 30 evaluation scenes. It checks Git
  history for a committed expert change after the original bench-profile
  implementation (`e8cd98b`). If available it uses the P3 bench profile;
  otherwise it records that the default expert is being used. P3 was still
  uncommitted during W's smoke. Bench traces and game traces have incompatible
  start/limit conventions: accepted play scenes are physically re-demonstrated
  by the bench expert, with explicit source/replay lineage. Default-mode
  training mixes the actual replayed play actions with expert traces. No
  replay acceptance or no successful expert trace stops training.
- **Evaluate:** checks the returned export's task/physics, absolute 15x6/30 Hz
  footprint-frame contract, start pose, observation metadata, weight hash, and
  all three recorded 30-scene noise evaluations. The B200 runs
  `scripts/robot/rehearse_policy.py` against the frozen real map, bench physics,
  streaming profile and daemon limits, for all nine cells at the selected cap.
  This is simulation and sends no commands to the real daemon. The evaluated
  export hash must match the local export.
- **Deploy:** all nine rehearsal goals must pass to select the learned runner.
  Its command includes `--argon-perception --transport armd --real
  --allow-unverified`, the profile and cap; it omits `--cell` so Argon chooses
  from the observed board. Otherwise Argon reads the board and supply and calls
  the proven `scripts/lead/ttt_armd.py` `place()` exactly once, including its
  measured held-cap offset. The ledger names the executor and reason. A
  persistent start marker prevents any retry or switch to IK after a possible
  learned motion. IK completion is recorded with **unverified task success**.

The B200 receives a Git bundle of committed code into a fresh detached worktree
under `/root/real2play-w/<run>`; the shared checkout is not reset. Frozen inputs
travel separately. An atomic result archive refreshes every 20 seconds while
outputs are produced, and the laptop polls/copies it every 15 seconds. Logs,
traces, replay evidence, export files and evaluation results are retained under
`runs/argon_orchestrator/<run>/remote`; the export is also copied to
`exports/<run>-policy`. These are active copies home, not `save_on_return`.

`DONE` means the six stages and one turn completed. It does not mean the twin
is verified, the game is published, or a learned policy transferred successfully.

## Acceptance

```powershell
.venv/Scripts/python.exe scripts/argon/w_chain_smoke.py --run w-final-smoke-02
# PASS, DONE, synthetic=true, 4.544 seconds

.venv/Scripts/python.exe -m pytest -q tests/argon/test_step_chain.py tests/argon/test_lab_chain.py tests/argon/test_orchestrator.py -k 'not offline_end_to_end'
# 74 passed, 1 deselected in 3.19 seconds
```

The smoke uses fake Argon, fake SSH, the actual offline managed-builder checks,
and **MockRobot**. It deliberately fails the fake rehearsal and proves the
single-turn fallback route. Tests separately cover the learned branch, failed
and partial learned attempts without fallback, immutable export/input checks,
authorization, remote launch-once behavior, lease-prefixed commands, archive
path validation, human provenance, P2 recipe, and bench play re-demonstration.
There is no local CPU training/headless collection in this step-mode smoke.
The old heavy CPU end-to-end test was excluded to honor the laptop constraint.

Evidence:

- [Final smoke result](../../runs/argon_orchestrator/w-final-smoke-02/smoke.json)
- [Transcript](../../runs/argon_orchestrator/w-final-smoke-02/smoke_transcript.log)
- [Stage ledger](../../runs/argon_orchestrator/w-final-smoke-02/ledger.jsonl)

An additional broader app run returned **90 passed, 1 failed, 1 deselected**.
The failure is `tests/app/test_game_loop.py::test_player_port_matches_actual_luau`:
the concurrent Roblox working tree differs by 1.81689 rad. The same trace
passes against isolated **HEAD** Roblox source (8.88e-16 rad; zero phase/command
mismatches). W did not edit those files. Comparison evidence is in
`runs/argon_orchestrator/w-smoke-01/{unrelated_player_parity,baseline_player_parity}.json`.
This integration issue needs the Roblox owner's attention before claiming
Python/Luau player parity for that working-tree revision.

## Files and remaining live work

Changed `r2p/orchestrator.py`, `r2p/orchestrator_client.py`; added
`r2p/orchestrator_step.py`, `r2p/orchestrator_remote.py`,
`scripts/argon/w_chain_smoke.py`, `tests/argon/test_step_chain.py`, and this report.
Pulled first; commit contains explicit W paths only. No edits to `scripts/lead`,
`r2p/lab`, `dashboard`, simulator, policy trainer, or Roblox source.

Remaining: the lead's watched live invocation, actual remote training/rehearsal,
real human collection and one physical turn. Live SSH execution has not been
claimed as tested by the fake SSH smoke. Restart/resume of a stopped orchestration
process is not implemented; launch markers deliberately prevent blind retries.
The arbitrary objects/visual geometry in the step sim are retained as evidence;
the game/training scene still comes from the task spec with converted physics.
