# Agent I: active twin

Implemented `r2p/real2sim/active.py`, `scripts/argon/active_twin.py`, and
`tests/argon/test_active_twin.py`. The implementation uses the existing physical
MuJoCo `TaskTwin`, calibrated CPU renderer, `fit_pose`, command replay, Gemini
role/cache client, and `PATCH_SCHEMA`. No other agent's source files were edited.

The menu covers holds, joint sweeps, straight pushes, and physical pick/lift/place.
Every interpolated joint target is checked for reach, height, mesh clearance,
speed, calibrated limits when available, object interference, and protected grid
cells. Successful plans end above the safe height. A refused/aborted live command
causes a hold, not an unvalidated retreat.

The campaign records three complete held-out experiments first, then spends the
remaining experiment budget on MAIN-selected training experiments. Invalid MAIN
choices fall back to a validated, informative, untried menu entry. Numeric fitting
uses bounded servo optimization plus a friction grid search before local contact
refinement. Camera pose noise and joint sensor resolution have separate weights.
Only held-out error and per-object visual non-regression decide which candidates
are kept. The optimizer and patch prompts cannot receive held-out observations.
Quota and cache-only errors propagate without a silent role fallback.

Outputs under `runs/active_twin/<run>/` include atomic `status.json`, residual
tables, per-iteration real|sim overlays, complete trace/frame sessions, a final
report, and a companion candidate scene. The inactive physics candidate is
`config/physics/so101_bench.<run>.json`. Scene material values must accompany the
physics candidate because explicit per-object/table values override defaults.

## Commands and evidence

```powershell
.venv/Scripts/python.exe scripts/argon/active_twin.py --mover sim --budget 6 --run agent-i-sim-final --max-nfev 7
.venv/Scripts/python.exe scripts/argon/active_twin.py --mover armd --budget 6 --plan-only --run agent-i-armd-plan
.venv/Scripts/python.exe -m pytest -q tests/core tests/argon tests/perception tests/robot tests/app tests/sim tests/policy
```

The completed `agent-i-sim-final` CLI campaign and independent acceptance test
both used six experiments, zero live Gemini requests, no GPU, and no hardware
writes. They recovered stiffness 1.349999999,
cap friction 0.450761240, and cap mass 0.004008152 kg against hidden truth 1.35,
0.45, and 0.004 kg. Table friction remained explicitly unidentifiable. Its
held-out normalized error fell from 1.163470 to 0.063101 (94.58%). The first push's
0.35 friction proposal was rejected on held-out regression; the opposite push
then resolved the fit. The test asserts stiffness within 0.02, friction within
0.04, mass within 0.5 g, and at least a 50% held-out improvement.

| Held-out metric | Before | After (`agent-i-sim-final`) |
|---|---:|---:|
| Normalized error | 1.163470 | 0.063101 |
| Joint RMS (degrees) | 0.138615 | 0.000261 |
| Jaw RMS (mm) | 1.292843 | 0.002756 |
| Vertical jaw residual (mm) | 1.161378 | 0.001291 |
| Final object pose (mm) | 1.109961 | 0.286131 |
| Mean mask IoU | 0.991086 | 0.998013 |

Final artifacts: `runs/active_twin/agent-i-sim-final/report.md`,
`status.json`, `residuals.json`, `candidate_scene.json`, `iter_03/numeric/*.jpg`,
and `config/physics/so101_bench.agent-i-sim-final.json`. Earlier development
runs remain separate and are not the final candidate.

The live task's board and supply lane were recalibrated during this work. Holds
now follow the scene's supply lane. New recordings save the prior scene/physics
and freeze grid geometry with the scene; older scenes recover that geometry from
their nine cell objects. The recovery test explicitly fixes the brief's original
benchmark, so live bench changes cannot alter its data partway through a run.

## Read-only bench findings and lead launch

The requested GET-only plan completed against `http://127.0.0.1:8793/state`, with
verdicts saved in `runs/active_twin/agent-i-armd-plan/plan.json`.
The daemon reports joint map `vcal_oct25`. Inspection of the specified legacy
`armd.py` and its `Kin.point` confirms xyz already uses the footprint/spec frame:
it subtracts `BASE_ORIGIN_MJCF` internally. Adding another offset would be wrong.
Its `min_z` object reports current mesh heights, not safety clearance settings.
The legacy clearance defaults are link 30 mm, gripper 20 mm, finger 15 mm; the
8-mm pushes are correctly refused. Current hold/sweep plans also exceeded a
calibrated owner limit and were refused.

There is an unavoidable telemetry incompatibility with that running legacy
daemon: `handle()` holds its lock for the entire blocking move, `/state` acquires
the same lock, and `/state` has no commanded target. Polling it in another thread
cannot obtain a >=10 Hz motion trace. The adapter detects this and refuses before
POST. It supports the repository daemon's existing `joints` + `stream` protocol,
which reports measured model q, owner degrees, and actual q_target. This work does
not modify, restart, or reconnect the running daemon, and never opens COM3.
Streaming plans respect its two-degree joint margin and conservatively use its
eight-degree/second near-table speed ceiling throughout the experiment.

A final GET-only recheck found a concurrent calibration change: the repository
now selects `vcal_oct25b`, while the running legacy daemon still reports
`vcal_oct25`. The adapter correctly refused with `Daemon and client joint maps
differ` before planning or POST. The lead must align the actual daemon and client
calibration as well as the telemetry protocol before the live command can run.

After the lead has safely arranged the compatible streaming daemon, a surveyed
scene at `runs/bench/active_scene.json`, and matching saved perception setup at
`runs/bench/active_perception.json`, the exact campaign command is:

```powershell
.venv/Scripts/python.exe scripts/argon/active_twin.py --mover armd --budget 6 --argon live --request-budget 30 --scene runs/bench/active_scene.json --perception-setup runs/bench/active_perception.json --run bench-active-01
```

Those bench input files are prerequisites, not artifacts produced by the
synthetic campaign. The lead should first use the same command with `--plan-only`
and a different run name to check the actual start pose and surveyed scene.
Before every live experiment, independent vision must confirm the supply is
within 4 mm of its surveyed reset position. The mover waits up to 60 seconds for
the lead or a reset fixture to restore it, using only reads while waiting; then it
re-reads the arm and validates the whole plan again. `--reset-timeout` sets this
bounded wait. Failure to observe a reset stops the campaign. This is supervised
reset support, not an autonomous physical reset mechanism.

## Limits of the evidence

No real-arm campaign, live MAIN call, real transfer validation, or automatic
activation was performed. Sim segmentation is ideal; final object positions are
recovered from those rendered masks, not read from hidden truth. Circular caps
have unobservable yaw, reported as null. Table friction can be unidentifiable
under MuJoCo's maximum-friction combination and retains explicit uncertainty.
Repeated held-out selection is not an independent final transfer audit. The
live camera's capture times are labelled estimates from camserver `age_s`, with
receipt times and raw sensor/command logs retained alongside the resampled trace.

## Commits and validation

- `b9af7c9`: active experiment engine, streaming/legacy daemon adapter, CLI,
  hidden-truth recovery and safety tests.
- Required full suite: **882 passed in 266.81 s** before the implementation commit.
- Final targeted safety, HTTP, timestamp, reset, and moved-board checks:
  **22 passed, 1 deselected in 14.12 s**. The full campaign is exercised separately
  and in the full suite.
- A later shared-checkout run had **926 passed, 5 failed** while other agents
  changed `tasks/place_on_cell/task.json`, `r2p/tasks/scene.py`, and their tests.
  Failures were the nine-target scene assertion, three `place_on_cell` perception
  accuracy cases (seeds 2/4/8), and the `place_on_cell` expert's 0.8 vs 0.9 gate.
  The scene test was subsequently fixed by its owner (`tests/core`: 619 passed).
  None of those files was edited by agent I.
- To obtain reproducible final acceptance, the standard full command is also run
  in `runs/active_twin/agent-i-validation`, a detached checkout of `1ae7090` plus
  the final three agent I source/test files. File hashes match the shared
  workspace. It uses the same `.venv` through a directory junction and does not
  include other agents' unfinished, uncommitted changes.
- Final isolated acceptance result: **907 passed in 216.58 s**. This includes all
  23 active-twin cases and the physical hidden-truth campaign. The live shared
  checkout's earlier failures are reported above, not represented as a clean run.
- The follow-up commit containing this report adds daemon-limit planning,
  supervised reset waiting, scene-bound grid geometry, the fixed benchmark,
  and the inactive final candidate. No baseline physics configuration or
  hardware setting is activated by either agent I commit.
