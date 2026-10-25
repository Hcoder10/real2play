# Agent L: game data through the twin — 25 October 2026

The data chain runs end to end and publishes real measurements to the dashboard.
Adding game traces reduced held-out action-chunk prediction error by **69.7%**
in the 4,000-update study and **69.2%** in a 12,000-update repeat.
It did **not** improve closed-loop task success: both small CPU policies scored
0/12 in clean simulation and 0/12 with simulated real noise. The gate is **HOLD**.
These are simulated players and simulation evaluations, not human play or real-arm results.

## Run it

```powershell
.venv/Scripts/python.exe -m r2p loop --from-game roblox/out/traces --task place_on_cell
```

For a fresh offline collection as part of the same command:

```powershell
.venv/Scripts/python.exe -m r2p loop --from-game roblox/out/new_headless/traces --collect-headless 32 --task place_on_cell --output runs/new_game_loop --steps 4000 --eval-n 12
```

The exact 4,000-update acceptance command was:

```powershell
.venv/Scripts/python.exe -m r2p loop --from-game roblox/out/l_current_received/training --task out/l_game_data_loop/task_1305.json --validation-game roblox/out/l_current_received/validation --output runs/l_game_data_loop_final --steps 4000 --eval-n 12 --seed 7026
```

It exited zero in **270.4 seconds**. Use a fresh `--output` when repeating it.
The task snapshot captures the current measured board at collection time, after
the lead's board/supply-lane update. The physics is the current
`config/physics/so101_bench.json`, fingerprint
`8f42f7a072096cbeeb3fb0376286a5cf1c7cde3aa37d74aed2268c5e33cb62d4`.
The task fingerprint is
`cf8647e4c536c4ccc456dca476f37a7ed1ddb82f6a2c5c2fd9a5c72a6fa3d770`.
No recorded code dependency changed during either current-board run.

## Where the play came from

`game_player.py` ports the actual Roblox `SkillOperator.lua` pick/place/release/
home state machine and `Kinematics.lua` IK equations. The port runs against
MuJoCo contact physics. It does not delegate actions to the sim Expert, attach
objects, teleport them, or manufacture successful placements. Metadata declares
`source=bot`, `headless=true`, `synthetic=true`, the controller hashes, task and
physics, and explicitly says this is a Python player with MuJoCo contacts.

For the final board snapshot:

| Split | Seeds | Episodes | Successful player rounds | Ticks |
| --- | --- | ---: | ---: | ---: |
| Training gameplay | 8026–8057 | 32 | 29 | 12,784 |
| Held-out gameplay | 9500–9507 | 8 | 7 | 3,265 |

All **40 episodes / 16,049 ticks** were compared against the actual pure-Luau
controller on the same recorded observations. There were zero command/phase
mismatches; maximum absolute target difference was **1.114e-9 radians**.
This validates the player port, not Roblox-versus-MuJoCo contact parity.
See [controller comparison](../../out/l_game_data_loop/current_player_parity.json).

All 40 traces then passed through the production authenticated HTTP receiver
(`roblox.tools.trace_server.make_server`, local ephemeral port and private key),
using its actual chunk protocol with at most 600 ticks per part. Reading the
received JSONL with `r2p.traces.read_trace` returned identical metadata and ticks.
No key was printed or retained. See
[authenticated round trips](../../out/l_game_data_loop/current_authenticated_upload.json).
The public `:8770` receiver uses this same implementation; the existing demo
receiver and arm daemon were not changed.

## Ingest, replay, labels, and training

The final training input contains those 32 received traces plus three unchanged
archived Roblox-engine bot captures from `roblox/out/placement_validation/traces`.

| Input | Ingested | Rejected | Relabeled | Replay successes |
| --- | ---: | ---: | ---: | ---: |
| Training input | 35 | 3 | 32 | 29 |
| Held-out prediction input | 8 | 0 | 8 | 7 |

The archived captures have a different task revision and are excluded. Separately
replaying each under its own recorded task but **current physics** also rejected
all three: joint RMS 2.646/4.666/4.921 degrees, jaw RMS 5.011/8.928/8.443 mm, and
maximum object displacement error 18.98/148.93/249.76 mm. Two no longer satisfy
the task in replay despite recorded success. See
[archived engine replay](../../out/l_game_data_loop/archive_recorded_task_replay.json).
This is evidence of mismatch with this reference, not an accusation of cheating.

The loop validates canonical traces, rejects fixtures/duplicates/incompatible
tasks, checks finite bounded absolute targets, and replays every eligible action
from reset without resetting objects to observed poses. Acceptance limits are
1 degree joint RMS, 4 degrees maximum joint error, 2 mm jaw RMS, 5 mm object
position error, and 5 degrees symmetry-aware yaw/object-axis tilt error.
Tests deliberately corrupt joint motion, object positions, dimensions, and
orientations and verify rejection.

Accepted observations come from the replayed twin. Labels retain original
absolute servo targets. Inputs use previous commanded q, reset to measured q
on any joint divergence over four degrees. `r2p.labeling.segment` preserves
the game's command events as `pick(piece)`, `place_at(piece,cell)`, `release`,
and `home` segments. These are the Argon command vocabulary with deterministic
phase labeling: **zero model requests, no claim of live Argon refinement**.
Training constructs 15×6 chunks at 30 Hz and pads at command boundaries.

Four fresh expert episodes, seeds 7026–7029, supply the baseline; three succeed
and one fails. The before network trains on 2,479 expert windows. The after
network trains on those plus 12,784 game windows, **15,263 total**. Failed
episodes remain in the mixture with the existing dataset's reduced weight.
Both networks use the same seed, 128-unit width, two blocks, 4,000 updates,
batch size 128, augmentation and two CPU threads. Their exports contain the
task, start pose, observation layout/constants, absolute-action declaration,
weights, standalone NumPy runtime, and recorded evaluations.

## Measured learning result

The paired plan contains **12 held-out scenes per condition**, independent of
training, with identical scenes/noise seeds for both policies and full 40-second
episodes. Training/evaluation scene hashes are checked for overlap. The local
high-level skeleton scheduler issues the command vocabulary without model calls;
it permits a lowered placement to advance to the separate release command.
The unchanged shared success checker still requires open jaws and its stable
half-second window.

| Policy | Expert episodes | Game episodes | Clean successes | Real-noise successes |
| --- | ---: | ---: | ---: | ---: |
| Before | 4 | 0 | 0/12 | 0/12 |
| After | 4 | 32 | 0/12 | 0/12 |

The **eight separate held-out game episodes / 3,265 replay ticks** measure
teacher-forced action prediction, without training on those scenes:

| Prediction error | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| Absolute 15×6 chunk MAE | 0.161531 rad | 0.048874 rad | 69.7% |
| First-action MAE | 0.153691 rad | 0.042889 rad | 72.1% |

The independent episode count is eight; ticks are correlated. This diagnostic
does not imply task success and does not affect selection. **HOLD retains the
baseline. No policy is promoted to hardware.** The requested positive
closed-loop success improvement has not been established.

Evidence: [manifest](../../runs/l_game_data_loop_final/game_loop.json),
[replay decisions](../../runs/l_game_data_loop_final/replay.json),
[evaluation scenes](../../runs/l_game_data_loop_final/eval_plan.json),
[before outcomes](../../runs/l_game_data_loop_final/before_eval.json),
[after outcomes](../../runs/l_game_data_loop_final/after_eval.json),
[prediction errors](../../runs/l_game_data_loop_final/imitation_eval.json),
[gate](../../runs/l_game_data_loop_final/gate.json).
The manifest hashes **132 artifacts**, including raw input snapshots, labels,
expert/mixed data, both exports, physics, scene plan and reports. Earlier
eight- and 32-game-trace experiments remain under `runs/l_game_data_loop` and
`runs/l_game_data_loop_more`; they also held at 0/12 and were not pooled with
the new-board experiment.

## Longer training repeat

To test whether additional fitting could produce closed-loop success, a second
current-board experiment used 12,000 updates per policy, seed 10026, four new
expert episodes, and fresh evaluation scenes:

```powershell
.venv/Scripts/python.exe -m r2p loop --from-game roblox/out/l_current_received/training --task out/l_game_data_loop/task_1305.json --validation-game roblox/out/l_current_received/validation --output runs/l_game_data_loop_extended --steps 12000 --eval-n 12 --seed 10026
```

It exited zero in **313.0 seconds**. The same 35 input files produced the same
3 rejections and 32 relabeled game episodes. Three of the four new expert
episodes succeed. The baseline has 2,847 windows; the mixed dataset has 15,631.
Both policies again scored **0/12 clean
and 0/12 real-noise**, so the published gate remains **HOLD**. On the same eight
held-out gameplay episodes, chunk MAE was **0.1533403 -> 0.0471666 radians**
(69.2% reduction), and first-action MAE was **0.1322732 -> 0.0386463 radians**
(70.8% reduction). These remain prediction-error measurements, not successful
control. This repeat also uses a fresh training seed and expert set; recorded
code fingerprints match across the two studies. Compare before versus after
within each study rather than attributing differences between studies solely
to update count.

The [repeat manifest](../../runs/l_game_data_loop_extended/game_loop.json)
records the source fingerprints, paired scenes, exports, evaluations and
diagnostics. Its **132 artifact hashes also all verify**, and no recorded code
dependency changed during execution. See the
[integrity check](../../out/l_game_data_loop/extended_artifact_integrity.json).
The completed repeat is now the dashboard's published run.

A [recorded diagnostic rollout](../../out/l_game_data_loop/extended_rollout_diagnostics.json)
on its first clean evaluation scene localizes a remaining control failure.
The baseline's jaw midpoint stayed at least 29.5 mm from the piece. The
mixed-data policy approached within 5.0 mm and the piece reached 28.1 mm above
the table, then fell back to 7.8 mm. Both policies were still in `pick(piece)`
at tick 1,199. This single-scene diagnosis supports further work on stable
grasp/lift control; it is not evidence of a successful placement.

## Dashboard and validation

Completion atomically publishes `runs/game_loop_latest.json`. GAME separates
human, native bot and headless bot counts and displays ingest/rejection/relabel
counts. POLICY shows the measured two-point curve and the selected policy's
paired evaluation, with n and simulation provenance. The chart title no longer
asserts that more play necessarily improved success. Explicit headline pins
remain authoritative.

Final [desktop capture](../../out/l_game_data_loop/dashboard_extended.png),
[mobile capture](../../out/l_game_data_loop/dashboard_extended_mobile.png), and
[adapter state](../../out/l_game_data_loop/dashboard_extended_state.json) retain the
published counts and gate. Captures use real wall-clock browser time, not
virtual-time screenshots; camera fetching was disabled for this visual check.
Both have zero JavaScript errors and no horizontal overflow. With the concurrent
Argon status strip present, the desktop document is 1,203 pixels tall at
1920×1080; the mobile document is 2,486 pixels tall at 390×844. Both pages
scroll vertically. The separate owner's layout was not changed.

The game-loop regression file passes **20 tests**. The preceding combined
game-loop/dashboard/gate/Roblox check passed **56 tests** before the extra
human-provenance regression was added. The full required suite was run before
each commit. The final run completed with **988 passed, 6 failed in 393.25 s**,
including all 20 passing game-loop regressions. Its failures are outside L's
modified paths; the repository-wide acceptance suite is **not green**.
See the [full-suite log](../../out/l_game_data_loop/full_suite_handoff.log).

The six final failures are:

- `tests/argon/test_active_twin.py::test_sim_campaign_recovers_hidden_truth_without_fit_leak`:
  fitted cap friction 0.40205 versus expected 0.45 +/- 0.04.
- `tests/argon/test_active_twin.py::test_streaming_mover_records_sensor_and_capture_times`:
  at least one recorded command lacks the expected `stream` flag.
- `tests/perception/test_tracking.py::test_synthbench_pose_accuracy`,
  `place_on_cell` seeds 2, 4 and 8: fixed cell pose errors of 18.66, 30.95 and
  47.65 mm exceed 3 mm; nominal cell poses bypass the randomized board transform.
- `tests/sim/test_traces.py::test_two_worker_collection`: MuJoCo could not
  allocate memory in a worker on the shared laptop.

An isolated rerun of these six tests completed with **3 passed, 3 failed in
147.66 s**. Both active-twin checks and the two-worker collection passed; the
three randomized-board perception failures remained. See the
[isolated rerun log](../../out/l_game_data_loop/isolated_shared_failures.log).

Those owners' files were not changed or their tests weakened. Before the
implementation commit, the same suite had 965 passes and five failures,
including a transient robot-runner import and a failed expert placement.
An earlier suite also crossed a concurrent task-schema update; its stale
validator errors were resolved by rerunning with the updated validator.

## Changes, commits, and remaining work

Implementation commit: **`e8a656e`** (`feat(pipeline): replay game traces through
twin and gate CPU learning`). It owns `r2p/pipeline/game_*.py`, the optional
tracker hook in `r2p/pipeline/evaluation.py`, `r2p/cli.py`, the pipeline README,
dashboard adapters/labels, and `tests/app/test_game_loop.py`. The report and
human-provenance regression are in the follow-up commit named in the handoff.

No hardware, live Gemini request or GPU was used. Human-play collection requires
a published current Roblox place, server-only HTTPS receiver URL and matching
key, a successful current engine-parity gate, and actual players completing
rounds. Start the existing receiver with `scripts/demo_up.py`, then point the
loop at its completed JSONL directory. Old task revisions must remain separate.
The physics preset remains provisional rather than a demonstrated exact real
twin. Human participation, successful learned closed-loop control, real-arm
transfer, and a green shared-repository acceptance suite remain unproven here.
