# Agent K: learned-policy real-arm handoff

The calibrated runner, physical-twin rehearsal, and one-placement live command
are implemented. **No available learned export is qualified for a real
placement.** The existing export is refused before motion; the new tiny CPU
fixture exercises the valid handover and control path but places no caps.

During this work the lead replaced the old waypoint daemon with this repo's
streaming daemon. The preferred live path is now `--transport armd`: it keeps
that daemon owning the bus after a settled, bounded handover. The captured
streaming state is refused: its wrist-flex target exceeds the physical limit.
The direct release-at-rest path remains available.

K opened no serial port, sent no POST to armd, made no Gemini request, and used
no GPU. The only arm access was read-only GET /state. Other agents' files,
tasks, bench calibration and physics configuration were not edited.

## Measured incompatibilities and fixes

The captured profile is [K_BENCH_PROFILE.json](K_BENCH_PROFILE.json). Its owner
limits also match the calibration file on disk. The active map is
`vcal_oct25b`, unverified, with wrist-roll offset -164.4 degrees and gripper
open_rad 2.699. Bounds are converted with the map, sorted for negative signs,
intersected with MJCF limits, and given the existing 2-degree guard margin.
Every final write is checked again in owner coordinates; invalid writes are
rejected. No runtime pi-flip or frame shift was introduced.

| Quantity | Finding |
| --- | --- |
| Wrist flex physical upper limit | 101.8 owner degrees = 82.87 model degrees |
| Wrist roll physical upper limit | 179 owner degrees = 14.6 model degrees |
| Old export start wrist flex | 113.93 owner degrees; 113.83 after compensation |
| New round-cap roll convention | 81.9 owner degrees = -82.5 model degrees |
| New gripper opening | owner 20 = 0.4038 model radians, about 50.7 mm fingertip gap |
| New squeeze | owner 6 = 0.00214 model radians |

`Session` now accepts owner limits, refuses an incompatible export start
before sending anything, counts start refusals/clipping separately, and reports
raw chunk, projected-target and per-joint limit events. The four-degree
commanded-q reset rule, swept clearance, existing speed caps and compensation
ordering remain active. Servo-target exports receive no second compensation
during policy execution. Start_q is the declared measured/nominal pose, so its
approach still gets feedforward.

The physical rest owner pose `[0.1,-89.9,99,-101.3,81.9,2.7]` maps outside the
nominal MJCF ranges in several joints. The rehearsal therefore uses the actual
calibrated physical joint ranges for the twin, while the policy guard retains
the stricter intersection. It does not silently initialize the twin at a
different, clipped rest pose. The valid fixture's guarded approach takes
590 ticks (19.67 simulated seconds), with zero refusals.

## Required training convention

The opt-in `--bench-profile` path in `r2p.sim.traces` changes the expert and IK
for upright cylindrical caps. Circular caps have continuous yaw symmetry:
keep the observed real roll instead of `_grasp_yaw` choosing a square-style
quarter-turn branch that can exceed the owner roll range. IK locks that roll,
honors mapped wrist-flex bounds, uses 50 mm travel height, 9/12 mm grasp/place
heights, the real owner-20/owner-6 gripper mapping, and a conservative 4-degree/s
joint-target rate. A new bounded start has compensation headroom. The policy
command tracker uses a 30 mm observed lift for this lower travel convention.

Training copies the profile, start, limits and gripper-open threshold into the
export. New exported outputs are bounded and keep the calibrated roll. Mixed
old/new profile traces are rejected. The default expert remains unchanged;
the profile refuses rectangular objects. Regenerate traces and retrain rather
than changing old weights or rotating their outputs.

The one CPU smoke trace lifted its cap but lost it during carry. The profile
expert now reports `dropped_during_carry` and stops rather than chasing the
dropped object with its placement offset. **This profile is a tested constraint
convention, not a successful expert recipe.** The lead must inspect fresh
physical-contact expert outcomes before starting the large trace run. The
conservative timing can exceed the task's old 40-second budget; evaluate a
chosen budget consistently in rehearsal and live operation.

```powershell
.venv/Scripts/python.exe -m r2p.sim.traces --task place_on_cell --n 1 --out-dir out/robot/k_profile_probe --workers 1 --seed 7300 --dart 0 --bench-profile scripts/robot/K_BENCH_PROFILE.json
```

## Rehearsal results

The same `Session` and owner-space checks run in rehearsal and live mode. The
rehearsal robot is `TaskTwin`: configured gravity/servo compliance from
`config/physics/so101_bench.json`, one latency tick, physical cap contacts,
calibrated owner limits and the captured rest. It freezes the CURRENT measured
task, including its board pose and 34 x 17 mm cap. The supply positions are
`[0.1274,-0.1272]`, `[0.1748,-0.1245]`, `[0.2205,-0.1248]` metres. Each cell is
tested with each selected cap, with the other two caps retained as physical
distractors. Observations are twin truth; this does not test camera accuracy.
The matrix uses an empty board; occupied-board games need their own rehearsal.
The lead continued resurveying the board during this task. Each invocation
freezes its input task once; [K_MEASURED_TASK.json](K_MEASURED_TASK.json) preserves
the survey used for the final runs. `--measured-task` replays that exact survey;
omitting it always reads the current bench task. The three requested supply
positions stay fixed even when the task's training lane changes.

Detailed, versioned per-case numbers and input snapshots:
[K_REHEARSAL_RESULTS.json](K_REHEARSAL_RESULTS.json). Full local outputs:
`out/robot/k_final_existing.json`, `out/robot/k_final_smoke.json`, and
`out/robot/k_final_stream.json`. The final board centre is `[0.178,0.0581]`
metres; the JSON includes the complete measured task and its hash. Earlier
local rehearsal files used an earlier board survey and are superseded.

| Cell | Successes / 3 | Final XY error in mm, caps 0 / 1 / 2 |
| --- | ---: | --- |
| r0c0 | 0 | 155.43 / 157.38 / 174.66 |
| r0c1 | 0 | 191.52 / 191.22 / 204.48 |
| r0c2 | 0 | 227.76 / 225.89 / 236.16 |
| r1c0 | 0 | 156.00 / 146.72 / 154.70 |
| r1c1 | 0 | 192.08 / 182.63 / 187.77 |
| r1c2 | 0 | 228.25 / 218.68 / 221.87 |
| r2c0 | 0 | 165.07 / 144.94 / 141.68 |
| r2c1 | 0 | 199.52 / 181.20 / 177.20 |
| r2c2 | 0 | 234.50 / 217.47 / 213.00 |

Both artifacts leave the cap at supply, hence the same final distances. These
are residual distances after failed attempts, not successful placement errors.

- `place_on_cell-dry_r2_a4`: **0/27**, all `start_refused`, zero commands,
  clipping, resets or executed limit hits. Raw first chunks contain 9-12
  out-of-owner-range targets, audited without execution and reported per case.
- `k_place_on_cell_cpu_smoke`: **0/27**, all `timeout` after 1,800 policy ticks.
  Each case has 68 clipped policy ticks (3.78%), zero refusals, zero commanded-q
  resets, zero projected-target limit hits and zero sent owner-limit hits.
  Each also has 590 rate-clipped preparation ticks, reported separately.
  There are 2,390 commands per case, 64,530 total. Total simulation is 2,151
  seconds; session wall time totals 303.81 seconds. The fixture has one trace
  from an earlier board survey, five CPU training steps, width 32, one block
  and one seed. Its recorded
  clean/real/realx0.5 smoke evaluations are all zero successes. It is supplied
  only to reproduce plumbing; live mode rejects its `smoke_only` flag.

```powershell
.venv/Scripts/python.exe scripts/robot/rehearse_policy.py --export place_on_cell-dry_r2_a4 --measured-task scripts/robot/K_MEASURED_TASK.json --out out/robot/k_final_existing.json
.venv/Scripts/python.exe scripts/robot/rehearse_policy.py --export k_place_on_cell_cpu_smoke --measured-task scripts/robot/K_MEASURED_TASK.json --timeout 60 --out out/robot/k_final_smoke.json
.venv/Scripts/python.exe scripts/robot/rehearse_policy.py --export k_place_on_cell_cpu_smoke --transport armd --bench-profile scripts/robot/K_STREAM_PROFILE.json --measured-task scripts/robot/K_MEASURED_TASK.json --out out/robot/k_final_stream.json
```

These commands deliberately return exit code 1 when placements fail. A single
case accepts `--cell r1c1 --cap 0`; `--record <directory>` records trace labels,
commanded observations and executed targets. Safety projection is never used
as evidence that the underlying learned motion is accurate.

## Streaming daemon and legacy compatibility

The current streaming snapshot is [K_STREAM_PROFILE.json](K_STREAM_PROFILE.json).
The client checks that `q` equals active-map conversion of `owner_deg`, that
the daemon implements the repo state protocol, that torque is on, and that its
pose still matches the rehearsed starting pose within four degrees. Its held
target must be within owner limits and at most four degrees from measured q.
Calibrated owner limits are read from the local servo calibration and rechecked before
execution. All checks use GET; no POST occurred during K's work.

The captured state has a wrist-flex target near **111.93 owner degrees** versus
the **101.8-degree** limit, with approximately **10 degrees** of target lag.
All 27 streaming cases therefore return `handover_refused`, with zero writes,
guard clips, commanded-q resets or executed limit hits. This is a recorded
preflight failure, not a simulated placement. The owner must first hold a
bounded, settled target, then capture and rehearse a fresh profile. K did not
adjust the daemon target or infer that the moving arm had settled.

With `--transport armd`, the existing `DaemonRobot` sends each guarded absolute
target as `{"joints":[model radians],"stream":true}`. The server guards it
again and returns the accepted target, which becomes the next commanded-q
observation. It does not drop the intermediate chunk targets. The server's
rate cap may further slow a late request; HTTP/inference scheduling fidelity
on the real arm has not been measured. The reference twin includes its one
configured servo latency tick, not a claimed measurement of HTTP latency.
Streaming rehearsal also uses the actual `ArmController` and `DaemonRobot`
through an in-process transport, exercising the server's second guard and
accepted-target feedback at nominal 30 Hz. Physics advances once per runner
tick. `daemon_clipped_commands` separately counts server-side target changes.

The owner must reserve command access to armd for the placement and hold still
while capturing the profile; the rehearsal assumes zero initial velocity. A
moved arm requires a fresh read-only profile and rehearsal. The streaming runner never
opens serial, never primes torque, and leaves the daemon holding on exit.

The following explains why the earlier external daemon is deliberately refused:

The external daemon at `argon-hack/deploy/scripts/armd.py` accepts
`{"cmd":"joints","owner":[...]}` and clips owner bounds before moving. Its
move is synchronous, holds its state lock, interpolates the move, sleeps at
least one 30 Hz tick and then settles for 0.4 seconds. Even a negligible move
therefore takes at least 0.433 seconds, before HTTP overhead. Sending one
endpoint per chunk would discard 14/15 policy targets and alter the measured
latency/control trajectory. Concurrent requests would queue behind the lock.

The earlier captured daemon also reported **vcal_oct25**, while the current map is
**vcal_oct25b**. Its old-map clearance is not a validation of the new-map motion.
This repo's `DaemonRobot` additionally speaks a different protocol; it must not
be pointed at that daemon as a substitute. K did not change or restart either
daemon. Direct `RealRobot` is used only after the owner's explicit release at
supported rest; the repo streaming daemon needs no such handover.

## Exact commands and owner handover

Replace `exports/<trained-profile-export>` with the lead's newly trained,
evaluated export. **Do not use either smoke export for physical placement.**

Preferred path, leaving the current streaming daemon running:

```powershell
$policyExport = 'exports/<trained-profile-export>'
.venv/Scripts/python.exe scripts/robot/rehearse_policy.py --capture-profile scripts/robot/K_STREAM_PROFILE.json
.venv/Scripts/python.exe scripts/robot/rehearse_policy.py --export $policyExport --transport armd --bench-profile scripts/robot/K_STREAM_PROFILE.json --timeout 180 --out out/robot/k_candidate_stream.json
.venv/Scripts/python.exe scripts/robot/policy_setup.py --cell r1c1 --cap 0 --out runs/robot/k_cap0_setup.json
.venv/Scripts/python.exe scripts/robot/run_real_policy.py --export $policyExport --transport armd --bench-profile scripts/robot/K_STREAM_PROFILE.json --cell r1c1 --cap 0 --real --allow-unverified --perception-setup runs/robot/k_cap0_setup.json --timeout 180 --record runs/robot/k_policy
```

Omit `--cell r1c1` on the last command to have Argon choose the cell. The
following checklist is the direct serial fallback:

1. While armd owns the arm, refresh its read-only owner-limit/rest snapshot if
   calibration has changed, then run the candidate's full rehearsal and check:

```powershell
.venv/Scripts/python.exe scripts/robot/rehearse_policy.py --capture-profile scripts/robot/K_BENCH_PROFILE.json
$policyExport = 'exports/<trained-profile-export>'
.venv/Scripts/python.exe scripts/robot/rehearse_policy.py --export $policyExport --timeout 180 --out out/robot/k_candidate.json
.venv/Scripts/python.exe scripts/robot/run_real_policy.py --export $policyExport --check --cell r1c1
```

2. Confirm the measured board/caps still match the task and selected supply.
   Create the cap appearance setup while armd still owns the arm. This command
   only reads the camera and makes one SMALL-role box request; `--boxes file.json`
   instead accepts pixel XYXY boxes such as `{"piece":[x0,y0,x1,y1]}` for zero
   requests. Use `--supply-xy X Y` consistently when a cap has moved.

```powershell
.venv/Scripts/python.exe scripts/robot/policy_setup.py --cell r1c1 --cap 0 --out runs/robot/k_cap0_setup.json
```

3. The owner parks through the proven lead path, supports the arm at its captured
   rest, and releases the existing daemon. Do not kill/reconnect it while the arm
   is suspended. K does not issue that release. The live runner requires
   `--arm-released`, verifies the daemon port refuses connections, compares the
   stored limits with the current servo calibration, and performs a read-only
   rest check before torque priming. An active or uncertain port refuses access.

4. One explicit cell, using the learned export for all low-level motion:

```powershell
.venv/Scripts/python.exe scripts/robot/run_real_policy.py --export $policyExport --cell r1c1 --cap 0 --real --arm-released --allow-unverified --perception-setup runs/robot/k_cap0_setup.json --timeout 180 --record runs/robot/k_policy
```

5. Or let Gemini 4 Argon read the board and select the cell; omit `--cell`:

```powershell
.venv/Scripts/python.exe scripts/robot/run_real_policy.py --export $policyExport --cap 0 --real --arm-released --allow-unverified --perception-setup runs/robot/k_cap0_setup.json --timeout 180 --record runs/robot/k_policy
```

`--board confirmed_board.json` accepts nine confirmed marks or `{ "board": [...] }`
for an ongoing game. Argon uses the MAIN role and reports its served model; this
entry point refuses an automatic minimax substitution. A terminal/waiting board
causes no motion. Saved appearance replay makes no additional model calls.
The selected cap must currently be seen at the requested supply before priming.
Current surveyed cell geometry overrides a saved setup's previous target cell.
FK tracking is admitted only on measured closing-pad contact and ends on pad
separation; it remains a geometric grasp estimate, not a force measurement.

The unverified map uses half speed. `p` pauses; `s`/Space or `q` holds and requests
the guarded retreat; the trace records failures and operator overrides. At exit
torque remains on. The owner must park/support the arm before another process
primes or releases it. Do not assume an exit at timeout is a supported rest.

## Acceptance and remaining work

Acceptance command (no real hardware, Gemini or GPU):

```powershell
.venv/Scripts/python.exe -m pytest -q tests/core tests/argon tests/perception tests/robot tests/app tests/sim tests/policy
```

Core implementation commit: **e8cd98b**. The required full suite immediately
before that commit completed with **989 passed, 5 failed** in 384.30 seconds.
All K tests passed. The failures were outside K's changed files:

- `tests/argon/test_active_twin.py::test_sim_campaign_recovers_hidden_truth_without_fit_leak`:
  cap-friction estimate 0.40205 versus 0.45 +/- 0.04.
- `tests/argon/test_active_twin.py::test_streaming_mover_records_sensor_and_capture_times`:
  command log includes a command without `stream`.
- `tests/perception/test_tracking.py::test_synthbench_pose_accuracy` for
  `place_on_cell`, seeds 2/4/8: fixed cell estimates retain the nominal survey
  while the synthetic board is randomized (18.03 / 31.65 / 47.49 mm error).

These are recorded acceptance failures, not waived passes. K did not edit those
owners' implementation/tests. The final required full suite completed with
**999 passed, 4 failed** in **366.06 seconds**, before the handoff commit.
All K tests passed. The two active-twin failures above passed in that fresh run.
The remaining full-run failures were:

- The same three `place_on_cell` perception cases, now at
  **18.66 / 30.95 / 47.65 mm** error against the later board survey.
- `tests/argon/test_orchestrator.py::test_owner_wait_stops_controller_without_resubmission`:
  the owner-wait exception escaped its test assertion. An isolated fresh-process
  rerun immediately afterward **passed (1 test, 0.66 seconds)**. This does not
  turn the recorded full-suite failure into a pass.

Full local logs are `out/robot/k_acceptance_full.txt` and
`out/robot/k_acceptance_final.txt`. Focused validation of K's final streaming
adapter and smoke-export refusal: **22 passed** in 13.85 seconds.
The actual exported fixture also passes the no-motion CLI check:
`.venv/Scripts/python.exe scripts/robot/run_real_policy.py --export k_place_on_cell_cpu_smoke --check --cell r1c1`.
This checks compatibility; the fixture remains explicitly ineligible for live use.

New tests cover signed bound conversion, the real map/gripper,
zero writes on invalid start, direct-owner exclusion, folded-rest twin fidelity,
closing-contact tracking, setup target changes, locked-roll IK, trace-profile
mixing, and CPU export preservation of calibrated constraints.

The follow-up handoff commit contains the streaming-controller rehearsal,
smoke-only live refusal, frozen survey, CPU fixture, per-case evidence and this
report. It changes only K's paths; the final response records its commit ID.

Not done: a fully passing repository acceptance suite, a successful trained
candidate, a passed held-out physics gate,
physical/camera validation, a real learned placement, owner release, or a
hardware timing measurement of the new streaming path. These remain explicit
deployment dependencies; this work does not claim physical success.
