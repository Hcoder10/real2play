# P3: physical expert under the calibrated bench profile

The current `place_on_cell` task improved from **1/64 (1.6%)** to
**63/64 (98.4%)** on seeds 0–63, without changing the task, physics, collision
geometry, calibrated limits, locked roll, or success checker. All grasps remain
MuJoCo contacts: no welds, attachments, object teleports, or applied forces.

The baseline had 44 `grasp_miss:piece` and 19 `dropped_during_carry:piece`
failures. The repaired expert has one `goal_not_met`, seed 34, with 13.01 mm
placement error at the nearest randomized cell. Median placement error across
all 64 scenes is 1.23 mm. Final verification and DART results are recorded in
[bench_expert_acceptance.json](bench_expert_acceptance.json).

With the trace collector's normal `--dart .01`, the final expert achieves
**59/64 (92.2%)**, with five `goal_not_met` outcomes and no grasp misses or carry
drops. The perturbation is applied before the profile's execution rate guard.

| Same 64 seeds | Successes | Median placement error |
| --- | ---: | ---: |
| Original expert, DART 0 | 1/64 | 112.29 mm |
| Repaired expert, DART 0 | 63/64 | 1.23 mm |
| Repaired expert, DART 0.01 | 59/64 | 2.65 mm |

The task snapshot/hash is recorded in that artifact. Its SHA-256 is
`e4335723e937d9e9c142e59af53fbac351f5ce110a5b98fe03e3027170c6db21`.
Baseline expert Git blob: `4f9fbb367b9b7eb6ade8b6d28527eb1a8fa5a112`.
The evaluator freezes the task and profile before dispatching a
`ProcessPoolExecutor`; two workers avoid the laptop's allocation failures.

## Root cause and repair

1. **A close endpoint was used as a holding position.** The captured profile
   correctly maps owner 6 to 0.00214 radians and owner 20 to 0.4038 radians.
   The real measurement says a cap stalls around owner 11, about 0.14559 radians;
   a stall is not a request to drive the hinge through the object. With
   `kp=20 Nm/rad`, the old endpoint saturates the twin's 0.8 Nm actuator during
   contact. Seed 0 showed roughly 5 mm of pad penetration, then a sideways cap
   and a fully closed empty gripper. The expert now bounds closing position
   error to `0.1 Nm / kp` (0.005 rad with the current physics), and rate-limits
   the gripper with the same profile step as the arm. The physical hinge still
   stalls against the cap; labels remain absolute servo targets.

2. **The midpoint controller moved the fixed jaw.** Closing/opening changes the
   midpoint because only one jaw moves. Re-solving arm IK to hold that midpoint
   during contact swept the fixed finger through the cap, and could knock it
   away on release. Closing and release now hold the five arm targets, allow
   the hinge to move, then rebase FK on the measured opening. Placement retains
   its observed cap-to-jaw offset. This respects the measured asymmetric jaw
   shift; it does not add a world-frame offset.

3. **The simulated grasp was too high on a thin cap.** The model's FK points
   are the bottom fingertip edges; its pads extend 16 mm upward. A 9 mm tip
   target contacts 14–20 mm caps above their centre and lets them tilt during
   the long carry. A 2 mm tip target and gentle contact pressure hold the cap
   across its side. The observed real jaw heights of 14.6–15.5 mm are **not**
   reproduced by this controller change. Reconciling that measurement with
   the provisional contact geometry remains a real-twin calibration task.

4. **Some carry targets were unreachable.** The fixed roll is valid for round
   upright caps; choosing a square-grasp yaw would violate the bench convention.
   With that roll and the wrist-flex stop, some near-row cells cannot be reached
   at the fixed 50 mm travel height. The expert searches downward for the highest
   reachable carry endpoint using the measured grasp opening and the same IK
   bounds. Remaining collisions are still resolved by physics. The one failed
   seed is retained in the acceptance denominator.

5. **The release threshold must describe a gentle hold.** A wide cap can stall
   above the captured half-stroke threshold even while firmly held. The expert
   derives a copy of the profile with release at 90% of its open stroke
   (0.363634 rad), and records this and the contact-control settings in trace
   metadata. The existing dataset/export/runner paths already consume that
   threshold. The captured JSON stays unchanged. Tests replay the existing
   command tracker through pick, place and home. Fresh traces must be used;
   their derived deployment convention deliberately differs from older traces.

DART also now preserves the locked roll and rate limit in executed commands,
as well as labels. The nominal command history advances independently of the
disturbed execution: anchoring every nominal step to the disturbed command
otherwise stalls the approach when the noise exceeds the allowed per-tick step.
`q_target` remains the unperturbed label; `q_exec` records guarded disturbance.
No frame shifts, new joint limits, or altered physics were introduced.

## Reproduce

Acceptance (exit code 1 if fewer than 70% succeed):

```powershell
.venv/Scripts/python.exe -m r2p.sim.bench --task place_on_cell --bench-profile scripts/robot/K_BENCH_PROFILE.json --n 64 --seed 0 --workers 2 --dart 0 --out r2p/sim/bench_expert_acceptance.json
```

Exact trace command with the normal DART setting:

```powershell
.venv/Scripts/python.exe -m r2p.sim.traces --task place_on_cell --bench-profile scripts/robot/K_BENCH_PROFILE.json --n 64 --seed 0 --workers 2 --dart .01 --out-dir out/p3/bench_traces
```

The trace collector retains its existing four-worker maximum; the evaluator
allows up to eight, but two were used here. Both successful and failed episodes
are preserved. Generate into a fresh directory instead of mixing old profile
traces with this contact-control convention.

## Validation and limits

The final serialized test run passed **40 tests, with 1 deselected**, in 83.94 s:

```powershell
.venv/Scripts/python.exe -m pytest -q tests/sim tests/robot/test_real_policy.py tests/policy/test_bench_export.py -k 'not render_rgb'
```

The OpenGL rendering test was excluded for this CPU-only assignment. A
four-worker probe and an earlier pytest run overlapping a two-worker evaluation
hit MuJoCo allocation failures; the final tests above ran without an evaluation
pool and passed. This is why the commands here use two workers serially.

Both final 64-seed evaluations have zero target/execution limit violations and
zero commanded roll error. Maximum target and executed steps are
0.002327106 rad, the declared 4 degrees/second at 30 Hz. Regression tests cover
physical lift/hinge stall, near-row placement, command alignment, target steps,
and locked-roll DART execution.

A two-worker trace smoke (`--n 2 --seed 0 --dart .01`, output
`out/p3/trace_smoke`) produced two successful episodes. Both pass
`validate_trace` with zero schema errors, retain distinct unperturbed labels
and guarded execution, and carry the derived release threshold/control settings.

The clean episodes take 67.43–124.03 simulated seconds (median 89.60), including
return home. `Expert.run` does not enforce the task's 40-second policy budget.
Policy training/rehearsal therefore needs a consistent longer budget; this work
does not claim a 40-second placement or change the shared task specification.

Not done: training or exporting a new policy, real hardware or camera validation,
fitting the real grasp height/contact geometry, the held-out twin accuracy gate,
or fixing the single failed nearest-cell scene. No GPU or Gemini requests were
used. No changes were made to `scripts/lead`, `r2p/lab`, or `r2p/policy`.
