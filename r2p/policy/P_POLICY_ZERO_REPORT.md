# Agent P: policy-zero investigation

The implementation now has a twenty-trace CPU regression that checks prediction
accuracy and physical closed-loop task completion. Full-scale retraining and a
qualified learned policy remain outstanding; the incumbent stays **HOLD**.

## Root cause

The failure is a closed-loop information/execution problem, rather than a frame,
units, slot-order, or double-gravity-compensation error. On the supplied expert
observations, `wide_v1` predicts its first action to 0.00941 rad MAE. At contact,
the gripper's measured angle differs from its squeeze target by more than four
degrees. The mandated whole-observation reset then clears the temporal ensemble
every tick, repeatedly selecting the first, almost stationary action of the
chunk. The expert's timed close/lift and lower/release transitions are ambiguous
from instantaneous q/slots/command: no elapsed time or history was provided to
the network. Keeping future predictions lets it lift and carry but still stalls
on release. Two additional contract defects compound this: calibration offsets
were applied only to input q despite **absolute** outputs, and the evaluator
rate-limited from reset measured q instead of the last sent target.

## Changes

- `features.py`, `data.py`, `runtime.py`: append an explicit `command_age_s`
  feature to new exports. Training derives it from trace timestamps and command
  boundaries. Runtime maintains it at the control cadence; command changes and
  episode reset restart it, measured-q resets do not. An explicit `tick` supports
  sparse replanning; explicit `command_age_s` supports teacher forcing. Old
  exports retain their original feature layout and require retraining.
- `data.py`, `train.py`: calibration augmentation changes both observation q
  and all absolute action targets in the same coordinate system, while leaving
  gripper targets unchanged. The convention is recorded in the export.
- `eval.py`: keep the sent-target rate-limit origin separate from the resettable
  policy observation, reset policy state per episode, supply control ticks, and
  expose an optional diagnostic callback. The four-degree reset, absolute target
  convention, contact physics, command scheduler and final success checker remain
  active.
- `diagnose.py`: reusable teacher-forced error measurement plus closed-loop replay
  of a trace's exact scene. `TraceDataset` now also accepts one trace file.
- `tests/policy`: clock/feature parity, calibration-target consistency, blocked
  gripper rate limiting, and twenty-trace learning regression. The fixture freezes
  the wide-v1 task independently of concurrent real-bench surveys.

## Measurements

All measurements below are CPU simulation, with no hardware or Gemini requests.
The supplied seed-200000 trace is a failed expert attempt and is absent from
wide-v1's successful training provenance; its observations still provide a useful
teacher-forced diagnostic. It is not counted as an expert success.
Runtime observation features and dataset features agree within 8.3e-8 on every
recorded tick, ruling out a slot/FK/command-encoding discrepancy on that trace.

| Experiment | Result |
| --- | --- |
| Original wide-v1, clean seeds 10000–10004, 1200 ticks | 0/5 |
| Wide-v1 on supplied recorded observations, all 15 targets | 0.01494 rad MAE |
| Same, first action only | 0.00941 rad MAE |
| Original wide-v1, exact seed-200000 scene | timeout, pick never completes |
| Rate-limit-origin fix alone, exact same scene | timeout, pick never completes |
| Retain temporal chunks despite resets, exact scene (diagnostic only) | picks/carries; times out on release |
| Commit 5/10/15 predicted actions, exact scene (diagnostic only) | picks/carries; all time out on release |
| Matched twenty-trace fit without command age, same 8000 updates/width/batch/seed | 0/5 training scenes |
| Twenty-trace clock model, 8000 updates, no observation augmentation | 1/5 training scenes |
| Same model, all 12,952 recorded training windows | chunk MAE 0.01352 rad; first-action MAE 0.01189 rad |
| Twenty-trace clock model with corrected default augmentation | 0/5 training scenes; chunk MAE 0.01470 rad |

The successful overfit episode is seed 200004: goal reached at tick 414, with
248 observation resets and all shared checker predicates passing over the final
15 ticks. Resets are expected under a contact-held jaw; eliminating their count
is not the success criterion. Seeds 200001–200005 are the fixed training-scene
evaluation set. The regression trains on twenty diverse traces, seeds
200001–200020, with DART 0.01. It uses seed 0, width 256, two residual blocks,
batch 128 and 8000 CPU updates. It has no privileged expert phase input and all
object motion comes from contact physics.

The matched no-clock model uses the same data, optimizer settings, seed and
network width/depth, removing only the clock input and its metadata. It still
times out on all five training scenes (1061–1087 resets). This is a matched
training ablation, separate from the original wide-v1 evaluation. Its chunk
MAE is 0.01829 rad and first-action MAE 0.01272 rad; the clock model's first-action
95th-percentile absolute error is 0.03272 rad.

The overfit and corrected-augmentation models each also failed the single
held-out scene 10000 in clean, real and realx0.5 noise. These small experiments
do **not** establish a held-out success improvement or repair the existing
wide-v1 weights. Keeping chunks/committing actions and changing the release
scheduler were diagnostic ablations, not shipped behavioral overrides.

## Reproduce and retrain

From the Windows checkout:

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:R2P_CACHE_ONLY = '1'
.venv/Scripts/python.exe -m r2p.policy.eval --export wide_v1 --n 5 --noise clean
.venv/Scripts/python.exe -m r2p.policy.diagnose --export wide_v1 --trace runs/train/expert_place_on_cell_00200000.jsonl --out runs/P_wide_diagnostic.json
.venv/Scripts/python.exe -m pytest -q tests/policy tests/robot
```

The exact large-data retrain command on the rig, from its checkout with its
Python environment active, is:

```sh
python -m r2p.policy.train --task tests/policy/fixtures/place_on_cell.json --traces /dev/shm/r2p/wide_ok --name wide_v2_command_age --steps 20000 --batch-size 256 --seeds 0 1 2 --width 512 --blocks 3 --device cuda:0 --memory-fraction "$LANDLORD_TORCH_MEMORY_FRACTION" --eval-n 30
python -m r2p.policy.eval --export wide_v2_command_age --n 100 --noise clean
python -m r2p.policy.eval --export wide_v2_command_age --n 100 --noise real
python -m r2p.policy.eval --export wide_v2_command_age --n 100 --noise realx0.5
```

This command assumes the original 4,449 successful traces are still at the
recorded rig path. Run it only after the lead obtains a landlord lease, applies
the returned environment and sets `LANDLORD_TORCH_MEMORY_FRACTION` to the returned
fraction. P ran no GPU job. The frozen task deliberately matches these traces;
do not train them against a subsequently edited bench survey. A real-bench
profile still requires newly collected compatible traces, its declared start
pose and the separate guarded rehearsal described in K's report.

For inference, reload or `reset()` the policy for every episode. The existing
real runner uses `replan_every=1`, matching the runtime's implicit 30 Hz clock.
Other callers must pass the actual episode `tick` when skipping control ticks.
Do not copy a new config onto old weights: the new feature changes the network's
input dimension.

## Validation and handoff

Acceptance: `.venv/Scripts/python.exe -m pytest -q tests/policy tests/robot`:
**72 passed in 145.29 seconds**, including a fresh twenty-trace collection,
training run and five physical closed-loop episodes. Implementation, regression
and evidence commit: **`f50e7b5`** (`fix(policy): preserve command progress through
contact resets`). This report's follow-up commit is recorded in the final handoff.
Local raw evidence is under `runs/P_policy_zero`, with baseline and ablation logs
in `runs/P_*.json` / `runs/P_*.log`. The versioned results JSON accompanies this
report: [P_POLICY_ZERO_RESULTS.json](P_POLICY_ZERO_RESULTS.json). The diagnostic
CLI was also exercised on the legacy failure and the successful new episode;
it reproduced goal completion at tick 414. No edits were made to the real-arm campaign, `r2p/real2sim/active.py`,
`scripts/lead`, task definitions, physics presets or other agents' files.

Not done: large-data rig retraining; a positive held-out policy gate; calibrated
real-bench placement; hardware transfer or timing validation. The twenty-trace
fit is a regression fixture and must not be presented as a deployment candidate.
