# P2: command-consistent training removes most policy stalls

The 200-trace CPU proof succeeds on **181/200 exact training scenes (90.5%)**,
versus **42/200 (21.0%)** for the matched control without the two input fixes.
This passes the requested >50% gate. It is a training-scene implementation proof,
not a held-out or real-arm qualification. The evaluator, temporal ensemble,
gripper thresholds, physical contacts and shared success checker are unchanged.

## Cause and fix

The arm-only reset fixed inference, but old expert records had already reset
their `q_commanded` field whenever a squeezing jaw differed from its target.
Reading that field preserved the old bug in training. In exact trace 200001,
46.5% of noninitial recorded inputs differ from the preceding executed target
by more than four degrees. New datasets recover the preceding `q_exec` (including
DART) and then apply the current arm-only reset. Clean action labels remain
`q_target`, and outputs remain absolute MJCF targets.

The expert and inference scheduler also start commands at different times.
For trace 200001 the recorded pick/place/release/home boundaries are ticks
0/263/546/561; replaying the inference scheduler on the same recorded state
gives 0/229/466/470. A clock trained against the former is out of sync at
inference. Training now replays `CommandTracker` on past/current observations
and uses its commands consistently for one-hot features, command age and chunk
padding. It does not use expert phases as policy inputs or future state to
decide a command.

The two conventions are declared in new export metadata:
`commanded_q_source=previous_executed`, `command_alignment=state_tracker`.
Old weights retain their original diagnostic convention. Retraining is required;
copying these flags onto old weights is not a repair.

The baseline exact-scene replay reaches pick completion at tick 224, releases
the cap 13.4 mm from its cell (tolerance 10 mm), and remains in placement for
the remaining 976 ticks. Its command clock then runs beyond recorded placement
durations and the arm drifts/recloses. First-action teacher-forced MAE is only
0.00853 rad, so that metric alone hid the closed-loop failure.

Paired `diagnose` replays on the first five exact scenes found:

| Trace seed | wide_v2 result / final nearest phase | Corrected CPU result |
| --- | --- | --- |
| 200001 | Timeout, carry after release/drift | Timeout, release; placement miss |
| 200002 | Timeout, home (694 ticks in home command) | Success at tick 441 |
| 200004 | Timeout, home_up (505 ticks in home command) | Success at tick 411 |
| 200005 | Timeout, home (712 ticks in home command) | Success at tick 440 |
| 200006 | Timeout, carry after release/drift | Timeout, release; placement miss |

All five old-policy final windows fail both `at` and `gripper_open`. The two
remaining CPU failures fail only `at`. Per-phase teacher errors and rollout
phase counts (approach/descend/close/lift/carry/lower/release/retreat/home) are
in `runs/P2/diagnostics.json`; full sampled states are in its individual JSONs.

## CPU proof

- Data: the first 200 sorted successful traces from the rig's
  `/dev/shm/r2p/wide_ok`, copied with symlinks dereferenced; 127,750 windows.
- Task: the trace's task, equal to `tests/policy/fixtures/place_on_cell.json`.
  This avoids the concurrently edited real-bench task definition.
- Fit: CPU, seed 0, width 256, two residual blocks, batch 128, 16,000 AdamW
  updates at 0.0003. Observation augmentation and label smoothing disabled.
- Initial 20-scene check: 17/20. Complete proof: 181/200, all exact saved scenes,
  clean noise, full 1,200-tick budget, unchanged shared checker.
- The 19 remaining failures are three before pick completion, fourteen in the
  placement command, and two after command progression completes. They remain
  failures in the proof; no tolerance or settling-window relaxation is used.
- Weights SHA-256:
  `9e6b313b8f45c267972c007b89f08e169b7fa4fefe27781a64d580aa839ce6b1`.
- Local export: `runs/P2/cpu200/export`; all episode outcomes:
  `runs/P2/cpu200/proof.json`. Fit plus initial 20 episodes took 153.4 seconds;
  replaying all 200 with four CPU workers took 80.2 seconds.

The matched control uses the same 200 traces, seed, architecture, batch size,
16,000 updates, optimizer and disabled augmentation. It removes only the two
new dataset convention markers, preserving the historical recorded inputs and
command boundaries. Both runs use the unchanged current evaluator:

| Input convention | Clean training-scene success | Final training loss |
| --- | --- | --- |
| Historical recorded q / command boundaries | 42/200 (21.0%) | 0.001422 |
| Reconstructed executed q / inference command boundaries | 181/200 (90.5%) | 0.001497 |

This is a **69.5 percentage-point** gain under matched training conditions.
The control deliberately exits with a failed >50% gate, after saving all 200
outcomes; it is not a runtime exception. Its export and evidence are under
`runs/P2/cpu200_recorded`. The control tests the two fixes together, rather than
separating their individual contributions.

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:R2P_CACHE_ONLY = '1'
.venv/Scripts/python.exe -m r2p.policy.prove --traces runs/P2/traces --out runs/P2/cpu200 --steps 16000 --workers 4
.venv/Scripts/python.exe -m r2p.policy.prove --traces runs/P2/traces --out runs/P2/cpu200 --reuse --workers 4
.venv/Scripts/python.exe -m r2p.policy.ablate_recorded --traces runs/P2/traces --out runs/P2/cpu200_recorded --steps 16000 --workers 4
.venv/Scripts/python.exe -m pytest -q tests/policy tests/robot
```

`prove` requires exactly 200 successful input traces, records every outcome and
the weight hash, and exits with failure at success <=50%. Its default evaluates
all 200; `--eval-scenes` is only for an initial subset check. `diagnose` now adds
teacher-forced errors per phase and nearest-reference-phase rollout estimates;
these phase estimates are audit information, not exact hidden policy phases.

## Rig retraining

The authorized lease `apt_e05759e0` was verified active on
`squaredcube1-ubuntu`, GPU 0, expiring 16:21:53 PT. It is shared with S.
The model is small and uses the specified 0.167 memory fraction.
Run from the P2 checkout, using the frozen task and the same unaugmented
convention as the successful proof:

```sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export LANDLORD_LEASE_ID=apt_e05759e0 LANDLORD_BUILDING=squaredcube1-ubuntu
export LANDLORD_VRAM_GB=16.0 LANDLORD_MAX_PCT=60
export PYTHONPATH="$HOME/r2p_deps:." R2P_CACHE_ONLY=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
~/venvs/diag/bin/python -u -m r2p.policy.train \
  --task tests/policy/fixtures/place_on_cell.json \
  --traces /dev/shm/r2p/wide_ok --name wide_v3 \
  --out-dir "$HOME/real2play/exports/wide_v3" \
  --steps 60000 --batch-size 256 --seeds 0 1 2 --width 512 --blocks 3 \
  --device cuda:0 --memory-fraction 0.167 --no-augmentation --eval-n 30
```

This increases sampling exposure on 4,449 traces while retaining the tested
input/label convention. Its clean/real/realx0.5 evaluations are held-out seeded
scenes and must be inspected before promotion. No claim about noisy or held-out
improvement follows from the CPU training-scene proof.

## Validation and remaining work

Acceptance: `.venv/Scripts/python.exe -m pytest -q tests/policy tests/robot`:
**74 passed in 121.26 seconds** on the final code (initial run: 121.29 seconds).
Implementation and full CPU proof commit:
**723e296**. No hardware, Gemini calls, changes to the success checker, or edits
to `r2p/lab`, `scripts/lead` or `r2p/real2sim` were made.
The final threshold-edge parity check also passed all five temporal-context
tests: the default twin treats exactly 0.15 rad as open. This only affects that
boundary; the cap expert uses negative squeeze targets and openings above 0.35.

Exactly one rig job was launched with `setsid nohup` after the full CPU proof
passed. Code was transferred by Git bundle into isolated worktree
`/home/cube/real2play-p2` at 723e296; the shared rig checkout was not reset or
changed. Process **51141**, log `/home/cube/real2play/runs/P2_wide_v3.log`,
PID file `/home/cube/real2play/runs/P2_wide_v3.pid`. Its output destination is
`/home/cube/real2play/exports/wide_v3`. The process was alive and building the
dataset at the final 14:48 PT check. Landlord independently confirmed PID 51141 was registered
to the active lease with no limit violation at 14:45 PT. The shared lease must
remain active for S and this run.

Remaining: completed wide_v3 training and held-out comparison, real-bench start
pose/guard qualification, and hardware transfer. The CPU fit is not a deployable
bench policy. The matched CPU control establishes the benefit of the input
fixes on this fit; it is not a full-scale comparison against wide_v2's different
dataset size, architecture and augmentation.
