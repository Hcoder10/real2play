# State policy

The input is the last commanded six joints (reset to measured after a >4°
divergence), K × 9 object slots, the shared skill/object command one-hot,
exact jaw-midpoint/finger-axis FK, and yaw-symmetry features. The 512-wide MLP
has three residual blocks and tanh-approximate GELUs. It predicts 15 × 6
**absolute** targets at 30 Hz. There is no addition of the observed joints to
the model output.

New exports also include `command_age_s`: elapsed control time since the current
command began. The timed expert can hold an almost identical contact pose before
lifting or releasing, so instantaneous joints/slots alone cannot identify its
next action. Training derives this clock from trace timestamps and command
boundaries; runtime resets it on command changes, not on measured-q resets.
Old exports without `temporal_context` retain their original input layout and
need retraining to use this feature.

New training exports declare `commanded_q_source: previous_executed` and
`command_alignment: state_tracker`. The loader reconstructs the preceding
executed target (including DART) before applying the arm-only four-degree reset.
This repairs old expert records whose `q_commanded` field was already reset by
a contact-held gripper. It also replays the evaluator's state-based command
tracker on recorded observations, so the command one-hot, elapsed clock and
chunk boundaries agree with inference. Expert phase labels never enter the
policy. Existing exports without these markers retain their original diagnostic
input convention and need retraining to benefit from the repair.

From the repository root (use `.venv/Scripts/python.exe` on Windows):

```sh
python -m r2p.sim.traces --task place_on_cell --n 100 --out-dir traces/cell --workers 1
python -m r2p.policy.train --task place_on_cell --traces traces/cell --name cell --steps 1000 --device cpu --seeds 0 1 2
python -m r2p.policy.eval --export cell --n 100 --noise clean
python -m r2p.policy.eval --export cell --n 100 --noise real
python -m r2p.policy.eval --export cell --n 100 --noise realx0.5
```

`r2p.sim.gen_traces(task, n, out_dir, workers)` accepts at most four workers.
It preserves unsuccessful attempts with their failure reason. DART perturbs
execution only; q_target remains the clean expert label. Training smooths arm
chatter within a phase, preserves gripper edges, pads action chunks at command
boundaries, and upweights recovery. Augmentations include joint
calibration offsets, stale/noisy manipulated-object poses and radial/lateral
aim bias paired with the true target. All augmentation settings are recorded.
Calibration offsets transform both observed joints and every absolute target in
the chunk; pose noise and DART still leave the clean action labels unchanged.

CUDA requires a current landlord lease, its environment, and the returned
PyTorch memory fraction (`--memory-fraction`). Agent B's checks use CPU only.
Do not start a second heavy process while training or evaluating on this laptop.

An export includes standalone `runtime.py`, config, normalization, and an NPZ
of all seed members. Copy the directory to a machine with NumPy, then call
`Policy.load(path).act({"q": q, "slots": slots, "command": command})`.
The config declares start pose, frame, command vocabulary, constant-feature
clamps, model/augmentation details, weights hash and actual evaluation results.
Never-varied features are clamped only when more than five training standard
deviations away (1e-3 floor). `act_ensemble` exposes individual seed predictions.

Call `policy.reset()` before a new episode. `act` and `act_ensemble` advance the
clock by one control tick per call by default (the real runner's default
`replan_every=1`). Callers that replan less often must supply `obs["tick"]`, the
integer 30 Hz episode tick; repeated queries at the same tick do not advance time.
For teacher forcing or externally tracked commands, pass `command_age_s`
explicitly. `features` and `predict_features` are stateless diagnostic methods.

Diagnose an existing export using recorded observations and the exact saved scene:

```sh
python -m r2p.policy.diagnose --export wide_v1 --trace runs/train/expert_place_on_cell_00200000.jsonl --out runs/policy_diagnostic.json
python -m pytest -q tests/policy tests/robot
```

The twenty-trace CPU regression fits clean observation windows, measures both
chunk and first-action error, then requires at least one physical closed-loop
success on five fixed training scenes. This is an overfit implementation check,
not a deployment qualification. See `P_POLICY_ZERO_REPORT.md` for the ablations,
remaining failures and the rig retraining command.

The stronger P2 proof fits exactly 200 successful traces and evaluates their
saved scenes with contact physics and the unchanged success checker:

```sh
python -m r2p.policy.prove --traces runs/P2/traces --out runs/P2/cpu200 --steps 16000 --workers 4
python -m r2p.policy.prove --traces runs/P2/traces --out runs/P2/cpu200 --reuse --workers 4
```

It exits unsuccessfully unless more than half of the evaluated scenes succeed.
The default evaluates all 200; `--eval-scenes` is only a preliminary subset check.
`diagnose` reports teacher-forced errors per expert phase and the nearest recorded
phase during rollout. The latter is an audit estimate, never a policy input.

The evaluator uses the shared task checker over the final half second and
state-based subtask completion. All noise modes use identical seeded scenes.
`real` adds persistent joint/pose bias, observation jitter, stale frames and
additional action latency; `realx0.5` halves their magnitudes/probabilities.
The baseline twin still sags and has servo latency in `clean` mode.

Integration contract: these expert labels are **servo targets**, including the
expert's gravity feedforward. Exports declare `target_space: "servo"` and
`gravity_compensation: "included_in_expert_targets"`. A consumer must not add
gravity compensation a second time to those targets. Nominal kinematic targets
from another source still need the runner's normal compensation. A high-level
controller must also advance the command: the runtime obeys `obs.command` and
does not silently replace an explicit Gemini instruction. `CommandTracker` in
`r2p.policy.eval` is the simulation-only state-based scheduler available to callers.

The tiny CPU test verifies the pipeline and NumPy/Torch parity; it is not a
trained deployment candidate. Successful hardware transfer, the held-out
physical twin gate, and a large-data/GPU trained incumbent remain separate
validation steps for the lead.

The current real bench additionally requires calibrated owner-space bounds.
`scripts/robot/rehearse_policy.py --export NAME` uses the same guarded runner as
`scripts/robot/run_real_policy.py`, starts at the captured supported rest pose,
and evaluates every measured cell against all three measured supply caps.
An incompatible `start_q` is refused before any command, never clipped into a
different start distribution. Existing HOME_Q exports exceed the real wrist-flex
limit under `vcal_oct25b`.

For fresh round-cap traces, opt into
`.venv/Scripts/python.exe -m r2p.sim.traces --task place_on_cell --n N --out-dir traces/cell_bench
--workers 1 --bench-profile scripts/robot/K_BENCH_PROFILE.json`.
This profile fixes wrist roll at the calibrated owner 81.9 degrees (model -82.5
degrees), uses the owner-20 opening / owner-6 squeeze mapped through open_rad
2.699, lowers travel height, and declares a reachable start with compensation
headroom. Training carries the profile into the export, bounds its absolute
outputs, and refuses mixed deployment conventions. The default expert remains
unchanged. Regenerate all traces; changing old export angles after training is
not a valid migration. The profile is a constraint convention, not evidence of
successful grasps: inspect the recorded expert failures before a large run.
