# State policy

The input is the last commanded six joints (reset to measured after a >4°
divergence), K × 9 object slots, the shared skill/object command one-hot,
exact jaw-midpoint/finger-axis FK, and yaw-symmetry features. The 512-wide MLP
has three residual blocks and tanh-approximate GELUs. It predicts 15 × 6
**absolute** targets at 30 Hz. There is no addition of the observed joints to
the model output.

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
boundaries, and upweights recovery. Observation augmentations include joint
calibration offsets, stale/noisy manipulated-object poses and radial/lateral
aim bias paired with the true target. All augmentation settings are recorded.

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
