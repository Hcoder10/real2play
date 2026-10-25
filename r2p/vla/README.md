# SmolVLA placement policy

This component fine-tunes `lerobot/smolvla_base`; it never substitutes a freshly
initialized action model. Loading checks every policy tensor. The export bundles
its SmolVLM configuration/tokenizer, full fine-tuned weights, actual dataset
statistics, camera calibration, source trace hashes and evaluation results.
See `S_VLA_REPORT.md` for measured outcomes and the hardware deployment decision.

The inputs are rectified scene RGB, wrist RGB, six last-commanded MJCF joint
angles, command age, and a natural sentence translated from Argon's command.
Any joint discrepancy above four degrees resets the state input to measured q;
it does not reset command age. Outputs are 15 **absolute servo targets** at 30 Hz.
They include the expert's gravity compensation. Do not add it again.

`contracts.instruction` is shared by collection and runtime. For example,
`place_at(piece,cell)` with target `r1c2` and Argon's `place_at(piece,r1c2)` both
become `Place the piece on cell r1c2.` The target is resolved from each randomized
scene, not the nominal board. Unknown commands are refused.

The scene camera uses `config/camera.json`, including its off-center principal
point. Real scene frames must be undistorted before inference. The wrist mount
in `cameras.json` is the measured Innomaker mount in the official gripper frame;
its remaining calibration uncertainty is recorded there. Images are 256x192.
The model's padding/resize size is recorded in its saved configuration (256 or
the pretrained model's 512). Camera feature names are exactly
`observation.images.scene` and `observation.images.wrist` throughout.

Dataset replay reconstructs the recorded pre-action joint and object poses in
MuJoCo, including full object quaternions. These poses came from physical expert
rollouts. Rendering does not invent attachments. Every sixth observation is
sampled, plus command boundaries; its 15 labels remain consecutive **30 Hz**
`q_target` values, padded at command boundaries. Executed/DART targets are used
only to reconstruct commanded-state feedback, never as action labels. Each
dataset freezes task, physics, start pose and deployment convention and refuses
mixed conventions. Normalization is fitted on that dataset. Constant clamps
apply to the seven scalar state features, with the required five-standard-
deviation threshold and 1e-3 floor.

Install the `vla` extra with uv. Local Python commands use
`.venv/Scripts/python.exe`; the rig uses its existing Linux Python environment.
The offline rig additionally needs its staged wheel directory on `PYTHONPATH`.
Online weight staging is `scripts/robot/S_download_vla.py`. A pre-existing HF cache
can be staged offline with `scripts/robot/S_prepare_rig.py --root ROOT --wheels
WHEELS --cache HUB_CACHE`; supply exactly one wheel version per package.

Example rig workflow, after applying the lead's current landlord lease and
memory cap (the GPU duty must not exceed that lease):

```sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export LANDLORD_LEASE_ID=<current-lead-assigned-lease>
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
python -m r2p.vla.dataset --traces /dev/shm/r2p/wide_ok --out /dev/shm/vla_data --episodes 400 --stride 6 --workers 4
python -m r2p.vla.audit --dataset /dev/shm/vla_data --out runs/vla_camera_audit
python -m r2p.vla.train --dataset /dev/shm/vla_data --assets /dev/shm/vla_assets --out exports/smolvla_cell --steps 3000 --batch-size 64 --device cuda:0 --memory-fraction .167 --gpu-duty .60
python -m r2p.vla.evaluate --export exports/smolvla_cell --n 30 --noise both --device cuda:0 --memory-fraction .167 --batch-size 6 --replan 15
python -m r2p.vla.diagnose --export exports/smolvla_cell --dataset /dev/shm/vla_data --device cuda:0 --memory-fraction .167
```

The evaluator uses physical `TaskTwin` contact, sag and servo latency, and the
shared `check_window` for the final half second. Clean/noisy runs use identical
scene seeds; training-seed overlap is refused. The policy sees images, joints
and language only. The evaluator's state-based command scheduler is explicitly
simulation-only; it is not described as a Gemini execution. Real noise includes
joint calibration error, pixel noise, brightness, blur, stale frames and one
extra actuator-latency tick. Every result records the exact weights hash.

The real runner can use `SessionPolicy` from `adapter.py`, backed by either
`runtime.Policy` or the Torch-free `client.RemotePolicy`. Call `observe` with
current measured joints, both camera frames and the control tick, then pass it
the runner's observation. The high-level controller owns `set_command`.
`argon_command(..., ask=gemini.call)` makes one cached MAIN request at a subtask
boundary and validates the returned vocabulary. Tests inject the call and never
access hardware or Gemini.

The adapter's minimal interface is:

```python
from r2p.vla.client import RemotePolicy
from r2p.vla.adapter import SessionPolicy, argon_command

policy = SessionPolicy(RemotePolicy("http://127.0.0.1:9106"), target_cell="r1c2")
policy.set_command(argon_command(scene_bgr, "r1c2", task_status))
policy.observe(measured_q, scene_bgr, wrist_bgr, control_tick)
chunk = policy.act({"q": last_commanded_q})  # radians, absolute (15, 6)
```

The control-loop owner supplies these values; no camera capture, command
completion detection or actuator write is hidden inside this adapter. An
explicitly unqualified export must not be passed to an armed Session.

To deploy inference on the leased rig:

```sh
python -m r2p.vla.serve --export exports/smolvla_cell --device cuda:0 --memory-fraction .167 --gpu-duty .60 --max-seconds 1800 --port 9106
```

Forward port 9106 over SSH from the bench. `GET /health` identifies weights and
qualification; `POST /act` receives the Argon command, q, measured q, tick and two
base64 PNGs, and returns an absolute action chunk. The server binds loopback and
never writes to a robot. Set its lifetime to end before the assigned lease;
the service stops automatically. A concrete Argon-to-VLA invocation from saved current
camera frames is:

```sh
python -m r2p.vla.client --url http://127.0.0.1:9106 --scene scene.jpg --wrist wrist.jpg --state state.json --cell r1c2 --argon --out chunk.json
```

`state.json` contains `q`, `q_measured`, `tick`, and optionally `command_age_s`.
Do not call this live command in tests; `--command 'pick(piece)'` selects an
explicit offline command. The existing runner must retain its owner-space
limits, start audit, speed/clearance guards and robot-bus ownership. A VLA export
does not waive them. Read-only hardware qualification is:

```powershell
.venv/Scripts/python.exe scripts/robot/S_vla_check.py --export exports/smolvla_cell
.venv/Scripts/python.exe -m pytest -q tests/vla
```

A model trained on the supplied wide traces retains their original start and
joint conventions. It cannot become compatible with the current bench by
clipping or rotating its output. Hardware transfer, visual-domain agreement and
actual 30 Hz execution timing require separate evidence.
