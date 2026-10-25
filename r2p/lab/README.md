# Argon Lab

Agent R's implementation of `docs/briefs/R_argon_lab.md`. The lab lets
`gemini-4-argon` choose experiments and author a static MuJoCo twin. The real
interface speaks only to the existing guarded arm daemon; it never opens the
motor bus. All rendering and simulation here run on the CPU, without GL/CUDA.

Run from the repository root with the shared Python environment:

```powershell
# Acceptance: fake Gemini + MuJoCo only, no hardware or live API calls.
.venv/Scripts/python.exe -m pytest tests/lab -q

# Entirely offline smoke turn. Choose a fresh run name each time.
.venv/Scripts/python.exe scripts/argon/lab.py --task tasks/cube_in_bowl/task.json --run lab-offline --real-mock --fake-argon --max-turns 1 --approve auto-sim

# Live Argon, simulated reference environment, one bounded API request.
.venv/Scripts/python.exe scripts/argon/lab.py --task tasks/cube_in_bowl/task.json --run lab-dry --real-mock --max-turns 1 --approve auto-sim --request-budget 1

# Exact real-bench command. Owner approves each real_plan after seeing its primitives.
.venv/Scripts/python.exe scripts/argon/lab.py --task tasks/place_on_cell/task.json --run bench-argon-lab-R --max-turns 20 --approve ask --request-budget 40
```

The real command expects the repository arm daemon on `127.0.0.1:8793`, the scene
camera on `8792`, and the wrist camera on `8794`. Both camera servers must provide
`/list` with frame age/sequence and `/frame/0.jpg`. The harness serves on
`127.0.0.1:8081`; `--port` overrides it. `--serve-only` exposes the harness without
starting the model loop. `--approve auto-sim` never authorizes physical movement;
it runs the hidden MuJoCo reference automatically when `--real-mock` is present.

Files:

- `harness.py`: HTTP `GET /obs`, `POST /plan`, imported bench motion adapters,
  live safety checks, camera freshness checks, 30 Hz state recording, sheet guard.
- `motion.py`: the shared primitive implementation, reach/grid/height/speed
  enforcement, contact descent, sustained gripper-stall verification, wrist aim.
- `simws.py`: basic calibrated SO-101 scene, static-file validation, PD replay,
  independent object labels, sealed held-out scoring.
- `render.py`, `media.py`: CPU mesh/primitive rendering and timestamp-based movies.
- `agent.py`: high-thinking Argon loop, Interactions state, owner approval,
  Antigravity static asset builds, ledger and dashboard status.
- `scripts/argon/lab.py`: CLI; `tests/lab/`: acceptance coverage.

`POST /plan` accepts, for example:

```json
{"primitives":[{"name":"goto","args":{"x":0.21,"y":-0.10,"z":0.06,"speed":12}},{"name":"wait","args":{"seconds":0.5}}],"split":"train"}
```

Coordinates are always footprint-frame metres: +x forward, +y left, +z up. Speeds
are **degrees/second**, capped at 20 and at 8 near the table. Primitives include
`goto`, `move_by`, `grasp`, `place`, `place_until_contact`, `move_until_contact`,
`look_at`, `deproject`, `open_gripper`, `close_gripper`, `home`, `wait`, and `done`.
Default open is the bench-proven owner value 20; requests outside the calibrated
IK workspace are refused even if their Cartesian point lies inside the reach band.

For real motion, imported `jaw_move.goto` and `jaw_move.gripper` retain their
calibrated JawIK, 81.9-degree owner roll and gravity compensation. Their daemon
calls are adapted locally to bounded streaming commands so long moves do not
block telemetry. `ttt_armd.park` provides the high-then-rest sequence. Geometry
and measured-height checks run on every step. Only a known daemon rest pose may
transit outside the ordinary radial working band, with all other guards active.

Grasp/place exceptions are limited to their declared XY and vertical descent or
retreat. The harness enforces 6 mm finger clearance otherwise and aborts on a
measured jaw below 7 mm. Contact descent uses 2 mm steps and relieves 4 mm when
joint encoder error exceeds 4 degrees. A grasp is verified only after sustained
closure error with a stationary encoder; an empty grasp is explicitly reported.
Printed-sheet measurements run only when the arm is away from the board; a
movement over 4 mm latches a refusal until a fresh run/survey. Missing telemetry,
stale camera evidence, daemon refusals and the 150-second episode limit stop the
motion and preserve the partial recording.

Each run retains `ledger.jsonl`, atomic `status.json`, archived per-turn images,
`sim/`, and `episodes/<n>/`. Episodes contain `primitives.json`, `states.jsonl`
(`q`, absolute `q_target`, jaw `xyz`, timestamps), exact command events,
timestamped raw JPGs, both MP4s, and `result.json`. The state poll target is 30 Hz;
a real recording below 25 Hz or with a gap over 150 ms fails its quality gate.
Frames retain their source capture timestamps rather than HTTP receipt time.
Movies hold the latest image on the actual timeline, including camera gaps.

Replay changes only robot position-servo controls after initialization. There
are no object teleports, attachments, equality constraints or applied forces.
Static XML/JSON/OBJ/STL/MTL/PNG/JPG edits are confined to `sim/`, validated as a
transaction, and rolled back on failure. Includes and asset references cannot
escape the directory. Robot assets and the footprint anchor are protected;
removing robot collision geometry, masking out the environment, adding robot
attachments, plugins, mocap, non-PD actuators or gravity changes is rejected.
Servo compliance (`kp`, `kv`, inertia/friction and latency) produces gravity sag;
`sag_gain` is compensation metadata, not an injected Cartesian sag offset.
`physics.json` also accepts `friction_overrides`, mapping a geometry or body name
to sliding friction or `[sliding, torsional, rolling]`; ordinary XML friction
remains authoritative when no explicit override is supplied.

`replay` saves synchronized real|sim movies for scene and wrist, joint RMS in
degrees, jaw-path RMS in mm, image similarity (1 minus normalized mean absolute
pixel error), and final object error from independent Argon frame reads. Missing
object labels are reported as unavailable, never as zero error. `episode_frames`
and `label_episode` let Argon inspect and freeze final-pose labels before evaluation.

Reserve at least two never-replayed episodes with `reserve_holdout`. Every movable
task object must have an independent final-frame label first. Baseline metrics
are withheld during fitting. Held-out records are fingerprinted; their replay,
labels, frame access and texture crops are blocked. `score(split="heldout")`
permanently seals further edits before returning its result. Completion requires
better held-out jaw and image residuals, no joint/object metric regression, and
Argon's explicit statement. It does not claim that two episodes establish
general real-world accuracy or policy-transfer success.

The loop follows the annotated observations, explicit plan, per-turn thought and
re-observation pattern in [Google's Argon robotics post](https://developers.googleblog.com/gemini-4-argon-for-robotics/).
Antigravity receives only the sim workspace (binary assets packed as base64
sources); it can generate meshes remotely. Returned files pass the same static
asset validator, and returned executable code is refused.

Remaining bench-dependent validation:

- No physical motion was run by agent R. The real streaming adapter is tested
  with injected protocols, but the owner must validate it on the connected bench.
- Metric wrist deprojection and `look_at` require `config/wrist_camera.json` with
  `K`, `dist`, `image_size`, `mount_body`, `mount_R` (body coordinates to OpenCV
  camera coordinates), and `mount_pos` (metres in that body). The repository has
  no measured wrist calibration. These two real operations fail explicitly until
  it exists; raw wrist capture and replay work without it. The sim has a wrist camera.
- CPU rendering uses decimated, painter-sorted triangles, simple lighting and
  planar texture sampling. Capsules use a coarse cylindrical visual proxy. Physics
  uses the full MuJoCo collision model. This is not a photorealistic renderer.
- A one-turn live dry run is evidence of a working model/tool loop, not evidence
  that the real twin has converged. Real convergence and live managed builds were
  not executed during implementation.
