# Real-arm tools (C1)

All commands run from the repository root. Python on Windows is
`.venv/Scripts/python.exe`; the bench uses its Python 3.12 environment with
`lerobot[feetech]==0.6.1`. Imports do not open a port or request Gemini.

## Offline acceptance

```console
.venv/Scripts/python.exe -m pytest -q tests/robot
.venv/Scripts/python.exe scripts/robot/run_policy.py --dry-run --policy stub --task cube_in_bowl --attempts 3
.venv/Scripts/python.exe scripts/robot/preflight.py --mock --policy stub
.venv/Scripts/python.exe scripts/robot/fit_sag.py --simulate .05 --out runs/sag_fit.json
```

The dry run uses C2's CPU synthetic bench and truth observations, a first-order
MockRobot, gravity sag from the rig config, compensation, the full guard, and
the shared task checker. Its summary says `source: synthbench_truth`; it does
not establish camera-perception accuracy or physical performance. A truth-only
cube-in-bowl fallback is available while C2 is absent. The stub is a scripted
smoke-test policy, not a learned export. Other tasks require a matching export.

Add `--record runs/robot` to render frames and save an attempt directory with
`trace.jsonl`, `meta.json`, and `frames/0000.jpg`, etc. The writer drops an entire
frame/tick pair when its queue fills. `q_target` is the uncompensated policy
label and `q_exec` is the guarded motor target. Recording uses CPU rendering
and may reduce real-time throughput; timing quantiles are in each summary.

## Bench sequence

1. Configure `config/bench.json` and its recursive `bench.darwin.json` overlay.
   Relative config paths resolve from the repository root. Camera intrinsics
   and extrinsics and a saved perception setup must already exist.
2. With armd stopped, run `scripts/robot/preflight.py --policy <export>`.
   This is read-only, including its connection and disconnection. FAIL returns
   exit code 1. HTTP cameras expose frame freshness but may not expose sensor
   exposure; that check reports WARN. Tests run unless `--skip-tests` is passed.
3. Verify the joint map with `mirror.py` and `read_pose.py`. The supplied audit
   map is intentionally unverified. `read_pose.py --direct` opens the bus
   read-only with armd stopped; otherwise it reads armd. Capture six or more
   fixed-jaw table touches and fit offsets offline with
   `read_pose.py --samples runs/touch_fit.json --out runs/fitted_offsets.json`.
   Fits report residual and Jacobian rank and never mark a map verified.
4. Support the arm at its physical rest pose before starting
   `scripts/robot/armd.py`. Startup primes Goal_Position from raw Present_Position
   before SOFollower.configure enables torque. The startup pose is remembered
   as the supported rest pose. The daemon is the only serial owner. A process
   lock rejects competing C1 tools. Shutdown preserves torque.
5. Run `scripts/robot/run_policy.py --policy <export> --task cube_in_bowl
   --perception-setup <saved-setup> --record runs/robot`. Real policy runs use
   armd and never reopen the serial port. An unverified map requires the explicit
   `--allow-unverified` flag on both commands; policy runs then cap speed at .5.
   The runner approaches the export's `start_q` through the guard before acting.
   Guard margins still apply to starts declared at mechanical joint limits.

No bench calibration or hardware validation has been performed by this build.
Never restart the daemon with the arm suspended: startup necessarily drops
torque for priming. A disconnected client leaves the last target held.

## Operator and local HTTP

Keys: `s`/Space hold and slowly retreat to the policy start, `p` pause, `g`
resume/next attempt, `y`/`n` override the score, `q` retreat and quit. `T` twice
within three seconds requests torque off. Daemon release is accepted only at
its supported rest pose. The file `runs/control.json` takes increasing integer
sequence numbers, e.g. `{"seq": 12, "cmd": "stop"}`. Existing commands are ignored
when a new Operator starts. Partial JSON is ignored until complete.

The daemon binds only `127.0.0.1:8793`:

- `GET /state`: owner degrees, model `q`, last target, loads, jaw xyz, rest/torque state.
- `POST /cmd`: exactly one of `{"joints": [six model radians]}`,
  `{"xyz": [spec-frame x,y,z metres]}`, `{"gripper": model_radians}`,
  `{"rest": true}`, `{"release": true}`, `{"hold": true}`, `{"stop": true}`.
- Moves interpolate at at most 20 degrees/s, slower near the table; all swept
  geometry is checked before moving. XYZ uses damped fingers-down IK. Lag/load
  aborts hold measured position with torque on. Release only succeeds at rest.
- Streaming policy clients add `"stream": true`; the elapsed-time step limit
  prevents fast clients from bypassing the speed cap. Browser-origin actuator
  requests are rejected.

`fit_sag.py` requests eight individually supervised moves through armd and fits
joint deflection against negative gravity torque. It writes a report; it does
not modify another component's physics config. `record_session.py` explicitly
requests release at rest, then records hand-guided motion without writing goals.
`mirror.py` only reads joints and overlays projected model hulls on the camera.

The LeRobot connection sequence was checked against the pinned upstream
[SOFollower implementation](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/robots/so_follower/so_follower.py)
and [motor bus API](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/motors/motors_bus.py).
