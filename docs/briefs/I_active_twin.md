# Agent I: active real2sim (Argon experiments on the real arm until the twin matches)

Read docs/briefs/COMMON.md, AGENTS.md, docs/INTERFACES.md, docs/SPEC.md sections 11-13 first. Repo
C:\Users\sarta\real2play, `.venv/Scripts/python.exe`. Commit with explicit paths only (never `git add -A`, no stash),
small commits, pull/rebase before editing shared files: agent H is concurrently fixing r2p/sim, r2p/robot,
r2p/pipeline, dashboard. Full suite before each commit:
`.venv/Scripts/python.exe -m pytest -q tests/core tests/argon tests/perception tests/robot tests/app tests/sim tests/policy`.

## Why
The product claim: Gemini 4 Argon keeps experimenting on the real rig until the sim is accurate (physics AND visuals),
so policies transfer with no adaptation. Today r2p/real2sim is PASSIVE (fits a recorded session: render, critique,
patch, hold-out score, keep). Build the ACTIVE part: Argon chooses the next experiment, the real arm runs it, the
camera and joint log observe it, the twin replays it, parameters are fitted, and held-out experiments decide.

## Bench facts
- Arm daemon (owns COM3) at http://127.0.0.1:8793. GET /state -> {"q": owner deg x6 (gripper 0-100), "tip", "rest",
  "lo", "hi", "min_z", "joint_map"}. POST /cmd: {"cmd":"xyz","xyz":[x,y,z]} fingers-down IK to the jaw midpoint
  (blocking; returns ok / refused / abort), {"cmd":"joints","owner":[6]}, {"cmd":"gripper","value":0-100},
  {"cmd":"rest"}. Read C:\Users\sarta\argon-hack\deploy\scripts\armd.py to learn its xyz FRAME and convert to the spec
  frame (r2p/frames.py; spec = mjcf - BASE_ORIGIN_MJCF) and its owner<->model joint convention (config/joint_map.json,
  active vcal_oct25). YOU MUST NOT POST to armd or open COM3: build an injectable Mover; the lead runs the real arm.
  Read-only GET /state is fine.
- Scene camera (read-only): http://127.0.0.1:8792/frame/0.jpg and /list ({"0": {"age_s", "frames"}}), ~18 fps,
  1-2 s latency; calibrated config/camera.json (0.70 px). Perception: r2p/perception (TaskPerceiver, fit_pose).
- Task: tasks/place_on_cell/task.json. Robot pieces are plastic bottle caps (cylinders ~30-37 mm, 14-20 mm tall,
  a few grams) in a supply line at x 0.10-0.20, y -0.09; printed 3x3 grid centred (0.165, 0.015), 36 mm cells.
- Twin: r2p/sim (TaskTwin, physics config config/physics/so101_bench.json: servo model that makes sag emerge,
  friction, masses ...). real2sim: r2p/real2sim (session, evaluate.physics_replay, patch.PATCH_SCHEMA, keep).
- Gemini: r2p/gemini (MAIN = gemini-4-argon for all reasoning, SMALL = gemini-3.8-flash only for per-frame pointing /
  JSON repair; no silent fallback). Tests never call Gemini: use a fake. Live budget for your own QA: 30 requests.

## Build
1. r2p/real2sim/active.py
   - Experiment menu (typed, bounded, validated before execution): hold_pose(xyz) -> static sag/tracking;
     joint_sweep(joint, amplitude_deg, speed) -> tracking lag/overshoot; push(obj, direction_deg, distance_m) ->
     approach beside the object, lower fingers to ~8 mm, push along a straight xyz path, retreat -> friction/mass;
     pick_lift_place(obj, xy, lift_m) -> grasp/slip. Safety validator: reach band 0.10-0.32 m, z >= armd min_z,
     never sweep across the grid cells, speed caps, every experiment ends at a safe height; refuse on doubt.
   - Mover protocol (execute(experiment) -> joint log + frames): ArmdMover (HTTP, thread polling /state at >= 10 Hz
     and camera frames with capture timestamps; writes an r2p_trace/1 + frames session real2sim can load) and
     SimMover (a twin with HIDDEN "truth" parameters + rendered/perceived observations) for tests and dry runs.
   - Replay in the twin with the current parameters; residuals: joint tracking (deg), sag (mm at the jaw), object
     final pose (mm, deg) from the camera vs sim, visual overlay IoU.
   - Fit: numeric least squares / bounded search over the few physics parameters that matter (servo gains / sag,
     cap friction and mass, table friction), PLUS Argon patches (existing PATCH_SCHEMA) proposed from residual tables
     and side-by-side real-vs-sim images; keep-rule on HELD-OUT experiments only.
   - Argon (MAIN) chooses the next experiment: give it the parameter estimates with uncertainty, residual history and
     the menu with bounds; JSON schema; validate; fall back to the most informative untried experiment if invalid.
   - Stop on budget or when held-out error stops improving. Live outputs for the dashboard under
     runs/active_twin/<run>/ (status.json, per-iteration real|sim side-by-side images, residual table) written
     atomically; final report.md with before/after numbers; candidate physics written to
     config/physics/so101_bench.<run>.json (never activated automatically).
2. scripts/argon/active_twin.py: `--mover sim|armd --budget N --plan-only` (plan-only prints the experiments with
   their safety verdicts and does not move anything).
3. Tests: SimMover with hidden truth (e.g. cap friction 0.45 vs prior 0.6, a different sag gain) -> the loop recovers
   the truth within tolerance and held-out error drops; the validator refuses unsafe experiments; fake Gemini.
4. Run a full SimMover campaign and a plan-only armd run (GET /state only) and put the numbers in the report.

Deadline 14:00 (demo 17:00). Report: what you built, commands, sim-campaign numbers, commits, and the exact command
the lead should run on the real arm.
