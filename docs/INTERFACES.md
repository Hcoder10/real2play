# Real2Play module ownership and interfaces (binding for all agents)

Read AGENTS.md and docs/SPEC.md first. Code against THESE names; if something you need is missing, write it in your
own module or ask the lead (do not edit other owners' files).

## Ownership
| Owner | Paths |
|---|---|
| lead | r2p/frames.py, r2p/model.py, AGENTS.md, docs/*, pyproject.toml (agents may append their extra) |
| A core | r2p/tasks/, tasks/, tests/core/ (except test_model.py), r2p/traces.py |
| B sim+policy | r2p/sim/, r2p/policy/, config/physics/, exports/, tests/sim/, tests/policy/ |
| C1 robot | r2p/robot/, config/bench.json, config/bench.darwin.json, config/joint_map.json, tests/robot/, scripts/robot/ |
| C2 perception | r2p/perception/, r2p/synthbench/, tests/perception/, scripts/perception/ |
| D argon | r2p/gemini.py, r2p/real2sim/, r2p/ttt/, r2p/labeling.py, r2p/sysid/, tests/argon/, scripts/argon/ |
| E roblox | roblox/, tests/roblox/ |
| F app | r2p/cli.py, r2p/pipeline/, dashboard/, tests/app/ |

## Shared (lead): r2p.frames, r2p.model
`JOINTS, BASE_ORIGIN_MJCF, HZ=30, HORIZON=15, HOME_Q, MJCF, FIXED_TIP, MOVING_TIP, mjcf_to_spec, spec_to_mjcf,
spec_to_roblox`; `Arm().tips/geometry/jac_tip/gravity_torque/range/fk/point`; `JointMap.load(path, candidate)`,
`.to_model(owner)`, `.to_owner(model)`. All positions in the spec frame (footprint, metres).

## A: r2p.tasks (pure numpy)
- `load_task(name_or_path) -> dict` (name -> tasks/<name>/task.json); `validate(task) -> list[str]` (empty = ok)
- `sample_scene(task, rng) -> scene` (scene_spec v0: objects [{id, role, shape, size_m, xyz, yaw_deg, color, mass_kg,
  friction, fixed}], plus "task": name). `xy_choice`, `p_present`, `min_sep_m` honoured.
- `encode_slots(task, poses) -> (K, 9)` rows `[x, y, z, sin yaw, cos yaw, sx, sy, sz, present]` in task object order;
  poses = {id: {"xyz": [3], "yaw_deg": float, "size_m": [...] optional}}; missing id = absent row.
- `encode_obs(task, q, poses, command=None) -> np.ndarray` = q(6) + slots.ravel() (+ command one-hot, see below)
- `commands(task) -> list[str]` the skill commands of the task skeleton, e.g. ["pick(cube)", "place_at(cube,cell)",
  "release", "home"]; `command_onehot(task, cmd) -> np.ndarray` (fixed vocabulary of skills x object slots).
- `check(task, state) -> {"ok": bool, "per_predicate": [{"pred": str, "ok": bool, ...}]}`; state =
  {"objects": poses, "gripper_open": bool, "site_xyz": [3]}. `check_window(task, states, hz=30) -> same` (final 0.5 s).
- `r2p/tasks/vectors.json`: >= 300 check cases + window cases covering all predicates and tolerance edges (Roblox port).
- Tasks: cube_in_bowl, pick_place_stick, stack_two_cubes, push_to_target, two_cubes_in_bowl, place_on_cell (tic-tac-toe,
  3x3 x 45 mm cells centred (0.245, 0), cell ids r<i>c<j>, i along x .200/.245/.290, j along y -.045/0/+.045).

## A: r2p.traces (one format for sim experts, Roblox players, real attempts)
jsonl: line 1 `{"meta": {schema: "r2p_trace/1", task: <task dict or name>, task_name, source: "expert"|"human"|"bot"|
"real", player, level, seed, scene, success, fail_reason, commands: [[t_start, t_end, "pick(cube)"], ...], rig}}`;
then one line per 30 Hz tick: `{"t", "q" (measured, 6), "q_target" (6, the LABEL), "poses" ({id: {xyz, yaw_deg}}),
"grip_closed", "command", "phase"}`. Helpers: `write_trace(path, meta, ticks)`, `read_trace(path) -> (meta, ticks)`.

## B: r2p.sim
- `config/physics/<rig>.json` (B owns; default rig "so101_bench"): servo kp/kv/armature/frictionloss per joint, gravity
  sag gain k (rad per Nm, default 0.05), latency ticks, object default friction/mass, contact params.
- `TaskTwin(scene, physics="so101_bench", q0=HOME_Q)`: `.step(q_target)` (one 30 Hz tick, internal substeps), `.q()`,
  `.poses()` (spec frame, {id: {xyz, yaw_deg, quat}}), `.grip_closed()`, `.render(camera) -> rgb` (optional).
  The servo model sags like the real arm (physics config); grasps are physical contact.
- `r2p.sim.expert.Expert(task)`: plays the skeleton (pick, place_in, place_on, place_at, push_to, release, home) with IK
  on the jaw midpoint; emits commands per segment; DART noise on executed q only.
- `r2p.sim.gen_traces(task, n, out_dir, workers)` -> r2p_trace/1 files.

## B: r2p.policy (export format)
`exports/<name>/{config.json, norm.json, weights.npz, runtime.py}`; config: name, task, task_spec, K, obs_layout,
command_vocab, start_q, obs_const, horizon 15, action "absolute_q", eval {clean, real, realx0.5, n}. runtime.py is
pure numpy: `Policy.load(dir).act(obs) -> (15, 6)` absolute targets; obs = {"q": commanded q (6), "slots": (K, 9),
"command": str}. `r2p.policy.load(name_or_dir)`.

## C1: r2p.robot
`RealRobot(cfg)`: primed torque-on, `read() -> owner deg`, `write(owner)`, `loads()`, `hold()`, `close(torque_off=False)`;
`ReadOnlyBus`; `MockRobot(sag=GravitySag)`; `GravityComp(arm, k)`; `Guard`; `Session` (30 Hz runner: perception ->
obs (commanded q) -> policy -> ensemble -> gravity comp -> guard -> send; stops: goal_reached, policy_done, timeout,
operator); `Operator` (keys + control file); `scripts/robot/{armd,preflight,fit_sag,run_policy,record_session,mirror,
read_pose}.py`. Bench config: config/bench.json + bench.<sys.platform>.json overlay. Attempts recorded as r2p_trace/1
+ frames/ (real2sim session).

## C2: r2p.perception, r2p.synthbench
`CameraModel` (K, dist, R, t; project, deproject to the table plane; save/load JSON), ArUco board calibration,
`open_camera(cfg)` (device / http camserver / synthetic), `TaskPerceiver(cam_model, task, detector)` with
`.setup(frame)` (one SMALL-role Gemini call boxes objects; Lab appearance models), `.save_setup/.load_setup`,
`.estimate(frame, q) -> {"poses": {id: {xyz, yaw_deg, seen, source}}, ...}`, `.check(...)`. Synthetic bench:
`synthbench.World(task, rng, cam)` renders any task's objects + arm with MuJoCo, rigid attach, truth poses, truth_check.

## D: r2p.gemini (+ real2sim, ttt, labeling, sysid)
`gemini.call(parts, *, role="main"|"small", tag, schema=None, thinking=None) -> (text, record)`; `image_part(bgr)`,
`video_part(path)`; errors `QuotaError`, `CacheMiss`; record has served_by, latency, cached. Env per AGENTS.md.
`r2p.real2sim`: session loader (r2p_trace/1 + frames), anchors, twin loop (render -> critique -> patch -> keep rule),
physics replay with r2p.sim, `task_from_video`, live output `out/live/<run>/run.json` (r2s_live/1).
`r2p.ttt`: board geometry from tasks/place_on_cell, `decide(frame, cam, history, robot_desc, human_desc) -> {board,
move, why, model, source}` (one MAIN call, local validation, minimax fallback). `r2p.labeling.segment(trace) ->
commands` (Argon labels game episodes into skill segments). `r2p.sysid`: experiment planner + parameter fit (physics).

## E: roblox/ (Rojo)
Harness per SPEC section 12: lobby with ~20 tables (sit to play), config presets (robot, task, game, physics),
JointSim servo model reading physics/<rig> values, TaskCheck port (r2p/tasks/vectors.json), game modes incl.
Tic-Tac-Toe vs Robot, traces in r2p_trace/1 uploaded to `scripts/trace_server.py` (E owns roblox/tools/trace_server.py).

## F: app
`python -m r2p run --task <t> [--rounds N] [--dry-run|--real]` orchestration (stages, gate vs incumbent on identical
scenes, manifests runs/<task>/manifest.json r2p_run/1), `python -m r2p status`; dashboard (REAL, REAL2SIM, GAME,
POLICY + /runs) reading the file contracts above.
