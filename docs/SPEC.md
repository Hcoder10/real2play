# ARGON real2sim2real: full project specification (rebuild-from-scratch edition)

Snapshot: Sun Oct 25 2026, ~07:55 PT, repo `C:\Users\sarta\argon-hack` at commit `f0ddd35` (+ this file).
Purpose: everything needed to recreate the system in a new repository with no access to the old one. Part I is the
design (what, why, how, constants, numbers, traps). Part II (appendices) reproduces the exact contracts, configs, task
files, operator guides and a file-by-file manifest of the old code.

How to read: section 3 (frames) and section 4 (contracts) are load-bearing: every component depends on them. Section 9
(pitfalls) lists bugs that each cost hours; avoid them by construction. Section 10 is the rebuild order with
acceptance tests.

---------------------------------------------------------------------------------------------------------------------

## 0. One-paragraph summary

A robot-agnostic **real2sim2real harness** for an SO-101 6-DoF arm, built for the Gemini 4 Argon hackathon (Cerebral
Valley + Google DeepMind, SHACK15 SF, Oct 25 2026; Track 2 "render what it made, watch the result, critique it against
the goal, fix it in a loop"; judging 50 % live demo, 25 % usefulness, 15 % originality, 10 % pitch). The robot does a
task once on its own (scripted probe), **Gemini 4 Argon** watches that video + joint log and writes a **task spec**
(objects, goal predicates, skill skeleton) and builds a **digital twin** of the scene, then renders the twin over the
real video, critiques the mismatch and patches the scene in a loop (keep-best, no-regression rule). The twin feeds two
data engines: **bot experts** in a MuJoCo twin and **human players** in a phone-first **Roblox game** that runs the same
arm physics (Luau port of MuJoCo's servo model). A small **state-based policy** (joints + object poses, 30 Hz, 15-step
action chunks) is trained on that data, gated against the incumbent on identical scenes, and deployed to the real arm
with a safety guard; the real attempt is recorded and fed back to Argon to correct the twin, closing the loop. A live
**dashboard** shows REAL -> REAL2SIM -> GAME -> POLICY. The demo centrepiece is **tic-tac-toe vs the robot**: Argon reads
the board from the camera and chooses the move (with a stated reason), the arm places the piece.

---------------------------------------------------------------------------------------------------------------------

## 1. Requirements and decisions (and why)

| Decision | Why |
|---|---|
| Idea 2 (real2sim -> game -> crowd data -> policy) over Idea 1 (Gemini + co-trained VLA drawing) | IK beats a VLA for drawing (1 deg joint error = 3-4 mm tip error); no owner demos for an overnight VLA post-train; LEGO needs ~0.5 mm, SO-101 has 3-7 mm |
| Gemini = brain (scene builder, critic, task writer, game move), small policy = hands | measured Gemini latency 1.3-5 s/call; control needs 30 Hz |
| **State-based policy** (joint angles + object poses), not pixels | the Roblox/MuJoCo render never has to look like a real camera; perception supplies poses |
| Roblox for crowd data | players are already there, runs on phones; physics made faithful via JointSim |
| Task spec JSON drives everything | a new task needs no new code: twin, experts, training, perception, game level, success check all read it |
| One success checker, three callers (Python, Luau, real-arm perception) | sim, game and real arm agree on what "done" means; shared test vectors |
| Gate: new policy deploys only if it beats the incumbent on identical scenes + noise | prevents regressions; the demo policy can never be replaced by a weaker one |
| ArUco board fixes the camera; Argon keypoints are a cross-check | keypoint-only calibration was 86-89 cm off (free focal trades against distance); with known focal 5.3 cm; ArUco 0.2 cm |
| Positions for the loop come from robot perception, not the twin | robot perception 0.1-0.14 cm vs twin 0.6-2.8 cm |
| Gravity compensation ON + policy fed last COMMANDED q | delta policies integrate real sag into the table (see 9.1) |
| Models: `gemini-4-argon` = MAIN role (all reasoning), `gemini-3.8-flash` = SMALL role (per-frame pointing, single-frame boxes, JSON repair) | owner decision; never silently fall back (would demo the wrong model) |
| Tic-tac-toe as demo | judge plays the robot (interactive), Argon visibly reasons, each robot move is one pick-and-place (proven skill), maps to a fun Roblox level |

Rejected: Connect 4 (vertical slot needs ~2 mm precision, 7x6 board exceeds reach), cornhole (it is Google's own
example), throwing (untested servo speed, gear stress). Curling (slide a puck to a target; physics-heavy, fits Track 2)
is a possible later act: pieces were designed to double as pucks.

---------------------------------------------------------------------------------------------------------------------

## 2. System architecture

```
                 +-------------------------------+
  real arm  ---> | recorded attempt (video+joints)| <---------------------------------------------+
  (probe)        +---------------+---------------+                                               |
                                 v                                                               |
              Argon: task_from_video -> tasks/<t>/task.json  (objects, goal predicates, skeleton) |
                                 v                                                               |
              Argon: twin loop (render over real, critique, patch, keep-best) -> scene.json      |
                                 v                                                               |
   +-----------------------------+-----------------------------+                                 |
   v                                                           v                                 |
 MuJoCo TaskTwin + bot expert (skills)          Roblox game (same arm physics, levels from tasks) |
   traces (q_target labels, DART)                 human + bot traces (meta.task)                  |
   +-----------------------------+-----------------------------+                                 |
                                 v                                                               |
                train (GPU via landlord lease) -> export                                         |
                                 v                                                               |
                eval in twin (clean + real-noise) -> GATE vs incumbent (same scenes)             |
                                 v                                                               |
                deploy on real arm (guard, gravity comp, perception) -> recorded attempt --------+
                                 |
                                 v
                 dashboard: REAL | REAL2SIM | GAME | POLICY  (+ /runs)
```

Repository layout (old repo; keep it for the rebuild):

| Dir | Owns |
|---|---|
| `spec/` | the contracts: scene_spec, policy_io, task_spec |
| `argon_tasks/` | task spec load/validate/sample, obs encoding (K slots), success `check()`, test vectors |
| `tasks/<name>/task.json` | one task per folder (+ scene.json when fitted) |
| `real2sim/` | Gemini client (roles, cache, quota), camera calibration, anchors, render/critique/fix loop, physics replay, task_from_video, live output |
| `roblox/` | Rojo project: twin arm (JointSim), SceneLoader, stations, game (levels, scoring, UI), trace upload, joint_server.py, tests |
| `policy/` | MuJoCo twin (sim.py, TaskTwin), scripted + generic expert, trace gen, training, eval, exports, runtime (numpy) |
| `deploy/` | real-arm stack: robot I/O, joint map, kinematics, guard, camera, perception, runner, scripts, configs, prints |
| `loop/` | close_loop.py (attempt + twin -> sim config -> traces -> retrain command) |
| `argon/` | one-command orchestration `python -m argon run`, manifests, gate |
| `dashboard/` | live 4-panel demo page + /runs |

---------------------------------------------------------------------------------------------------------------------

## 3. Frames, units, conventions (LOAD-BEARING)

- **Spec frame** (scene_spec, tasks, perception, deploy): robot base frame in metres, x forward (the arm reaches +x at
  q=0), y left, z up, **origin on the table surface at the base FOOTPRINT centre**.
- **MJCF frame**: the official TheRobotStudio `so101_new_calib.xml` base. `BASE_ORIGIN_MJCF = [0.0168, 0, -0.0024]`
  (footprint centre / mesh bottom in the MJCF base frame). spec = mjcf - BASE_ORIGIN_MJCF.
- **Training frame trap**: Roblox SceneLoader and policy/sim.py put spec xy at the MJCF base origin (they only lift by
  BASE_Z = 2.4 mm). Deploy/real2sim use the footprint. They differ by **16.8 mm in x**. Exports declare
  `"obs_frame": "mjcf_base_xy"`; deploy wraps them in `FrameShift` (adds the offset to every present object slot).
  Unfixed, d_ens3_search dropped 0.935 -> 0.625 clean. Rebuild: pick ONE frame for everything (footprint) and drop the shift.
- **Roblox mapping**: 1 m = 10 studs, spec (x, y, z) -> Roblox (X, Z, -Y) i.e. `Vector3.new(p[1], p[3], -p[2]) * 10`,
  table top at Y = 7.5 studs (SceneLoader.ORIGIN = CFrame.new(0, 7.5, 0)).
- **Joint order** everywhere: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.
- **Owner degrees** = LeRobot 0.6.1 `so_follower` readings with `use_degrees=True` (gripper 0-100). **Model** = MJCF
  radians. Joint map: `model_deg = sign * owner_deg + offset_deg` (arm), gripper linear between closed_rad and open_rad.
  Candidates in deploy/config/joint_map.json; active `via_corpus_audit` offsets [0, -11.5, 0, 0, -76] deg (UNVERIFIED;
  wrist roll evidence says ~90 deg off). Verify with the mirror + touch fit at the bench.
- **Action**: chunk of H = 15 absolute joint targets (MJCF rad) at 30 Hz starting at the next tick, executed with
  ACT-style temporal ensembling (weights exp(-m * age_rank), m = 0.1, oldest weighted most).
- **Policies predict deltas on the q they are given** (export runtime adds to obs q). This is why sag integrates (9.1).
- Cell ids for tic-tac-toe: `r<i>c<j>`, i = row along x (.200/.245/.290, r0 nearest the robot), j = column along y
  (-.045/0/+.045, c0 = robot's right). Grid centre (0.245, 0), 45 mm cells.

---------------------------------------------------------------------------------------------------------------------

## 4. Contracts (exact texts in Appendix A)

1. **scene_spec v0** (spec/scene_spec.md): engine-agnostic scene JSON: robot, table, objects (box/cylinder/sphere/bowl,
   `container: true`), cameras (pos/lookat/fovy), provenance.
2. **policy_io v0** (spec/policy_io.md): cube_in_bowl obs (q, cube_xyz, cube_size, bowl_xy, bowl_d), action chunk, trace
   jsonl (meta line, then per-tick `t, q, q_target, ee_cmd, cube_pose, bowl_xy, phase`). **q_target is the label, never q_exec.**
3. **argon_task/1** (spec/task_spec.md): one JSON per task: objects (shape, size_m, color, role, grasp), goal predicates
   (closed vocabulary of 11: inside, on_top, at, near, lifted, grasped, upright, yaw, not_moved, stacked + gripper_open
   rule), skeleton skills (pick, place_in, place_on, place_at, push_to, release, home, wait), randomize ranges (+
   `xy_choice`, `p_present`), limits (episode_s, max_objects K). `xy` may name an object (target zone). Goals may be
   `"optional": true` (only checked when the object is present). Policy obs: q + K slots x 9
   `[x, y, z, sin yaw, cos yaw, sx, sy, sz, present]`.
4. **Success**: `argon_tasks.check(task, state)` -> `{ok, per_predicate}`; all must hold over the final 0.5 s
   (`check_window`), gripper open unless `grasped` is a goal. Ported to Luau (Twin/TaskCheck.lua); shared vectors
   (argon_tasks/tests/vectors.json, 315 cases + 5 window cases) must pass in both.
5. **r2s_live/1** (real2sim live output for the dashboard): `real2sim/out/live/<run>/run.json` (status, stage,
   iters[{iter, file, score, kept, best_so_far, thumb}], best_iter, stop_reason, calls{by_model, served_by},
   calibration, anchors, session_kind, video), per-iteration `iter_N.json` (metrics, kept/why/verdict, patch, timings,
   images) and `iter_N_overlay_J.jpg` (1280x480: real frame with sim outlines | sim render) + `iter_N_physics.jpg`.
   Written atomically at every stage.
6. **argon_run/1** manifest: `runs/<task>/manifest.json`, per round per stage status/result, written atomically.
7. **Game traces**: cube_in_bowl human rounds `human_*.jsonl` (legacy keys kept); every other level/bot/tutorial
   `game_<level>_*.jsonl` with meta.task (dict), meta.task_name, meta.level, meta.tutorial, meta.source, fail_reason,
   ticks with K x 9 slots, obj_pose, grip_closed. Tic-tac-toe: one trace per player move with target_cell and board.
8. **Export**: `policy/export/<name>/{config.json, norm.json, weights.npz, runtime.py}`; config has task, task_spec,
   K, yaw_sym, obs_frame, start_q, obs_const, eval, horizon, fk. `Policy.load(dir).act(obs) -> (15, 6)`; pure numpy.

---------------------------------------------------------------------------------------------------------------------

## 5. Hardware, machines, accounts

- **Arm**: SO-101 follower (Feetech STS3215 x6, 12 V brick), the OLD follower with calibration `follower_1.json`
  (homing offsets 1706, -1862, -1849, -1394, 775, -1488; ranges in Appendix C). A second (new) follower's gripper servo
  is seized (does not move at 40 % torque even with the horn free); not used. The **leader arm is unusable**: its servos
  are 7.4 V parts (motor 2 max-voltage 8 V, others 12 V) and only a 12 V supply exists (voltage-error flag; LeRobot
  sees no motors). Use the hand-guided follower (torque off) instead.
- **Ports (Mac)**: follower `/dev/tty.usbmodem5B3D0496341`, (leader `/dev/tty.usbmodem5B610354601`).
- **Bench computer**: Mac mini (Apple silicon, 48 GB, macOS 26.4) on the tailnet `squaredcube@100.123.146.17`;
  repo `~/argon/repo`, venv `~/argon/.venv` (uv, Python 3.12, `lerobot[feetech]==0.6.1`, numpy 2.2.6, scipy 1.18.1,
  mujoco 3.13.0, opencv-contrib 4.13.0.92). Synced from the laptop by git bundle + scp.
- **Cameras**: wrist camera Innomaker U20CAM-720P (on the claw) + a scene camera on a tripod (mount HIGH, >= 40 deg
  down, never move after calibration). macOS only lets a Terminal-launched process open cameras: `camserver.py`
  (launched via `open ~/argon/camserver.command`) serves `/frame/<i>.jpg` on 127.0.0.1:8792; everything else reads
  over HTTP (`HttpCamera`).
- **Laptop**: Windows 11, RTX 5080 (GPU only via the **landlord** MCP lease system), Roblox Studio (+ studio-bridge
  plugin on localhost:3000), Bambu Studio, Git Bash. Memory is tight (31 GB): one heavy process at a time.
- **Printers**: Bambu H2C (dual nozzle, AMS: black, dark blue, brown, teal + external white on the left nozzle) and P2S
  (AMS; its bed was faulty, flipped). Print files in deploy/assets/print.
- **Gemini**: the owner's key is free tier: ~20 requests/day/model; gemini-4-argon had **0 quota** (429) until billing
  is enabled. All responses cached on disk by sha256(model, config, parts); `R2S_CACHE_ONLY=1` = zero requests.

---------------------------------------------------------------------------------------------------------------------

## 6. Component specifications

### 6.1 argon_tasks (pure numpy)
API: `load_task`, `validate(task) -> errors`, `sample_scene(task, rng) -> scene_spec`, `encode_slots(task, poses) ->
(K, 9)`, `encode_obs(task, q, poses) -> 6 + 9K`, `legacy_obs` (cube_in_bowl 13-value obs, byte-compatible with the
legacy runtime, tested over 200 random states), `check(task, state)`, `check_window`, `yaw_symmetry` (k = 4 square box,
2 stick/oblong, 0 round), `slot_size(shape, size, axis)`, `pose_of(p)`, `poses_from_scene`. CLI: `python -m argon_tasks
validate|sample|check`. Randomize: per-object polar ranges r/az, yaw range, min_sep_m, `xy_choice` (discrete spots;
two objects on different spots), `p_present`. Tests: 12; vectors generated for the Luau port.

### 6.2 Tasks (tasks/<name>/task.json, Appendix D)
cube_in_bowl (38 mm brown cube, bowl 110 mm x 30 mm blue), stack_two_cubes, push_to_target, pick_place_stick
(screwdriver-like 12 cm stick, modelled as a SQUARE prism: a round stick slips out of voxel-box fingers 0/8),
two_cubes_in_bowl, place_on_cell (tic-tac-toe: 3x3 x 45 mm cells at (0.245, 0); target cell slot drawn over 9 cells;
up to 4 other pieces as obstacles p_present 0.5; goal at(cube, cell, 12 mm) + optional not_moved(piece, 10 mm)).

### 6.3 Real2sim (real2sim/)
- **Gemini client** (`r2s/gemini.py`): roles `R2S_MODEL_MAIN` (default gemini-4-argon) and `R2S_MODEL_SMALL`
  (default gemini-3.8-flash), legacy `R2S_MODEL` = main. Per-request disk cache; `R2S_CACHE_ONLY=1` raises CacheMiss.
  A PerDay / limit-0 429 on main raises QuotaError (exit 1) unless `--allow-fallback` / `R2S_ALLOW_FALLBACK=1`; RPM 429s
  back off and retry (do NOT match the word "billing": every 429 contains it). `served_by` logged per call. Bounded
  thinking budget (`R2S_THINKING`, default 2048). JSON repair (e.g. dropped "label" key) reuses the cached reply.
- **Data**: public SO-101 episode (`public:0` = BrutalCaesar/phi_so101_cubes_cylinder_recovery_v1 ep 0, front camera)
  or a bench/deploy session folder (`frames/NNNN.jpg`, `joints.jsonl` t/owner_deg/frame/cmd_owner_deg, `meta.json`
  with inline joint map, `camera.json`). Synthetic sessions with ground truth: `scripts/synth_bench.py`,
  `synth_task_bench.py`.
- **Camera**: ArUco GridBoard (DICT_4X4_50, 4x3, 40 mm markers, 10 mm gaps) intrinsics + placed-board extrinsics; or
  keypoint calibration (Gemini points at robot keypoints, FK gives 3D, solvePnP) as a cross-check (needs known focal).
- **Anchors**: grasp/release frames found from the robot's own gripper COMMANDS (the empty-closed reference = the most
  closed command; percentiles picked the hold plateau); jaw-gap gives object width; release point must be inside the
  container. On public:0 the joint log and camera disagree 3.5-4.3 cm, so anchors are not enforced there.
- **Twin loop** (`scripts/run_bench.py`): iteration 0 from detection (+ perception fitters in task mode), then each
  iteration: render sim over real on critic frames (eval frames are held out and never shown to the critic), Argon
  critiques and returns a validated patch, score on held-out frames (object IoU, centroid error, anchor error,
  physics replay trajectory error). **Keep rule** (`r2s/keep.py`): keep only if score improves > 0.003 AND no
  object's held-out IoU drops > 0.10 below its best AND (anchors not enforced) no container IoU drops > 0.02 AND the
  camera term does not worsen (+1 px) AND (anchors enforced) grasp-anchor error does not worsen (+0.5 cm) and release
  stays inside; otherwise REJECTED with a reason. Stop on plateau (2 iterations without gain). ~5-7 requests/run,
  ~100-240 s.
- **Physics replay** (`r2s/physics.py`): MuJoCo replay of the recorded commands with the fitted scene; traj error px;
  objects named `obj_<id>` inside MuJoCo (a task object named "base" collided with the robot body).
- **task_from_video** (`scripts/task_from_video.py`): ONE main-role request with the WHOLE episode as a video + first
  and last frame + joint log (5 Hz table) + anchor facts; validated with argon_tasks (max 2 re-asks with errors);
  randomize filled locally; grasp width pinned to the jaw-gap anchor if Gemini is > 25 % off; then the twin loop in
  task mode -> tasks/<name>/{task.json, scene.json}.
- Results: synthetic bench cube 3.1 mm, bowl 4.1 mm; screwdriver->green mat 2.0 mm / 1.7 deg, valid first try 8.6 s;
  bare mp4 correct task in 1 request; public:0 "put the red cube in the cardboard box".

### 6.4 Roblox twin and game (roblox/, Rojo)
- **JointSim** (`ReplicatedStorage/RobotKit/JointSim.lua`): MuJoCo's servo model (full mass matrix + bias, armature)
  integrated in Luau at 480 Hz; HingeConstraint motors follow it each frame; links stay real bodies for contact;
  torque cap. Needed because SO-101 armature is 100-1000x link inertia (inertia-box recipe gave 100 mm rms error).
  Results vs MuJoCo: FK 0.001 mm, joints 0.01-0.06 deg rms, EE 1.3 mm rms, push 5-12 mm, friction grasp 4/4.
- Collision = 4 mm voxel boxes (825 parts, Transparency 1); visuals = 13 imported MeshParts in
  `ReplicatedStorage/SO101Meshes` (SmoothPlastic, PLA yellow 242,206,92, dark-grey motors), also in
  `meshes/SO101Meshes.model.json` so `rojo build` keeps them. **Parts must be server-owned** (else the joining client
  takes network ownership and the arm thrashes).
- **Stations**: up to 8 per server (per-player twin), touch joystick + UP/DOWN + TWIST + GRAB/DROP, keyboard/gamepad;
  `Twin/TouchMap.lua` mapping (56 tests). ArmIK for end-effector control.
- **Game**: menu "TEACH THE ROBOT", big "PLAY THE ROBOT" (tic-tac-toe) button, 10 s tutorial (ghost hand + coach
  bubble), daily challenge (same seed for all), 6+ level cards generated from tasks/*.json (`tools/gen_levels.py` ->
  `Twin/TaskData.lua`), stars (time + clean grasp), points (base 100 + speed + clean grab, combo up to x3), streak,
  leaderboards (all / level / daily / VS ROBOT), gripper colour unlocks at 2/5/8/12/16 stars, juice (GO!, grab click,
  sparkles, swish, thud + camera shake, confetti, last-10-s ticks, haptics), goal glow + gripper shadow dot (depth cue),
  REAL ARM card (animated placeholder, `/realarm <assetid>` sets a photo), "N rounds today, you've sent M" counter,
  attract mode (bot plays behind the menu, never recorded). Success is decided server-side by `Twin/TaskCheck.lua`
  (exact port of argon_tasks.check; holds 0.5 s).
- **Tic-Tac-Toe vs Robot**: player places the brown cube on a free square (centre within 18 mm, at rest, jaws open,
  0.5 s) -> it becomes their X; the robot answers 0.8 s later with an O disc (minimax with 25 % slip, never skips a
  winning move); 40 s per move; win 300 / draw 100; tests: 3095 checks incl. exhaustive game tree.
- **Bots**: `SkillOperator` plays every task skeleton (cube 85 %, stack 85 %, stick 90 %, push 80 %, two cubes 60 %,
  place_on_cell 3/4); `ScriptedOperator` for the legacy cube task; headless `node tools/runmode.mjs spikes/collect.luau`.
- **Trace upload**: per round, parts of <= 600 frames with retries to TRACE_URL/TRACE_KEY (server-only TwinConfig.lua;
  owner `/traceurl` chat command). Server `tools/joint_server.py --port 8770 --key <secret> --out out` (also /q /since
  /sse /ws live joint streaming at 60 Hz; POST /episode /set; 32 MB body limit; UTF-8 meta). Public URL via cloudflared.
- Tests: levels_test 5803, taskcheck_test 315 + 5 windows (1150 checks), touchmap 56, tictactoe 3095; runner
  `C:\Users\sarta\g4studio\backend\bin\luau.exe tests/<t>.luau` from roblox/ (never lune).
- Publish (owner): File > Publish under the owner account (meshes belong to it); Game Settings: Allow HTTP Requests,
  Studio API ON, Phone/Tablet, Public, Max Players 8; publish again; `/traceurl https://<tunnel>`; dashboard game_url.

### 6.5 MuJoCo twin + experts (policy/)
- `sim.py`: official SO-101 + Roblox voxel collision boxes, table, cube, bowl (disc + 16 staves); `sample_scene`
  (cube 25-50 mm, bowl d 100-160 mm, mass 20-100 g, friction 0.4-1.0, polar r 0.19-0.30 az +-55); `TaskTwin` for K
  objects of any shape (rolling friction for lying cylinders). The twin sags only 0.1-0.2 mm (real arm 5-12 mm).
- `scripted.py`: phase machine (approach above, descend, close, lift, carry, open, retreat), damped least-squares IK
  on the jaw midpoint, fingers-down + roll hold in the null space, several seeds; `home_q()` = training start pose
  `[-5e-05, -0.06786, -0.28181, 1.65806, 0.0, 0.9]` rad (fingers 12 cm above the table).
- `expert.py`: generic skill player from the skeleton (wrist roll so jaws close across the narrow side, placement that
  tracks the held object, spreading several objects in one container, re-grasp/re-push recovery; place_at: release
  only to 0.5 rad and square the held object to the cell, re-measure the grasp offset before lowering). DART
  perturbations on q_exec only. Expert success: cube_in_bowl 0.94, stack 0.88, push 0.89, stick 0.93, two_cubes 0.96,
  place_on_cell 0.74 (median placement 6.9 mm). Fragile to 1 cm perception bias (0.34-0.57).

### 6.6 Policy learning (policy/)
- Model: MLP, width 512, 3 residual blocks, GELU, inputs = obs + FK features (gripper site + finger direction) +
  yaw-symmetry features `[sin k*yaw, cos k*yaw, sin k*(yaw - bearing), cos k*(yaw - bearing)]`; output 15 x 6 deltas.
- Recipe that worked (legacy cube, d_ens3_search): joint-offset augmentation (essential: without it offset-only 0.345),
  stale-cube augmentation, **aim relabel** (`--aim-mm 10 4`: shift the cube estimate 10 mm outward / 4 mm sideways
  while labels grasp the true cube), **recovery up-weight** (`--miss-weight 4`), **chatter smoothing** (average IK
  chatter instead of cutting episodes: cutting dropped 40 % before release), 3-seed ensemble, runtime **grasp search**
  (after a detected miss re-open and shift the estimate). Position-noise aug, wider gripper, bigger model, b1-only hurt.
- **obs_const**: obs dims that never varied in training are clamped to the training value when > 5 training stds away
  (a ±20 deg zone yaw was a ~300-sigma input and took pick_place_stick_v1 from 0.87 to 0.00; plain-mean clamping broke it).
- Data volume: K-slot policies need ~3000+ successful traces (cube 0.11 @364, 0.58 @1622, 0.80 @3583, plateau 0.82).
- Evals: twin, n = 100-200, noise sets clean / real (pos + offset + latency + stale) / real x0.5. Every export records
  its eval; `policy/tests/test_task_exports.py` re-runs fixed seeds and requires identical outcomes.
- Training is GPU work: landlord `rent_apartment` with an honest purpose, launch with the returned env, apply
  `torch.cuda.set_per_process_memory_fraction`, `move_out` with an outcome. ~40-190 s per 10-30k steps on the RTX 5080.

### 6.7 Real-arm stack (deploy/)
- `robot.py`: `RealRobot` (LeRobot SOFollower; torque-on is PRIMED: torque off -> read present -> Goal := present ->
  configure(); never drops torque except an explicit operator command), `ReadOnlyBus` (no writes; must keep the
  SOFollower object alive or GC closes the port), `MockRobot` (dry runs; sag model), `GravitySag` (dq = -k * tau_g(q)
  on lift/elbow/wrist_flex), `GravityComp` (target + k * tau_g(target), <= 3 deg/joint, applied BEFORE the guard),
  `load_magnitude` (LeRobot 0.6.1 already sign-decodes Present_Load; masking -32 gave 992 = phantom overload).
- `jointmap.py`, `kin.py` (MuJoCo FK/Jacobian of the official model in the footprint frame, link/gripper/finger hull
  geometry for clearance), `guard.py` (joint limits - 2 deg margin, per-tick step cap, slow zone near the table, finger/
  gripper/link clearance vs table plane, roll-excursion cap, refusals HOLD with torque on; 60 consecutive refusals ->
  held_after_refusals; retreat never refused).
- `camera.py`: LiveCamera (cv2 index + backend), HttpCamera (camserver), StillCamera, SynthCamera, `open_camera(cfg)`.
- `perception.py`: CameraModel (K, dist, R, t; deproject onto the table plane), HSV cube/bowl perception (legacy),
  ArUco board. `taskperception.py` + `appearance.py`: generic: ONE small-role Gemini call boxes every task object on a
  setup frame; GrabCut + task colour; per-object **Lab** histogram model (L 8 x a* 32 x b* 32) fitted against the rest of
  the frame, back-projection tracking at 30 Hz (~5.7 ms / 3 objects), online background learning of whole distractor
  blobs (rolled back if the object would lose > 10 % of its pixels), pose: hull-over-yaw for boxes/sticks, rim circle
  for bowls, FK-attach while held, stacked objects re-posed on their support's top; lost > 1 s -> async main-role
  re-detect (<= 6 per run, 1 per 8 s; 0 when replaying a saved setup). Saved setups (`--setup-out` / `--perception-setup`)
  make policy runs zero-request. Real wrist-cam frames: 0 wrong locks, IoU .89-.97 on grasp frames (HSV had 4 wrong locks).
- `policy.py`: `load_policy` (stub = ScriptedPolicy; exports via runtime.py), `FrameShift`, `task_obs`, `TASK_HOME_Q`,
  `policy_start_model` (export start_q, else expert home for task exports, else bench start), `Ensembler`, `check_chunk`.
- `runner.py` Session: 30 Hz loop: read q, grab frame, perceive, wait until objects seen, obs (task K-slot or legacy),
  act every `replan_every` ticks, ensemble, gravity comp, guard.project, load check, send, record, operator poll,
  **goal_reached** stop (perceived goal held 1 s, object released, jaw midpoint >= 7 cm up), policy_done, timeout.
  **policy_q = commanded** (default): the policy sees the last commanded q, reset to measured when |cmd - meas| > 4 deg.
  Summary: outcome, timing p50/p90, refused/clipped ticks, perceived + truth success, argon_tasks auto_check.
- `record.py`: attempts as real2sim sessions (writer thread, bounded queue; a full queue drops frame AND row).
- `operator.py` + `kb.py`: keys s/SPACE stop (hold, slow retreat), p pause, g go, y/n score, q quit, T twice within
  3 s torque off; POSIX cbreak keyboard on macOS; control file `deploy/runs/control.json` {"seq", "cmd"}.
- `synth.py`/`synth_task.py`: synthetic bench (MuJoCo render of the official arm, table, ArUco board; legacy World:
  brown cube + bowl; task_world: any task's objects incl. screwdriver with yellow handle, rigid attach in the gripper
  frame when the jaws close within 2.5 cm, release onto container floor / target top, finger-pad push; truth_check).
- Scripts: `armd.py` (one process owns the bus; HTTP :8793 /state, POST /cmd joints|xyz|gripper|rest|release;
  path clearance + lag/load abort), `camserver.py`, `preflight.py` (read-only: EEPROM calib == file, supply, temps,
  torque, errors, camera fps/exposure, calib file, joint map verified, policy acts, tests), `fit_sag.py` (fit k from 8
  supervised armd moves), `calib_camera.py`, `mirror.py`, `read_pose.py` (touch fit), `verify_offsets.py`,
  `record_session.py`, `perceive.py`, `run_policy.py`, `tictactoe.py`, print generators.
- `ttt.py` + `scripts/tictactoe.py`: per robot turn ONE main-role request (scene frame with the 9 cells projected and
  labelled + a perspective-warped top-down crop + move history + piece descriptions) -> {board, move, why}; local
  validation (earlier marks unchanged, at most one new human mark, legal move), one re-ask with the error, then local
  minimax; the answering model is printed; "argon_says" on the publisher's status.json. Robot's piece picked from a
  measured supply mark (x .23, y .15) as a known pose; `--piece-size`, `--robot-desc`, `--human-desc`. Dry games: robot
  won (3/3 placements within 7 mm), drew vs minimax (4/4), 30x30x14 mm piece 3/3 within 1 mm.

### 6.8 Loop + orchestration (loop/, argon/)
- `loop/close_loop.py`: compare the twin's replay with reality (truth/operator/perception); append one focus scene per
  attempt to `loop/rounds/<r>/sim_config.json` (positions + bowl size from perception, bowl height from bench.json,
  cube size from the jaw-gap anchor, friction/mass from the twin only when its replay agrees); generate traces
  (`gen_traces.py`, 200 in ~29 s); print retrain/redeploy/next-round commands; `eval_focus.py`. One round ~4.5 min.
- `argon`: `python -m argon run --task <name|task.json|"instruction"> [--video <session>] [--rounds N] [--dry-run|--real]
  [--main-model --small-model --allow-fallback --no-cache-only] [--force r1:<stage> --no-cascade] [--device cuda|mps|cpu]`
  and `python -m argon status`. Stages per round: session -> twin -> close -> data -> train -> eval -> gate -> deploy;
  idempotent + resumable, cached under runs/<task>/<round>/. Plan mode (no flag) stops at each arm step and prints
  the exact bench command. Gate line names the incumbent and the scenes. Dry runs cache-only by default (cache miss =
  BLOCKED); daily 429 = BLOCKED. Train refuses without an active landlord lease (prints the rent step). ~6-10 min/round.
  Incumbents: cube_in_bowl d_ens3_search (both paths), pick_place_stick ens3, others <task>_v1.

### 6.9 Dashboard (dashboard/)
`server.py` (live + `--mock`), port 8800: panels REAL (camera via deploy publisher :8791 frame.jpg/status.json, cube/
bowl detection, real attempts only; dry runs hidden unless --show-dry-runs), REAL2SIM (r2s_live run, pinned via
config real2sim_run; gap = 1 - mean object IoU; motion match px chart; best twin when done; image rotation SIM / REAL /
MOTION with badge synced at 50 % of the JS cross-fade; SYNTHETIC badge for synthetic sessions), GAME (QR to game_url,
rounds/successes/hours from traces by meta, per-level strip, bots separate, leaderboard), POLICY (pinned headline_eval:
clean + real-noise, scaling curve from policy/out/scaling.json). /runs page from argon manifests (SIM DRY RUN labels).
Screenshots via `dashboard/shots.py` over Chrome DevTools in real time (virtual-time screenshots lie).

---------------------------------------------------------------------------------------------------------------------

## 7. Results (all simulation unless stated; nothing learned has run on the real arm yet)

| What | Number |
|---|---|
| Roblox twin vs MuJoCo | FK 0.001 mm; joints 0.01-0.06 deg rms; EE 1.3 mm rms; push within 5-12 mm; grasp 4/4 |
| Real2sim synthetic (v2) | cube IoU .96 / bowl .69, 3D 0.1 / 1.2 cm, motion 79 -> 13 px (old keep rule) |
| Real2sim public:0 | cube .78, 3D 4.3 cm, motion 27 -> 16 px |
| Camera from keypoints | 86-89 cm off (free focal); 5.3 cm (known focal); ArUco 0.2 cm |
| d_ens3_search (cube_in_bowl, 200 scenes) | 0.935 clean / 0.565 real / 0.87 real x0.5; on yaw +-180 task scenes 0.95 / 0.84 |
| Cube size sensitivity | 25 mm: 0.55 clean; 35-40 mm: 0.99 |
| Scaling curve (clean) | 53 / 68 / 82 / 89 / 93 % at 50 / 100 / 200 / 400 / 873 game episodes |
| Task policies (twin) | cube_in_bowl_v1 .80/.72; stack .66/.51; push .45/.34; stick_v1 .87/.72; stick_ens3 .82/.80; two_cubes .73/.57; place_on_cell_v1 .40/.35, ens3 .51/.37 |
| Sag (synthetic bench, seed 11) | d_ens3 no sag 9/12, real sag 0/12, + comp 10/12, + commanded-q 10/12; cube_v1 2/10 -> 7/10 |
| argon-trained exports with comp | 3/4 reach the goal (0/4 without) |
| Real arm (Oct 25 night) | all joints move, tracking <= 0.4 deg (elbow 1.9 deg sag); wrist-cam screwdriver pick + lift + place worked on try 2 |
| Perception (real wrist frames) | Lab+adapt: 0 wrong locks, handle IoU .89-.97 on 25 grasp frames |

---------------------------------------------------------------------------------------------------------------------

## 8. The demo (3 minutes) and the bench morning

Pitch (PITCH.md, Appendix F): problem -> Argon builds the twin from the robot's own attempt (gap shrinking, rejected
rounds shown honestly) -> people play the twin (QR) -> the robot plays tic-tac-toe vs a judge (Argon reads the board,
says why, the arm places) -> product ("any robot, any task: show it once"). Quote only numbers measured that day.
Fallbacks: scripted pick-and-place; hand-guided arm mirrored in the twin.
Morning order (DEMO_DAY.md, Appendix F): preflight -> scene camera high + camserver -> ArUco calibration -> joint
offsets (mirror, touch fit; wrist roll) -> fit_sag -> one Gemini perception setup -> recorded scripted probe -> policy
at 0.3 speed -> Argon twin of the probe -> tic-tac-toe. Pieces: judge = coins; robot = any 25-40 mm x >= 12 mm piece.

---------------------------------------------------------------------------------------------------------------------

## 9. Pitfalls and lessons (each cost real time; design them out)

1. **Delta policies integrate sag.** cmd(t+1) = measured(t) + delta; with sag the command walks into the table. Fix:
   feed the last commanded q (reset on > 4 deg divergence) and/or gravity comp. Also make the training twin sag like
   the real arm, or train absolute targets.
2. **Frame mismatch 16.8 mm** between Roblox/MuJoCo training (MJCF base xy) and deploy (footprint). One frame everywhere.
3. **Mock arm sagged the wrong way** (constant owner-degree offsets lifted the jaw): practice-bench numbers were
   misleading. Calibrate the mock to measured real behaviour before trusting dry runs.
4. **Kinematic mock grasps lie**: counted a grasp when the jaw midpoint was near the cube; the gripper closed fully on a
   held cube. Use rigid attach in the gripper frame + a jaw floor at the object width.
5. **Policies start where they were trained**: task exports stall from the folded bench pose. Declare start_q in exports.
6. **Out-of-distribution constant inputs** (zone yaw +-20 deg) -> 300-sigma input -> 0.00 success. obs_const clamp.
7. **Motors power up with Goal_Position 0**: anything that enables torque without priming jumps the arm
   (lerobot-calibrate enables torque on configure!). Prime Goal := Present first.
8. **Never reconnect RealRobot with the arm in the air** (it drops torque to prime): one daemon (armd) owns the bus.
9. **LeRobot 0.6.1 sign-decodes Present_Load**: don't mask bit 10 again (phantom 992 overloads).
10. **ReadOnlyBus GC**: keep the SOFollower object alive or the port closes after __init__.
11. **macOS camera permission**: only a Terminal-launched process (TCC) can open cameras; serve frames over HTTP.
    No `timeout` command on macOS; zsh glob errors on unmatched patterns.
12. **Leader arm at 12 V**: 7.4 V servos -> voltage-error flag, invisible to LeRobot. Check voltage limits before calibrating.
13. **Roblox client network ownership** thrashes the arm: server-own every twin part.
14. **Inertia-box recipe fails** when armature >> link inertia: run the servo model in Luau (JointSim).
15. **Studio hitches** ~2 s every ~10.5 s with plugins; keep one Studio window; SSE beats polling (~8 Hz cap on live servers).
16. **Rojo 7.6.1 cannot parse Studio's new rbxm** (Folder.Tags) -> store imported meshes as `.model.json` with asset ids.
17. **Studio bulk import uploads as the current user**: publish under the same account.
18. **Keypoint-only camera calibration** has a focal/distance ambiguity; use a board or known intrinsics.
19. **Keep rules must forbid regressions** of what the judges see (IoU), not just maximise a combined score.
20. **Gemini free tier**: 20 req/day/model, 0 for Argon without billing; cache everything; one validation call per
    prompt type on the scarce model; separate dev stand-in models from demo models; never match "billing" in 429 text.
21. **Shared checkout with many agents**: commit with explicit paths only, never `git add -A`, no stash; announce
    edits to shared files first; one heavy process at a time (laptop OOM crashed runs); GPU only via landlord leases.
22. **Headless Chrome virtual-time screenshots** run timers ahead of painting: take real-time CDP screenshots.
23. **Bambu Studio**: launching with file args fails; drive the GUI (screenshot + click helpers); print-by-layer with
    3 colours on one plate = 4 h of swaps; by-object needs spacing and splits plates; P2S bed was faulty.
24. **Physical**: SO-101 sags 9-12 mm at 30 cm reach; grasp height for a round handle must be below its centreline
    (fingertips ~3 mm above the table); wrist roll zero of the provisional joint map ~90 deg off; coins are not graspable.
25. **Positions from perception beat the twin**: use perception for loop positions, the twin for size/physics.
26. **Small evals lie**: 4/4 on one seed vs 6/12 on another; use >= 12 scenes and several seeds before quoting.

---------------------------------------------------------------------------------------------------------------------

## 10. Rebuild plan (order, with acceptance tests)

Principles: one frame (footprint) everywhere; absolute-target or commanded-q policies; every component testable with
no hardware and no Gemini (cache-only); contracts first.

| # | Build | Acceptance |
|---|---|---|
| 1 | spec/ (scene, task, policy_io, trace, r2s_live, manifest) + argon_tasks (load/validate/sample/encode/check/window) + vectors | 300+ check vectors; legacy obs byte-compatible test |
| 2 | Robot model: official so101_new_calib.xml, Kin (FK, Jacobian, hull clearance), joint map with candidates | FK vs MuJoCo 0.001 mm |
| 3 | MuJoCo TaskTwin with a calibrated sag model + rigid grasp; scripted IK expert + skill player | expert >= 0.9 on cube_in_bowl, stick; placement error reported |
| 4 | Trace gen (DART, q_target labels) + training (aim relabel, miss weight, chatter smoothing, offset aug, ensembles, obs_const, start_q) + eval (clean/real/realx0.5, fixed seeds, recorded) | cube_in_bowl >= 0.9 clean on 200 scenes; reproducible eval test |
| 5 | Deploy stack: RealRobot (primed torque), ReadOnlyBus, guard, Ensembler, runner (commanded q, gravity comp, goal_reached), operator keys + control file, armd, preflight, MockRobot calibrated to measured sag, synthetic bench (any task) | dry runs >= 0.8 for the incumbent with realistic sag; preflight 0 FAIL on the bench |
| 6 | Cameras: camserver + HttpCamera; ArUco calibration; generic Lab appearance perception with one-call setup + saved setups + cache-only | synthetic <= 2 mm; real frames 0 wrong locks |
| 7 | Real2sim: Gemini roles + cache + quota errors; anchors from commands; twin loop with no-regression keep rule; physics replay; task_from_video (whole video, one request) | synthetic task inferred correctly, objects <= 5 mm, 0 requests on rerun |
| 8 | Roblox: JointSim twin, server ownership, meshes as model.json, stations + touch controls, levels from tasks, TaskCheck port + vectors, game loop (tutorial, stars, combo, leaderboards, tic-tac-toe vs robot), trace upload + joint_server, bots | Luau tests pass; Play smoke; trace round-trips |
| 9 | Loop + argon orchestration (stages, gate vs incumbent on same scenes, plan/dry/real modes, leases) + dashboard (4 panels + /runs, honest labels) | 2-round dry run per task, cache-only, 0 requests |
| 10 | Tic-tac-toe brain (one Argon call per turn, local validation, minimax fallback) + bench runbook | dry games; frame-only board read |

Bench bring-up: preflight -> camera high + calibrate -> joint offsets verified (mirror + touch fit, wrist roll) ->
fit_sag -> perception setup (1 request) -> scripted probe recorded -> Argon twin -> policy at 0.3 -> tic-tac-toe.

---------------------------------------------------------------------------------------------------------------------

## 11. Product pipeline (owner's definition, Oct 25 08:00) + what it requires

A company that wants a policy:
1. **Connect Gemini 4 Argon to their IRL rig** (their robot + cameras + a reset-able workspace). Argon continuously
   experiments on the physical arm (system identification) and builds an accurate sim (the "reference twin").
2. **Argon deploys the game** (Roblox harness, section 12) mostly autonomously: picks presets, writes the task levels,
   injects the identified physics, runs parity tests, publishes, and brings it to the public to collect data.
3. **After enough data**, train a policy (VLA / state policy) **conditioned alongside Gemini 4**: Gemini issues
   high-level commands (subtasks), the policy does low-level control in real time (30 Hz). Real deployment data flows
   back into the twin and the training set (closed loop).

Requirements this implies (gaps to design explicitly):
- **Gemini <-> policy interface**: a command vocabulary (the task_spec skills: pick(obj), place_at(obj, target), ...,
  or language), a done/failed/stuck signal back from the policy (argon_tasks.check on perception + timeouts), re-plan
  rules, a latency budget (~1 Gemini call per subtask, policy at 30 Hz). Training data must be **segmented and
  labelled with commands**: Argon auto-labels game episodes into skill segments (the skeleton + phase boundaries).
- **Sim accuracy is a measured gate, not a claim**: identified parameters (servo gains/armature/damping, joint
  friction, gravity sag, latency, camera intrinsics/extrinsics, object mass/friction/restitution, contact softness);
  an experiment design (excitation trajectories, pushes, grasps, drops) chosen by Argon to reduce the largest residual;
  acceptance on HELD-OUT real experiments, e.g. jaw position < 2 mm rms, object end pose < 5 mm, grasp outcome
  agreement >= 95 %, timing < 1 tick. When passed: freeze, domain-randomize narrowly around the identified values,
  and monitor drift (re-identify when a real check fails). "No adaptation needed" = this gate passed.
- **Autonomous reset** for unattended IRL experiments (robot returns objects, a reset fixture, or a reset routine), and
  a **safety envelope** (workspace limits, force/load caps, contact anomaly stop, rate limits, watchdog).
- **Engine parity**: the reference twin and the Roblox runtime must produce the same trajectories for the same
  commands (JointSim-style servo model in Luau, identical parameters, automatic parity tests on every twin update).
- **Data quality**: anti-cheat/outlier filters, bots for coverage, task balancing, player skill weighting, curriculum
  that keeps play fun while targeting the policy's weak cases; success decided server-side by the shared checker.
- **Observation choice**: state-based (perception supplies poses; render need not match the camera) unless a vision
  VLA is required; then the in-game camera must match the real camera and augmentation must be heavy.
- **Evaluation**: fixed real protocol, gate vs incumbent on identical scenes, deploy only on improvement.
- **Onboarding**: URDF/MJCF import -> robot template, remote rig API for Argon, task authoring (task_from_video),
  per-customer branding. **Platform**: Roblox consent/data rules (young users), HttpService limits, DataStore quotas,
  moderation; ~20 tables per server, matchmaking across servers.

## 12. Roblox game harness (build target for the rebuild; delegate to codex gpt-6-astra xhigh, priority tier)

Goal: a reusable harness, not one game. Preset tools in easy, customizable configs to make different games and
gamify tasks; it does the hard parts for Argon (faithful arm physics, data logging, success checking, UI), while
letting Argon quickly adapt it to its own reference sim and keep it exactly matched.

- **World**: a lobby room with ~20 **tables** (stations). A player walks up and **sits at a table** (ProximityPrompt /
  Seat) to start; the camera moves to a fixed over-the-shoulder table view; touch/gamepad/keyboard controls appear.
  Standing up frees the table. Each table = one arm twin + one task instance, server-owned physics.
- **Config-driven presets** (JSON/Luau modules generated from files):
  - `robot/<name>`: kinematics + servo model params (from MJCF/URDF + identified values), meshes, collision voxels,
    joint limits, gripper model.
  - `task/<name>`: argon_task/1 (objects, goals, skeleton, randomization).
  - `game/<name>`: mode (solo, vs bot, vs player, timed, puzzle, daily), scoring (base, speed, clean grasp, combo),
    stars thresholds, round length, tutorial steps (ghost hand script), juice set (sounds, particles, camera shake),
    UI theme, leaderboard keys, unlocks.
  - `physics/<rig>`: the identified parameters (versioned; every change re-runs parity tests).
- **Hard parts done by the harness**: JointSim servo model at 480 Hz, IK end-effector control, server-authoritative
  TaskCheck (shared vectors), episode logging (30 Hz, K-slot obs, q_target labels, meta: task/level/player/source/
  fail_reason/commands), chunked authenticated upload, bots (SkillOperator) for coverage, attract mode, anti-cheat.
- **Argon API** (HTTP from the game server to the harness backend + file configs): create/modify a game from a task
  spec, set physics params, run parity tests headless, publish a level, read stats (rounds, success rates, fail
  reasons) to decide what data to collect next.
- **Parity tests**: same command sequences in the reference twin and in Roblox (headless Run mode) -> joint rms,
  jaw position, object end pose, grasp outcome; fail the publish if outside tolerance.
- **Built-in games**: Tic-Tac-Toe vs Robot (place_on_cell; minimax opponent with slip), cube in bowl, stick pick,
  stack, push, two cubes, tutorial; curling (slide a puck to a target) as a physics-showcase mode.

## 13. Visual twin: Argon calibrates appearance as well as physics (vision VLA without a sim-to-real gap)

- **Split of roles**: Roblox = data collection (states, q_target actions, Gemini commands), exact on PHYSICS (section
  11 gate). Pixels for the VLA come from **Argon's visual twin**: every game episode is replayed and rendered from the
  calibrated real camera in a renderer we control (Roblox renders differ per device and cannot be captured per tick
  at scale).
- **Composite renderer (build first)**: the scene camera is fixed, so photograph the EMPTY workspace once and render
  only the arm + objects (official meshes, task objects) with a matching camera model (intrinsics, distortion,
  extrinsics from ArUco), shadow catcher on the table plane, then composite over the real background. Parameters Argon
  tunes: object/arm albedo and roughness, light direction/intensity/colour, ambient, shadow softness, exposure, white
  balance, blur, sensor noise, JPEG quality. Later: Gaussian-splat the whole scene.
- **Argon's loop (same render -> critique -> patch -> keep-best as real2sim)**: real frames vs composite renders of the
  same recorded joint states/object poses; metrics on HELD-OUT frames: per-object mask IoU, colour error in Lab inside
  masks, edge/SSIM in arm regions, and a learned-feature distance (e.g. DINO/CLIP embedding cosine). No-regression keep
  rule. Argon chooses new real captures (poses, lighting) where the residual is largest.
- **Gate for "seamless transfer"**: (1) physics gate (section 11); (2) visual gate on held-out shots; (3) the decisive
  test: a policy trained ONLY on rendered game data succeeds on the real arm at the rate it achieves in the twin.
- **VLA**: post-train (e.g. SmolVLA 450M on the RTX 5080 via landlord lease) on rendered game episodes + real rig
  footage from Argon's experiments, conditioned on Gemini's skill commands (episodes segmented and labelled by Argon).
  Safe track in parallel: command-conditioned state policy. Decide on the real arm at 3 PM (>= 8/10 to go on stage).


=====================================================================================================================

# PART II: APPENDICES (exact texts from the old repo)


# Appendix A: contracts


## A: `spec/scene_spec.md`

# Scene spec v0 (the contract between real2sim/ and roblox/)

One JSON file describes a tabletop scene for one robot. Gemini writes it from video, the critique loop patches it,
MuJoCo (reference / renders) and Roblox (the game) both build from it. Keep it engine-agnostic.

Frame: robot base frame, metres, right-handed, **x forward (away from the base's front face), y left, z up**,
origin at the centre of the robot base footprint **on the table surface**. Angles in degrees. Colours "#rrggbb".
Roblox conversion (robot-academy recipe): 1 m = 10 studs, 1 kg = 1 RMU, MuJoCo (x, y, z) -> Roblox (x, z, -y).

```json
{
  "version": 0,
  "robot": {"type": "so101", "model": "<path or name of the MJCF used for FK>",
            "joint_convention": "mjcf_rad"},
  "table": {"size_m": [0.8, 0.6], "center_xy": [0.2, 0.0], "color": "#b08a5a", "friction": 0.6},
  "objects": [
    {"id": "cube_red", "shape": "box", "size_m": [0.03, 0.03, 0.03],
     "xyz": [0.22, 0.05, 0.015], "yaw_deg": 0, "color": "#d03030",
     "mass_kg": 0.02, "friction": 0.6}
  ],
  "cameras": [
    {"id": "front", "pos": [0.45, 0.0, 0.35], "lookat": [0.2, 0.0, 0.0], "up": [0, 0, 1],
     "fovy_deg": 55, "width": 640, "height": 480}
  ],
  "provenance": {"source": "<video / dataset episode>", "model": "<gemini model id>", "iteration": 0}
}
```

Shapes: `box` (size = full extents x,y,z), `cylinder` (size = [diameter, height]), `sphere` (size = [diameter]).
`xyz` is the object's centre. Joint trajectories travel separately as `{"t": [...], "q": [[6 joints] ...]}` in
MJCF radians, joint order = shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.

## A: `spec/policy_io.md`

# Policy I/O v0 (contract between roblox/ traces, policy/ training, deploy/ real arm)

Task v0: **pick the cube, put it in the bowl.** Both start at random spots on the table in front of the arm.
State-based policy (no pixels), so the Roblox render never has to look like a real camera.

## Observation (one step, 30 Hz)
| key | shape | meaning |
|---|---|---|
| `q` | 6 | measured joint positions, MJCF radians, order shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper |
| `cube_xyz` | 3 | cube centre, robot base frame (spec/scene_spec.md), metres |
| `cube_size` | 1 | cube edge, metres |
| `bowl_xy` | 2 | bowl centre on the table, base frame, metres |
| `bowl_d` | 1 | bowl inner diameter at the rim, metres |

On the real arm `cube_xyz` and `bowl_xy` come from the calibrated camera (deprojection onto the table plane) and
are refreshed at the camera's rate; while the cube is in the gripper its last estimate may be stale, so traces must
be robust to a stale/noisy `cube_xyz` during transport (train with that noise).

## Action
Chunk of `H = 15` absolute joint targets (MJCF radians, same order) at 30 Hz, starting at the next tick. Executed
with temporal ensembling or execute-5-replan (decide by sim eval). Gripper target in MJCF radians.

## Episode trace format (jsonl, one file per episode)
First line: `{"meta": {episode_id, source: "scripted"|"human", player, seed, scene: <scene_spec>, success, dr: {...}}}`.
Then one line per 30 Hz tick: `{"t", "q", "q_target", "ee_cmd", "cube_pose": [x,y,z,qw,qx,qy,qz], "bowl_xy", "phase"}`.
`q_target` (what the arm was told to do) is the action label. `phase` is optional (scripted players only).

## Real-arm mapping
LeRobot `so_follower` degrees -> MJCF radians: the official `so101_new_calib.xml` takes LeRobot new-calib degrees
directly (rad = deg * pi/180) on a correctly calibrated arm; the owner's arm may carry per-joint zero offsets
(see reachy-duplex docs/wg/so101_owner_to_corpus.json, `owner_to_model.official_v3`) — verified on the bench, never
assumed. Gripper: LeRobot 0-100 <-> MJCF gripper range, linear between the calibrated endpoints.

## A: `spec/task_spec.md`

# Task spec v1 (`argon_task/1`): one JSON file per task drives the whole loop

Argon writes it from a real video (or a first frame + a sentence); every stage reads it, so a new task needs no new
code: twin build (real2sim), bot experts + trace generation (policy/), training, real-arm perception + success check
(deploy/), and the Roblox level players play (roblox/). Lives in `tasks/<name>/task.json` (+ `scene.json`, a
scene_spec v0 instance, once real2sim has fitted it). Frames/units: spec/scene_spec.md (robot base frame, metres).

```json
{
 "schema": "argon_task/1",
 "name": "cube_in_bowl",
 "instruction": "put the brown cube in the blue bowl",
 "source": {"kind": "video|frame+text|hand", "path": "...", "model": "gemini-4-argon"},
 "objects": [
  {"id": "cube", "role": "manipulated", "shape": "box", "size_m": [0.038, 0.038, 0.038], "color": "brown",
   "mass_kg": 0.03, "friction": 0.6, "grasp": "top"},
  {"id": "bowl", "role": "container", "shape": "bowl", "size_m": [0.11, 0.03], "wall_m": 0.005, "color": "blue",
   "fixed": false}
 ],
 "goal": [{"pred": "inside", "a": "cube", "b": "bowl"}],
 "skeleton": [{"skill": "pick", "obj": "cube"}, {"skill": "place_in", "obj": "cube", "target": "bowl"},
              {"skill": "release"}, {"skill": "home"}],
 "randomize": {"cube": {"r": [0.19, 0.30], "az_deg": [-55, 55], "yaw_deg": [-180, 180]},
               "bowl": {"r": [0.20, 0.31], "az_deg": [-55, 55]}, "min_sep_m": 0.03},
 "limits": {"episode_s": 40, "max_objects": 4}
}
```

## Vocabulary (closed; extend here first, then everywhere)
- `shape`: `box` (size x,y,z), `cylinder` (diameter, height; lying if `"axis": "x"|"y"`), `sphere` (diameter),
  `bowl` (inner diameter at rim, height; open straight-walled), `plate` (x, y, thickness), `stick` (length, diameter;
  lies along its yaw axis: screwdriver, marker, pen).
- `role`: `manipulated` | `container` | `target` (a support or a place) | `obstacle` | `fixture` (never moves).
- `grasp`: `top` (fingers down, jaws across the narrow side) | `side` | `along` (sticks: jaws across the long axis).
- `goal` predicates (all must hold, checked on the final 0.5 s, gripper open unless `grasped` is a goal):
  `inside(a,b)`, `on_top(a,b)` (a resting on b), `at(a, xy, tol_m)`, `near(a,b,d_m)`, `lifted(a,h_m)`,
  `grasped(a)`, `upright(a, tol_deg)`, `yaw(a, deg, tol_deg)`, `not_moved(a, tol_m)`, `stacked([a,b,c])`.
- `skeleton` skills (the bot expert composes these; the policy learns the whole thing): `pick(obj)`,
  `place_in(obj,target)`, `place_on(obj,target)`, `place_at(obj, xy)`, `push_to(obj, xy)`, `release`, `home`,
  `wait(s)`. A skill may take `"approach": "top"|"side"`.
- `xy` (in `at`, `place_at`, `push_to`) is a literal `[x, y]` or an object id, meaning that object's current xy (a
  `"role": "target"` zone, e.g. a plate with `"fixed": true, "collide": false`), so a randomized target needs no code.

## Policy I/O for any task (extends spec/policy_io.md; cube_in_bowl stays byte-compatible)
Observation: `q` (6) + `objects`: K = `limits.max_objects` slots in `objects` order, each
`[x, y, z, sin yaw, cos yaw, sx, sy, sz, present]` (9), absent slots zero + present 0; `task_id` (string, for
per-task models). Action: unchanged (15 x 6 absolute joint targets at 30 Hz). Per-task models are fine; a
multi-task model conditions on the task's goal encoding. Exports declare `"task": <name>` and `"obs_frame"`.

## Success (one implementation, three callers)
`argon_tasks.check(task, state) -> {ok, per_predicate}` where `state` = object poses (+ gripper). Called by the
MuJoCo twin (truth), the Roblox game (server-side, decides the score), and the real arm (perception estimate,
operator y/n overrides).

# Appendix B: success checker semantics (reference implementation)


## B.1 predicate checker: `argon_tasks/check.py`

```python
"""The one success checker (MuJoCo twin = truth, Roblox game = score, real arm = perception estimate).

state = {
  "objects": {id: {"xyz": [3], "yaw_deg": float | "quat": [w, x, y, z], optional "v": linear speed m/s,
                   optional "size_m" (task convention; default the task's)}},
  "gripper": {"open": bool, optional "site_xyz": [3] (gripper site), optional "holding": [ids]},   (optional)
  "initial": {id: {"xyz": [3]}},                                                                    (for not_moved)
}
Absent objects (not in state["objects"]) fail every predicate that names them, unless the goal has "optional": true
(then it holds; per_predicate says "absent").

Predicates (geometry: argon_tasks.scene slot sizes; hz = sz / 2; robot base frame, m):
  inside(a, b)     a's centre within b's opening shrunk by 4 mm (bowl/cylinder/sphere: circle of radius sx/2;
                   else the yaw-rotated sx x sy rectangle) and a.z < b.z - b.hz + b.sz + a.hz (below the rim).
                   For a bowl this is exactly Twin.cube_in_bowl (policy/sim.py).
  on_top(a, b)     a's centre over b's top face (same opening test, no shrink) and a's bottom within 12 mm of b's
                   top: |(a.z - a.hz) - (b.z + b.hz)| < 0.012.
  at(a, xy, tol)   |a.xy - xy| < tol_m; xy = [x, y] or an object id (its current xy).
  near(a, b, d)    |a.xy - b.xy| < d_m.
  lifted(a, h)     a's bottom (a.z - a.hz) > h_m.
  grasped(a)       a in gripper.holding if given, else gripper closed, |site - a| < max(size)/2 + 0.02 and
                   a's bottom > 5 mm above the table.
  upright(a, tol)  angle between a's local z and world z < tol_deg (needs "quat"; yaw-only poses count as upright).
  yaw(a, deg, tol) |wrap(a.yaw - deg)| < tol_deg modulo a's symmetry: 90 deg for a square box/plate, 180 for any
                   other box/plate, stick, lying cylinder; always true for upright cylinders, spheres and bowls.
  not_moved(a,tol) |a.xy - initial[a].xy| < tol_m.
  stacked(objs)    objs listed bottom first: on_top(objs[i+1], objs[i]) for every i.
Implicit checks (also in per_predicate): "gripper_open" unless a goal is grasped(.) (skipped when the state has
no gripper), "settled": every object named by a goal moves slower than 0.5 m/s (only where "v" is given).
"""
from __future__ import annotations

import math

import numpy as np

from .scene import pose_of, slot_size
from .spec import objects_by_id

SHRINK = 0.004
ON_TOP_TOL = 0.012
V_SETTLED = 0.5
ROUND = ("bowl", "sphere")


def _geo(task_obj, st):
    xyz, yaw = pose_of(st)
    size = st.get("size_m", task_obj["size_m"]) if isinstance(st, dict) else task_obj["size_m"]
    s = slot_size(task_obj["shape"], size, task_obj.get("axis", "z"))
    return {"xyz": xyz, "yaw": yaw, "s": s, "hz": s[2] / 2, "shape": task_obj["shape"],
            "round": task_obj["shape"] in ROUND or (task_obj["shape"] == "cylinder" and task_obj.get("axis", "z") == "z")}


def _over(a, b, shrink):
    """a's centre over b's opening/top face."""
    d = a["xyz"][:2] - b["xyz"][:2]
    if b["round"]:
        return float(math.hypot(*d)) < b["s"][0] / 2 - shrink, float(math.hypot(*d))
    c, s = math.cos(-b["yaw"]), math.sin(-b["yaw"])
    lx, ly = c * d[0] - s * d[1], s * d[0] + c * d[1]
    return bool(abs(lx) < b["s"][0] / 2 - shrink and abs(ly) < b["s"][1] / 2 - shrink), float(math.hypot(*d))


def _sym_deg(g, task_obj):
    if g["round"]:
        return None
    if g["shape"] in ("box", "plate") and abs(g["s"][0] - g["s"][1]) < 1e-3 * max(g["s"][0], 1e-6) + 1e-4:
        return 90.0
    return 180.0


def _tilt_deg(st):
    if not isinstance(st, dict) or "quat" not in st:
        return 0.0
    w, x, y, z = st["quat"]
    zz = 1 - 2 * (x * x + y * y)  # world z component of the local z axis
    return math.degrees(math.acos(max(-1.0, min(1.0, zz))))


def check(task: dict, state: dict) -> dict:
    """{"ok": bool, "per_predicate": [{"pred": "inside(cube,bowl)", "ok": bool, ...measurements}]}"""
    tobj = objects_by_id(task)
    objs = state.get("objects", {})
    geo = {k: _geo(tobj[k], v) for k, v in objs.items() if k in tobj and v is not None}
    grip = state.get("gripper")
    res = []

    def need(*ids):
        miss = [i for i in ids if i not in geo]
        return miss

    def xy_of(v):
        if isinstance(v, str):
            return geo[v]["xyz"][:2] if v in geo else None
        return np.asarray(v, float)

    def on_top(a, b):
        A, B = geo[a], geo[b]
        over, dxy = _over(A, B, 0.0)
        gap = float((A["xyz"][2] - A["hz"]) - (B["xyz"][2] + B["hz"]))
        return over and abs(gap) < ON_TOP_TOL, {"dxy": round(dxy, 4), "gap": round(gap, 4)}

    for g in task["goal"]:
        p = g["pred"]
        ids = list(g["objs"]) if p == "stacked" else [g[k] for k in ("a", "b") if k in g]
        if isinstance(g.get("xy"), str):
            ids.append(g["xy"])
        label = f"{p}({','.join(ids)})" if p != "at" or isinstance(g["xy"], str) else f"at({g['a']},{list(g['xy'])})"
        miss = need(*ids)
        if miss and g.get("optional"):
            res.append({"pred": label, "ok": True, "absent": miss})
            continue
        if miss:
            res.append({"pred": label, "ok": False, "missing": miss})
            continue
        r = {"pred": label}
        if p == "inside":
            A, B = geo[g["a"]], geo[g["b"]]
            over, dxy = _over(A, B, SHRINK)
            top = B["xyz"][2] - B["hz"] + B["s"][2] + A["hz"]
            r.update(ok=bool(over and A["xyz"][2] < top), dxy=round(dxy, 4), z=round(float(A["xyz"][2]), 4),
                     z_max=round(float(top), 4))
        elif p == "on_top":
            ok, m = on_top(g["a"], g["b"])
            r.update(ok=bool(ok), **m)
        elif p == "at":
            d = float(np.linalg.norm(geo[g["a"]]["xyz"][:2] - xy_of(g["xy"])))
            r.update(ok=d < g["tol_m"], d=round(d, 4))
        elif p == "near":
            d = float(np.linalg.norm(geo[g["a"]]["xyz"][:2] - geo[g["b"]]["xyz"][:2]))
            r.update(ok=d < g["d_m"], d=round(d, 4))
        elif p == "lifted":
            h = float(geo[g["a"]]["xyz"][2] - geo[g["a"]]["hz"])
            r.update(ok=h > g["h_m"], h=round(h, 4))
        elif p == "grasped":
            if grip is None:
                r.update(ok=False, missing=["gripper"])
            elif "holding" in grip:
                r.update(ok=g["a"] in grip["holding"])
            else:
                A = geo[g["a"]]
                d = float(np.linalg.norm(np.asarray(grip.get("site_xyz", [9, 9, 9]), float) - A["xyz"]))
                ok = (not grip.get("open", True)) and d < max(A["s"]) / 2 + 0.02 and A["xyz"][2] - A["hz"] > 0.005
                r.update(ok=bool(ok), d_site=round(d, 4))
        elif p == "upright":
            t = _tilt_deg(objs[g["a"]])
            r.update(ok=t < g["tol_deg"], tilt_deg=round(t, 2))
        elif p == "yaw":
            A = geo[g["a"]]
            per = _sym_deg(A, tobj[g["a"]])
            if per is None:
                r.update(ok=True, symmetric=True)
            else:
                d = (math.degrees(A["yaw"]) - g["deg"]) % per
                d = min(d, per - d)
                r.update(ok=d < g["tol_deg"], err_deg=round(d, 2))
        elif p == "not_moved":
            ini = state.get("initial", {}).get(g["a"])
            if ini is None:
                r.update(ok=False, missing=["initial"])
            else:
                d = float(np.linalg.norm(geo[g["a"]]["xyz"][:2] - np.asarray(pose_of(ini)[0][:2])))
                r.update(ok=d < g["tol_m"], d=round(d, 4))
        elif p == "stacked":
            o = list(g["objs"])
            parts = [on_top(o[i + 1], o[i]) for i in range(len(o) - 1)]
            r.update(ok=all(x[0] for x in parts), levels=[x[1] for x in parts])
        res.append(r)
    if grip is not None and not any(g["pred"] == "grasped" for g in task["goal"]):
        res.append({"pred": "gripper_open", "ok": bool(grip.get("open", False))})
    named = set()
    for g in task["goal"]:
        named.update(g["objs"] if g["pred"] == "stacked" else [g[k] for k in ("a", "b") if k in g])
    vs = {k: float(objs[k]["v"]) for k in named if k in objs and isinstance(objs[k], dict) and "v" in objs[k]}
    if vs:
        res.append({"pred": "settled", "ok": all(v < V_SETTLED for v in vs.values()),
                    "v_max": round(max(vs.values()), 3)})
    return {"ok": all(r["ok"] for r in res), "per_predicate": res}


def check_window(task: dict, states: list, hold_s: float = 0.5) -> dict:
    """The spec's final-0.5 s rule: states = [{"t": seconds, ...state}] in time order; ok iff check() holds on every
    state with t >= t_last - hold_s. Returns the last state's check with "ok" replaced and "held_s" added."""
    if not states:
        return {"ok": False, "per_predicate": [], "held_s": 0.0}
    t_end = states[-1].get("t", 0.0)
    win = [s for s in states if s.get("t", 0.0) >= t_end - hold_s - 1e-9]
    outs = [check(task, s) for s in win]
    last = outs[-1]
    held = 0.0
    for s, o in zip(reversed(win), reversed(outs)):
        if not o["ok"]:
            break
        held = t_end - s.get("t", 0.0)
    return {"ok": all(o["ok"] for o in outs), "per_predicate": last["per_predicate"], "held_s": round(held, 3)}
```

## B.2 task spec validation + vocabulary: `argon_tasks/spec.py`

```python
"""argon_task/1 vocabulary, loading and validation (spec/task_spec.md is the contract; this is its executable form).

Argument layout of the JSON entries (every name below is a key of the entry):
  goal:      inside {a, b} | on_top {a, b} | at {a, xy, tol_m} | near {a, b, d_m} | lifted {a, h_m} | grasped {a}
             upright {a, tol_deg} | yaw {a, deg, tol_deg} | not_moved {a, tol_m} | stacked {objs: [bottom, ..., top]}
  skeleton:  pick {obj} | place_in {obj, target} | place_on {obj, target} | place_at {obj, xy} | push_to {obj, xy}
             release | home | wait {s};  any skill may add "approach": "top" | "side"
Any goal may add "optional": true: it then holds trivially when an object it names is absent from the state (a
piece that may or may not be on the board this episode). `xy` is either a literal [x, y] (robot base frame, m) or the id of an object (usually role "target"): its current
xy, so a randomized target zone works without new code.
"""
from __future__ import annotations

import json
from pathlib import Path

SCHEMA = "argon_task/1"
SHAPES = {"box": 3, "cylinder": 2, "sphere": 1, "bowl": 2, "plate": 3, "stick": 2}  # shape -> len(size_m)
ROLES = ("manipulated", "container", "target", "obstacle", "fixture")
GRASPS = ("top", "side", "along")
PREDICATES = {  # pred -> required keys (besides "pred")
    "inside": ("a", "b"), "on_top": ("a", "b"), "at": ("a", "xy", "tol_m"), "near": ("a", "b", "d_m"),
    "lifted": ("a", "h_m"), "grasped": ("a",), "upright": ("a", "tol_deg"), "yaw": ("a", "deg", "tol_deg"),
    "not_moved": ("a", "tol_m"), "stacked": ("objs",)}
SKILLS = {"pick": ("obj",), "place_in": ("obj", "target"), "place_on": ("obj", "target"), "place_at": ("obj", "xy"),
          "push_to": ("obj", "xy"), "release": (), "home": (), "wait": ("s",)}
RANDOMIZE_KEYS = ("r", "az_deg", "yaw_deg", "yaw_rel_deg", "size_scale", "size_m", "mass_kg", "friction", "x", "y",
                  "xy_choice", "p_present")


def load_task(path_or_dict) -> dict:
    """Task dict from a path (task.json or its directory) or a dict; raises ValueError listing every problem."""
    if isinstance(path_or_dict, dict):
        task = path_or_dict
    else:
        p = Path(path_or_dict)
        if p.is_dir():
            p = p / "task.json"
        task = json.loads(p.read_text())
    errs = validate(task)
    if errs:
        raise ValueError(f"invalid task {task.get('name')!r}: " + "; ".join(errs))
    return task


def objects_by_id(task) -> dict:
    return {o["id"]: o for o in task["objects"]}


def max_objects(task) -> int:
    """K, the number of observation slots."""
    return int(task.get("limits", {}).get("max_objects", len(task["objects"])))


def _is_xy(v, ids):
    return (isinstance(v, str) and v in ids) or (isinstance(v, (list, tuple)) and len(v) == 2
                                                  and all(isinstance(x, (int, float)) for x in v))


def validate(task) -> list:
    """List of human-readable problems (empty = valid)."""
    e = []
    if not isinstance(task, dict):
        return ["task is not an object"]
    if task.get("schema") != SCHEMA:
        e.append(f"schema must be {SCHEMA!r}")
    if not isinstance(task.get("name"), str) or not task.get("name"):
        e.append("name missing")
    objs = task.get("objects")
    if not isinstance(objs, list) or not objs:
        return e + ["objects must be a non-empty list"]
    ids = [o.get("id") for o in objs]
    if len(set(ids)) != len(ids) or not all(isinstance(i, str) and i for i in ids):
        e.append("object ids must be unique non-empty strings")
    ids = set(i for i in ids if isinstance(i, str))
    for o in objs:
        oid = o.get("id")
        if o.get("shape") not in SHAPES:
            e.append(f"{oid}: shape {o.get('shape')!r} not in {sorted(SHAPES)}")
        else:
            sz = o.get("size_m")
            if not isinstance(sz, list) or len(sz) != SHAPES[o["shape"]] or not all(isinstance(x, (int, float)) and x > 0 for x in sz):
                e.append(f"{oid}: size_m must be {SHAPES[o['shape']]} positive numbers for a {o['shape']}")
        if o.get("role") not in ROLES:
            e.append(f"{oid}: role {o.get('role')!r} not in {ROLES}")
        if "grasp" in o and o["grasp"] not in GRASPS:
            e.append(f"{oid}: grasp {o['grasp']!r} not in {GRASPS}")
        if o.get("shape") == "cylinder" and o.get("axis", "z") not in ("x", "y", "z"):
            e.append(f"{oid}: cylinder axis must be x, y or z")
        if "xy" in o and not _is_xy(o["xy"], set()):
            e.append(f"{oid}: xy must be [x, y]")
    K = max_objects(task)
    if K < len(objs):
        e.append(f"limits.max_objects {K} < {len(objs)} objects")
    goal = task.get("goal")
    if not isinstance(goal, list) or not goal:
        e.append("goal must be a non-empty list")
        goal = []
    for g in goal:
        p = g.get("pred")
        if p not in PREDICATES:
            e.append(f"goal: unknown predicate {p!r}")
            continue
        miss = [k for k in PREDICATES[p] if k not in g]
        if miss:
            e.append(f"goal {p}: missing {miss}")
            continue
        refs = list(g["objs"]) if p == "stacked" else [g[k] for k in ("a", "b") if k in g]
        if p == "stacked" and len(refs) < 2:
            e.append("goal stacked: objs needs at least 2 ids")
        for r in refs:
            if r not in ids:
                e.append(f"goal {p}: unknown object {r!r}")
        if p == "at" and not _is_xy(g["xy"], ids):
            e.append(f"goal at: xy must be [x, y] or an object id, got {g['xy']!r}")
    skel = task.get("skeleton")
    if not isinstance(skel, list) or not skel:
        e.append("skeleton must be a non-empty list")
        skel = []
    for s in skel:
        k = s.get("skill")
        if k not in SKILLS:
            e.append(f"skeleton: unknown skill {k!r}")
            continue
        miss = [a for a in SKILLS[k] if a not in s]
        if miss:
            e.append(f"skeleton {k}: missing {miss}")
            continue
        for a in ("obj", "target"):
            if a in s and s[a] not in ids:
                e.append(f"skeleton {k}: unknown object {s[a]!r}")
        if "xy" in s and not _is_xy(s["xy"], ids):
            e.append(f"skeleton {k}: xy must be [x, y] or an object id")
        if s.get("approach", "top") not in ("top", "side"):
            e.append(f"skeleton {k}: approach must be top or side")
    rnd = task.get("randomize", {})
    for k, v in rnd.items():
        if k == "min_sep_m":
            continue
        if k not in ids:
            e.append(f"randomize: unknown object {k!r}")
            continue
        bad = [x for x in v if x not in RANDOMIZE_KEYS]
        if bad:
            e.append(f"randomize {k}: unknown keys {bad}")
    for o in objs:
        oid = o.get("id")
        r = rnd.get(oid, {})
        if "xy" not in o and not ("r" in r and "az_deg" in r) and not ("x" in r and "y" in r) and "xy_choice" not in r:
            e.append(f"{oid}: needs a fixed xy or randomize r+az_deg (or x+y, or xy_choice)")
    return e
```

## B.3 scene sampling + obs encoding: `argon_tasks/scene.py`

```python
"""Scene sampling and the K-slot policy observation.

Geometry conventions (shared by the twin, the game and the checker):
  slot size [sx, sy, sz] = the object's extents in its own frame (x along its yaw):
    box/plate [x, y, z]; cylinder upright [d, d, h], lying "axis": "x" [h, d, d], "axis": "y" [d, h, d];
    sphere [d, d, d]; bowl [d, d, h] (d = inner diameter at the rim); stick [length, d, d] (lies along x).
  An object resting on the table has its centre at z = sz / 2 (scene_spec xyz is the centre).
Legacy cube_in_bowl obs (spec/policy_io.md, 13 values) = q, cube slot xyz, cube sx, bowl slot xy, bowl sx.
"""
from __future__ import annotations

import math

import numpy as np

from .spec import max_objects, objects_by_id

SLOT = 9  # x, y, z, sin yaw, cos yaw, sx, sy, sz, present
TABLE = {"size_m": [0.8, 0.6], "center_xy": [0.2, 0.0], "color": "#b08a5a", "friction": 0.6}
TABLE_HALF_Y = 0.29
COLORS = {"red": "#d03030", "brown": "#8b5a2b", "blue": "#3050c0", "green": "#30a040", "yellow": "#e0c020",
          "orange": "#e07020", "white": "#f0f0f0", "black": "#202020", "gray": "#808080", "grey": "#808080",
          "purple": "#8040b0", "pink": "#e070a0"}


def hex_color(c, default="#c0c0c0"):
    if isinstance(c, str) and c.startswith("#") and len(c) == 7:
        return c
    return COLORS.get(str(c).lower(), default)


def slot_size(shape: str, size_m, axis: str = "z") -> list:
    s = list(size_m)
    if shape in ("box", "plate"):
        return s
    if shape == "cylinder":
        d, h = s
        return {"x": [h, d, d], "y": [d, h, d]}.get(axis, [d, d, h])
    if shape == "sphere":
        return [s[0]] * 3
    if shape == "bowl":
        return [s[0], s[0], s[1]]
    if shape == "stick":
        return [s[0], s[1], s[1]]
    raise ValueError(f"unknown shape {shape}")


def yaw_symmetry(obj) -> int:
    """k such that the object looks the same after a 360/k deg yaw: 4 square box/plate, 2 other box/plate, stick or
    lying cylinder, 0 round (bowl, sphere, upright cylinder: yaw is meaningless)."""
    sh = obj["shape"]
    if sh in ("bowl", "sphere") or (sh == "cylinder" and obj.get("axis", "z") == "z"):
        return 0
    sx, sy, _ = slot_size(sh, obj["size_m"], obj.get("axis", "z"))
    return 4 if sh in ("box", "plate") and abs(sx - sy) <= 1e-3 * max(sx, 1e-6) + 1e-4 else 2


def footprint_r(obj) -> float:
    """Radius of the object's footprint circle (separation test): half-diagonal of its xy extents (+ bowl wall)."""
    sx, sy, _ = slot_size(obj["shape"], obj["size_m"], obj.get("axis", "z"))
    if obj["shape"] in ("bowl", "sphere") or (obj["shape"] == "cylinder" and obj.get("axis", "z") == "z"):
        return sx / 2 + (obj.get("wall_m", 0.005) if obj["shape"] == "bowl" else 0.0)
    return 0.5 * math.hypot(sx, sy)


def rest_z(obj) -> float:
    return slot_size(obj["shape"], obj["size_m"], obj.get("axis", "z"))[2] / 2


def _range(rng, v):
    return float(rng.uniform(v[0], v[1])) if isinstance(v, (list, tuple)) else float(v)


def sample_scene(task: dict, rng: np.random.Generator, tries: int = 500) -> dict:
    """scene_spec v0 instance (+ "role", "axis", "wall_m", "fixed", "collide" per object and "task") with every
    object resting on the table at a random pose drawn from task["randomize"]:
      r [lo, hi] m and az_deg [lo, hi] (polar around the base) or x, y ranges; yaw_deg [lo, hi] absolute or
      yaw_rel_deg [lo, hi] relative to the object's bearing; size_scale [lo, hi] (all dims) or size_m (per-dim
      [lo, hi] or number); mass_kg, friction [lo, hi]; xy_choice [[x, y], ...] (a uniformly drawn spot, e.g. a board
      cell; two xy_choice objects only need distinct spots); p_present (probability the object is in the scene at all;
      absent objects are left out of the scene and their slot is absent). Footprints keep min_sep_m (default 0.03) apart and inside
      the table's |y| < 0.29. Objects without a randomize entry use their own "xy" (and "yaw_deg")."""
    rnd = task.get("randomize", {})
    sep = float(rnd.get("min_sep_m", 0.03))
    base = task["objects"]
    for _ in range(tries):
        out, ok = [], True
        for o in base:
            r = rnd.get(o["id"], {})
            if "p_present" in r and rng.uniform() >= float(r["p_present"]):
                continue  # absent this episode: not in the scene, an absent slot in the observation
            ob = {"id": o["id"], "shape": o["shape"], "role": o["role"], "color": hex_color(o.get("color")),
                  "mass_kg": _range(rng, r.get("mass_kg", o.get("mass_kg", 0.03))),
                  "friction": _range(rng, r.get("friction", o.get("friction", 0.6)))}
            for k in ("axis", "wall_m", "fixed", "collide", "grasp"):
                if k in o:
                    ob[k] = o[k]
            size = list(o["size_m"])
            if "size_scale" in r:
                k = _range(rng, r["size_scale"])
                size = [x * k for x in size]
            if "size_m" in r:
                size = [_range(rng, v) for v in r["size_m"]]
            ob["size_m"] = [round(x, 5) for x in size]
            if "xy_choice" in r:  # one of a discrete set of spots (board cells), uniformly
                x, y = r["xy_choice"][int(rng.integers(len(r["xy_choice"])))]
            elif "r" in r:
                rad, az = _range(rng, r["r"]), math.radians(_range(rng, r["az_deg"]))
                x, y = rad * math.cos(az), rad * math.sin(az)
            elif "x" in r:
                x, y = _range(rng, r["x"]), _range(rng, r["y"])
            else:
                x, y = o["xy"]
            if "yaw_rel_deg" in r:
                yaw = math.degrees(math.atan2(y, x)) + _range(rng, r["yaw_rel_deg"])
            else:
                yaw = _range(rng, r.get("yaw_deg", o.get("yaw_deg", 0.0)))
            ob["xyz"] = [round(x, 5), round(y, 5), round(rest_z(ob), 5)]
            ob["yaw_deg"] = round((yaw + 180) % 360 - 180, 3)
            fr = footprint_r(ob)
            def clash(p):
                d = math.hypot(x - p["xyz"][0], y - p["xyz"][1])
                if "xy_choice" in r and "xy_choice" in rnd.get(p["id"], {}):
                    return d < 0.01  # two cell objects: only not the same cell
                return d < fr + footprint_r(p) + sep

            if abs(y) + fr > TABLE_HALF_Y or any(clash(p) for p in out):
                ok = False
                break
            out.append(ob)
        if ok:
            return {"version": 0, "task": task["name"], "table": dict(TABLE), "cameras": [], "objects": out}
    raise RuntimeError(f"{task['name']}: no collision-free scene in {tries} tries (randomize ranges too tight)")


def _yaw_from_quat(qw, qx, qy, qz):
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def pose_of(p):
    """(xyz (3,), yaw rad) from a pose given as {"xyz", "yaw_deg"} / {"xyz", "quat"} / [x, y, z, qw, qx, qy, qz] /
    [x, y, z, yaw_deg]."""
    if isinstance(p, dict):
        xyz = np.asarray(p["xyz"], float)[:3]
        if "quat" in p:
            return xyz, _yaw_from_quat(*p["quat"])
        return xyz, math.radians(float(p.get("yaw_deg", 0.0)))
    v = np.asarray(p, float).reshape(-1)
    if v.shape[0] == 7:
        return v[:3], _yaw_from_quat(*v[3:7])
    if v.shape[0] == 4:
        return v[:3], math.radians(v[3])
    if v.shape[0] == 3:
        return v, 0.0
    raise ValueError(f"pose with {v.shape[0]} values")


def encode_slots(task: dict, object_poses: dict) -> np.ndarray:
    """(K, 9) slots in task["objects"] order. object_poses: id -> pose (see pose_of); a dict pose may carry
    "size_m" (task convention, e.g. a randomized scene's or perception's measured size), else the task's size is
    used. Ids missing from object_poses (or None) are absent: all zeros, present 0."""
    K = max_objects(task)
    out = np.zeros((K, SLOT))
    for k, o in enumerate(task["objects"]):
        p = object_poses.get(o["id"])
        if p is None:
            continue
        xyz, yaw = pose_of(p)
        size = p.get("size_m", o["size_m"]) if isinstance(p, dict) else o["size_m"]
        out[k, :3] = xyz
        out[k, 3:5] = math.sin(yaw), math.cos(yaw)
        out[k, 5:8] = slot_size(o["shape"], size, o.get("axis", "z"))
        out[k, 8] = 1.0
    return out


def encode_obs(task: dict, q, object_poses: dict) -> np.ndarray:
    """Flat policy observation: q (6, MJCF rad) + K x 9 slots = 6 + 9K values."""
    q = np.asarray(q, float).reshape(-1)
    if q.shape[0] != 6:
        raise ValueError("q must have 6 values")
    return np.concatenate([q, encode_slots(task, object_poses).reshape(-1)])


def poses_from_scene(scene: dict) -> dict:
    """id -> pose dict (with size_m) from a scene_spec instance, for encode_obs."""
    return {o["id"]: {"xyz": o["xyz"], "yaw_deg": o.get("yaw_deg", 0.0), "size_m": o["size_m"]} for o in scene["objects"]}


def legacy_obs(task: dict, x) -> np.ndarray:
    """The 13-value cube_in_bowl observation (spec/policy_io.md order q, cube_xyz, cube_size, bowl_xy, bowl_d) from a
    K-slot observation of a task whose first box-shaped manipulated object is the cube and whose first bowl is the
    bowl. Byte-identical to policy/data.obs_vector on the same inputs (tests/test_obs.py)."""
    x = np.asarray(x, float)
    order = [o["id"] for o in task["objects"]]
    obj = objects_by_id(task)
    cube = next(i for i in order if obj[i]["shape"] == "box" and obj[i]["role"] == "manipulated")
    bowl = next(i for i in order if obj[i]["shape"] == "bowl")
    c, b = 6 + SLOT * order.index(cube), 6 + SLOT * order.index(bowl)
    return np.concatenate([x[..., :6], x[..., c:c + 3], x[..., c + 5:c + 6], x[..., b:b + 2], x[..., b + 5:b + 6]], -1)
```

# Appendix C: configs


## C: `deploy/config/bench.json`

```json
{
 "schema": "argon_bench/1",
 "robot": {
  "port": "COM3",
  "id": "follower_1",
  "calibration_dir": "C:/Users/sarta/ego16-storage/lerobot_calibration/robots/so_follower",
  "calibration_note": "~/.cache/huggingface/lerobot/calibration does not exist on this machine (the HF cache is a symlink to D:); the owner's follower_1.json (identical to reachy-duplex sim/so101/owner_calibration_20260916.json) lives here. Never recalibrate during a session: every offset depends on it.",
  "start_pose_owner_deg": [
   0.0,
   -78.0,
   46.2,
   86.0,
   76.0,
   4.0
  ],
  "start_pose_note": "sag-safe support pose of session 01 (lift -78 / wrist_flex 86 in the owner's frame; elbow 46.2 and roll 76 from the session-02 home). FK under both joint-map candidates: every link >= 105 mm above the table, fingers >= 122 mm.",
  "torque_on_owner_supports_wrist": true,
  "load_limit": 650,
  "load_sustain_s": 0.5,
  "temperature_soft_c": 55
 },
 "camera": {
  "index": 0,
  "backend": "DSHOW",
  "width": 640,
  "height": 480,
  "fps": 30,
  "model_file": "deploy/config/camera.json",
  "mount_note": "fixed, high (>= 40 deg down at the workspace), seeing base, table 10-35 cm in front, bowl and cube; never moved after calibration"
 },
 "board": {
  "dictionary": "DICT_4X4_50",
  "grid": [
   4,
   3
  ],
  "marker_length_m": 0.04,
  "separation_m": 0.01,
  "dist_from_base_front_m": 0.08,
  "dist_from_base_front_right_m": null,
  "left_edge_y_m": -0.095,
  "board_z_m": 0.0,
  "note": "Measure marker_length_m on the PRINT with a ruler (printers scale). Place: printed side up, top marker row toward the robot and parallel to the base front face. dist_from_base_front_m = base front face -> top edge of marker 0 (black square edge); _right_ = same at the right end of the row (gives yaw); left_edge_y_m = robot-y of marker 0's left edge (robot's right is negative y)."
 },
 "perception": {
  "cube": {
   "hsv": [
    [
     [
      5,
      90,
      30
     ],
     [
      22,
      255,
      210
     ]
    ]
   ],
   "size_m": 0.038,
   "min_px": 40,
   "centroid_height_frac": 0.5,
   "attach_radius_m": 0.035,
   "hsv_red_previous": [
    [
     [
      0,
      120,
      60
     ],
     [
      8,
      255,
      255
     ]
    ],
    [
     [
      170,
      120,
      60
     ],
     [
      180,
      255,
      255
     ]
    ]
   ],
   "color": "brown"
  },
  "bowl": {
   "hsv": [
    [
     [
      95,
      120,
      50
     ],
     [
      130,
      255,
      255
     ]
    ]
   ],
   "rim_height_m": 0.03,
   "wall_m": 0.005,
   "min_px": 400,
   "inner_d_m": 0.11
  },
  "note": "HSV ranges are OpenCV (H 0-180). Tune with scripts/perceive.py --tune on the real bench light. cube size, bowl rim height and wall by ruler. centroid_height_frac is set by scripts/perception_accuracy.py for the fitted camera.",
  "printed_note": "Oct 25: printed objects from deploy/assets/print (make_prints.py): 38 mm red cube, blue straight-walled bowl 110 mm inner (bench_plate_h2c.3mf; bowl_d130_h30.stl is the larger alternative) / 30 mm tall / 5 mm wall, matching policy/sim.py's training distribution. Two-colour printed board: markers exactly 40 mm. Cubes printed in BROWN PLA (no red loaded, Oct 25): hsv is a provisional brown range, tune with scripts/perceive.py --tune; the red range is kept as hsv_red_previous."
 },
 "guard": {
  "plane_z": 0.0,
  "link_clear_m": 0.015,
  "gripper_clear_m": 0.005,
  "finger_floor_m": 0.002,
  "slow_zone_m": 0.015,
  "r_max": 0.4,
  "z_max": 0.45,
  "limit_margin_deg": 2.0,
  "step_deg": 3.0,
  "slow_step_deg": 1.0,
  "grip_step_deg": 6.0,
  "speed": 1.0,
  "roll_excursion_deg": 120.0,
  "lag_cap_deg": 8.0
 },
 "control": {
  "hz": 30,
  "ensemble_m": 0.1,
  "replan_every": 1,
  "gripper_closed_below_rad": 0.35,
  "max_consecutive_refusals": 60,
  "episode_seconds": 40,
  "gravity_comp_k": 0.05,
  "gravity_comp_note": "Default ON (Oct 25): sag-free policies output deltas on measured q, so real sag (9-12 mm at 30 cm, measured) integrates into the table without comp. 0.05 matches that sag; refit on the bench with deploy/scripts/fit_sag.py (owner present) and put the printed k here. Capped at 3 deg/joint, applied before the guard."
 },
 "mirror": {
  "url": "http://127.0.0.1:8767/set",
  "hz": 30,
  "note": "bench joint_server on 8767 (roblox/tools/joint_server.py --source leader --port 8767) and Workspace attribute TwinServer = http://127.0.0.1:8767, so it never collides with a twin server on 8765"
 }
}
```

## C: `deploy/config/bench.darwin.json`

```json
{
 "note": "Mac mini bench (Oct 25): overlays bench.json via so101d.bench.load_cfg. Leader identified by a read-only movement watch.",
 "robot": {
  "port": "/dev/tty.usbmodem5B3D0496341",
  "leader_port": "/dev/tty.usbmodem5B610354601",
  "id": "follower_1",
  "leader_id": "leader_mac",
  "calibration_dir": "/Users/squaredcube/argon/calibration/robots/so_follower",
  "leader_calibration_dir": "/Users/squaredcube/argon/calibration/teleoperators/so_leader",
  "calibration_note": "The OLD follower (the laptop arm, follower_1.json from Sept 16; its motor EEPROM matches the file exactly), moved to the Mac on Oct 25 after the new follower's gripper servo seized. Never recalibrate mid-session."
 },
 "camera": {
  "source": "http",
  "url": "http://127.0.0.1:8792/frame/{index}.jpg",
  "index": 0,
  "width": 640,
  "height": 360,
  "fps": 30,
  "note": "camserver.py (launched from Terminal: macOS camera permission) owns the devices. index = the SCENE camera's camserver index (set it once the tripod camera is plugged in; 0 is the wrist cam tonight). 640x360 keeps the 16:9 aspect; calibrate through this same source."
 }
}
```

## C: `deploy/config/joint_map.json`

```json
{
 "schema": "argon_joint_map/1",
 "rule": "model_deg = sign * owner_deg + offset_deg (arm joints); gripper model_rad = closed_rad + g/100 * (open_rad - closed_rad). owner = LeRobot so_follower use_degrees=True readings with the active arm's calibration; model = official so101_new_calib.xml (real2sim/assets).",
 "active": "via_corpus_audit",
 "candidates": {
  "mac_fresh_identity": {
   "verified": false,
   "source": "UNUSED: was for the new Mac-mini follower (gripper servo seized Oct 25). Mac mini arms (Oct 25), freshly calibrated with lerobot-calibrate 0.6.1 (middle-pose homing). The official so101_new_calib.xml takes LeRobot new-calib degrees directly, so the prior is identity; the real2sim spike also found identity best on public SO-101 data (13.9 vs 15.8 px). PROVISIONAL until mirror + touch fit.",
   "joints": {
    "shoulder_pan": {
     "sign": 1,
     "offset_deg": 0.0,
     "sigma_deg": 5.0,
     "measured": false
    },
    "shoulder_lift": {
     "sign": 1,
     "offset_deg": 0.0,
     "sigma_deg": 3.0,
     "measured": false
    },
    "elbow_flex": {
     "sign": 1,
     "offset_deg": 0.0,
     "sigma_deg": 3.5,
     "measured": false
    },
    "wrist_flex": {
     "sign": 1,
     "offset_deg": 0.0,
     "sigma_deg": 2.0,
     "measured": false
    },
    "wrist_roll": {
     "sign": 1,
     "offset_deg": 0.0,
     "sigma_deg": 8.0,
     "measured": false
    }
   },
   "gripper": {
    "closed_rad": -0.17,
    "open_rad": 1.745,
    "measured": false,
    "source": "official so101_new_calib gripper limits; LeRobot gripper 0-100 assumed to span them. Bench-check closed/open with read_pose."
   }
  },
  "via_corpus_audit": {
   "verified": false,
   "source": "owner_to_corpus from reachy-duplex docs/wg/so101_owner_to_corpus.json (corpus audit 2026-09-19, PROVISIONAL, never bench-measured) composed with corpus->official = identity. Identity is supported by the argon real2sim spike: public corpus episode, official model, zero offsets gave 13.9 px held-out keypoint error vs 15.8 px with the simfix offsets, pan sign +1 (10.8 vs 31.8 px).",
   "joints": {
    "shoulder_pan": {
     "sign": 1,
     "offset_deg": 0.0,
     "sigma_deg": 5.0,
     "measured": false
    },
    "shoulder_lift": {
     "sign": 1,
     "offset_deg": -11.5,
     "sigma_deg": 3.0,
     "measured": false
    },
    "elbow_flex": {
     "sign": 1,
     "offset_deg": 0.0,
     "sigma_deg": 3.5,
     "measured": false
    },
    "wrist_flex": {
     "sign": 1,
     "offset_deg": 0.0,
     "sigma_deg": 2.0,
     "measured": false
    },
    "wrist_roll": {
     "sign": 1,
     "offset_deg": -76.0,
     "sigma_deg": 8.0,
     "measured": false
    }
   },
   "gripper": {
    "closed_rad": -0.17,
    "open_rad": 1.944,
    "measured": false,
    "source": "reachy-duplex sim/so101/v3/calibration.json jaw_closed_rad / jaw_span_rad (owner raw range 1467-2845 = 121 deg); open end exceeds the model limit 1.745 and is clamped by the guard"
   }
  },
  "simfix_official_v3": {
   "verified": false,
   "source": "reachy-duplex sim/so101/v3/calibration.json (wrist-camera table-normal match + two table touches on the owner's arm, 2026-09-19). Pitch split (lift/elbow/wrist) open; roll from the wrist camera.",
   "joints": {
    "shoulder_pan": {
     "sign": 1,
     "offset_deg": 0.0,
     "sigma_deg": null,
     "measured": false
    },
    "shoulder_lift": {
     "sign": 1,
     "offset_deg": -8.0,
     "sigma_deg": null,
     "measured": false
    },
    "elbow_flex": {
     "sign": 1,
     "offset_deg": -4.0,
     "sigma_deg": null,
     "measured": false
    },
    "wrist_flex": {
     "sign": 1,
     "offset_deg": -9.0,
     "sigma_deg": null,
     "measured": false
    },
    "wrist_roll": {
     "sign": 1,
     "offset_deg": -164.0,
     "sigma_deg": null,
     "measured": false
    }
   },
   "gripper": {
    "closed_rad": -0.17,
    "open_rad": 1.944,
    "measured": false,
    "source": "same as via_corpus_audit"
   }
  }
 },
 "conflict_note": "The two candidates disagree by 88 deg on wrist_roll and by up to 9 deg on the pitch joints. They cannot both be right; scripts/read_pose.py (gripper level, folded stops) and scripts/verify_offsets.py (arm silhouette vs the ArUco-calibrated camera) decide. Until then every live run is reduced-speed and needs --allow-unverified.",
 "bench_readings": {}
}
```

## C: `dashboard/config.json`

```json
{
 "host": "0.0.0.0",
 "port": 8800,
 "title": "Play the twin. Train the robot.",
 "event": "Gemini 4 Argon Hackathon - SF",
 "game_url": "https://www.roblox.com/games/0/SO-101-Twin",
 "game_name": "SO-101 Twin",
 "paths_note": "relative to the repo root (the parent of dashboard/)",
 "real2sim_live": "real2sim/out/live",
 "real2sim_run": "ep0_g37flash",
 "real2sim_mock": "real2sim/out/live/synth_g3flash",
 "traces": ["roblox/out/traces", "roblox/out/traces_dev"],
 "traces_tmp": "roblox/out/traces_tmp",
 "count_bots": true,
 "evals": "policy/out/evals",
 "headline_eval": "export_d_ens3_search_real_ensemble_200.json",
 "scaling": "policy/out/scaling.json",
 "deploy_runs": "deploy/runs",
 "runs": "runs",
 "camera_url": "http://127.0.0.1:8791/frame.jpg",
 "camera_status_url": "http://127.0.0.1:8791/status.json",
 "mock_frames": "real2sim/data/ep0/frames",
 "poll_s": 1.0
}
```

## C: follower_1.json (LeRobot calibration of the arm in use)

```json
{
    "shoulder_pan": {
        "id": 1,
        "drive_mode": 0,
        "homing_offset": 1706,
        "range_min": 841,
        "range_max": 3317
    },
    "shoulder_lift": {
        "id": 2,
        "drive_mode": 0,
        "homing_offset": -1862,
        "range_min": 872,
        "range_max": 2946
    },
    "elbow_flex": {
        "id": 3,
        "drive_mode": 0,
        "homing_offset": -1849,
        "range_min": 746,
        "range_max": 3023
    },
    "wrist_flex": {
        "id": 4,
        "drive_mode": 0,
        "homing_offset": -1394,
        "range_min": 879,
        "range_max": 3218
    },
    "wrist_roll": {
        "id": 5,
        "drive_mode": 0,
        "homing_offset": 775,
        "range_min": 0,
        "range_max": 4095
    },
    "gripper": {
        "id": 6,
        "drive_mode": 0,
        "homing_offset": -1488,
        "range_min": 1467,
        "range_max": 2845
    }
}
```

# Appendix D: task files


## D: `tasks/cube_in_bowl/task.json`

```json
{
 "schema": "argon_task/1",
 "name": "cube_in_bowl",
 "instruction": "put the brown cube in the blue bowl",
 "source": {"kind": "hand", "path": "spec/task_spec.md", "model": null},
 "objects": [
  {"id": "cube", "role": "manipulated", "shape": "box", "size_m": [0.038, 0.038, 0.038], "color": "brown",
   "mass_kg": 0.03, "friction": 0.6, "grasp": "top"},
  {"id": "bowl", "role": "container", "shape": "bowl", "size_m": [0.11, 0.03], "wall_m": 0.005, "color": "blue",
   "mass_kg": 0.25, "fixed": false}
 ],
 "goal": [{"pred": "inside", "a": "cube", "b": "bowl"}],
 "skeleton": [{"skill": "pick", "obj": "cube"}, {"skill": "place_in", "obj": "cube", "target": "bowl"},
              {"skill": "release"}, {"skill": "home"}],
 "randomize": {"cube": {"r": [0.19, 0.30], "az_deg": [-55, 55], "yaw_deg": [-180, 180], "size_scale": [0.66, 1.3],
                        "mass_kg": [0.02, 0.10], "friction": [0.4, 1.0]},
               "bowl": {"r": [0.20, 0.31], "az_deg": [-55, 55], "size_m": [[0.10, 0.16], 0.03]}, "min_sep_m": 0.03},
 "limits": {"episode_s": 40, "max_objects": 4}
}
```

## D: `tasks/pick_place_stick/task.json`

```json
{
 "schema": "argon_task/1",
 "name": "pick_place_stick",
 "instruction": "pick up the screwdriver and put it on the green square",
 "source": {"kind": "hand", "path": "real arm Oct 25: wrist-camera screwdriver pick, fingertips ~3 mm above the table, jaws across the handle", "model": null},
 "objects": [
  {"id": "stick", "role": "manipulated", "shape": "stick", "size_m": [0.12, 0.022], "color": "yellow",
   "mass_kg": 0.04, "friction": 0.7, "grasp": "along"},
  {"id": "zone", "role": "target", "shape": "plate", "size_m": [0.10, 0.10, 0.002], "color": "green",
   "fixed": true, "collide": false}
 ],
 "goal": [{"pred": "at", "a": "stick", "xy": "zone", "tol_m": 0.04}],
 "skeleton": [{"skill": "pick", "obj": "stick"}, {"skill": "place_at", "obj": "stick", "xy": "zone"},
              {"skill": "release"}, {"skill": "home"}],
 "randomize": {"stick": {"r": [0.19, 0.27], "az_deg": [-45, 45], "yaw_deg": [-180, 180], "size_scale": [0.9, 1.1],
                         "mass_kg": [0.025, 0.06], "friction": [0.5, 1.0]},
               "zone": {"r": [0.19, 0.29], "az_deg": [-50, 50]}, "min_sep_m": 0.0},
 "limits": {"episode_s": 40, "max_objects": 4}
}
```

## D: `tasks/place_on_cell/task.json`

```json
{
 "schema": "argon_task/1",
 "name": "place_on_cell",
 "instruction": "put your brown cube on the chosen tic-tac-toe cell",
 "source": {
  "kind": "hand",
  "path": "tic-tac-toe demo: Argon reads the board and picks the cell; the arm places its cube",
  "model": null
 },
 "board": {
  "doc": "3x3 printed grid of 45 mm cells centred at (0.245, 0.0), robot base frame (m). Cell id r<i>c<j>: i = row along x (r0 nearest the robot, x .200/.245/.290), j = column along y (c0 at y -.045 = robot's right, c2 at y +.045). The object 'cell' is the TARGET cell this episode (its xy = that cell's centre); the policy sees it as a slot. Pieces (p1-p4, each present with p 0.5) sit on other cells: cubes 30-35 mm yawed +-20 deg (so neighbours never touch at 45 mm pitch), discs 31-37 mm; knocking one >1 cm fails (optional not_moved).",
  "cell_m": 0.045,
  "center_xy": [
   0.245,
   0.0
  ],
  "cells": {
   "r0c0": [
    0.2,
    -0.045
   ],
   "r0c1": [
    0.2,
    0.0
   ],
   "r0c2": [
    0.2,
    0.045
   ],
   "r1c0": [
    0.245,
    -0.045
   ],
   "r1c1": [
    0.245,
    0.0
   ],
   "r1c2": [
    0.245,
    0.045
   ],
   "r2c0": [
    0.29,
    -0.045
   ],
   "r2c1": [
    0.29,
    0.0
   ],
   "r2c2": [
    0.29,
    0.045
   ]
  }
 },
 "objects": [
  {
   "id": "cube",
   "role": "manipulated",
   "shape": "box",
   "size_m": [
    0.038,
    0.038,
    0.038
   ],
   "color": "brown",
   "mass_kg": 0.03,
   "friction": 0.6,
   "grasp": "top"
  },
  {
   "id": "cell",
   "role": "target",
   "shape": "plate",
   "size_m": [
    0.045,
    0.045,
    0.001
   ],
   "color": "white",
   "fixed": true,
   "collide": false
  },
  {
   "id": "p1",
   "color": "red",
   "role": "obstacle",
   "shape": "box",
   "size_m": [
    0.034,
    0.034,
    0.034
   ],
   "mass_kg": 0.03,
   "friction": 0.6
  },
  {
   "id": "p2",
   "color": "red",
   "role": "obstacle",
   "shape": "box",
   "size_m": [
    0.034,
    0.034,
    0.034
   ],
   "mass_kg": 0.03,
   "friction": 0.6
  },
  {
   "id": "p3",
   "color": "blue",
   "role": "obstacle",
   "shape": "cylinder",
   "size_m": [
    0.034,
    0.006
   ],
   "mass_kg": 0.008,
   "friction": 0.6
  },
  {
   "id": "p4",
   "color": "blue",
   "role": "obstacle",
   "shape": "cylinder",
   "size_m": [
    0.034,
    0.006
   ],
   "mass_kg": 0.008,
   "friction": 0.6
  }
 ],
 "goal": [
  {
   "pred": "at",
   "a": "cube",
   "xy": "cell",
   "tol_m": 0.012
  },
  {
   "pred": "not_moved",
   "a": "p1",
   "tol_m": 0.01,
   "optional": true
  },
  {
   "pred": "not_moved",
   "a": "p2",
   "tol_m": 0.01,
   "optional": true
  },
  {
   "pred": "not_moved",
   "a": "p3",
   "tol_m": 0.01,
   "optional": true
  },
  {
   "pred": "not_moved",
   "a": "p4",
   "tol_m": 0.01,
   "optional": true
  }
 ],
 "skeleton": [
  {
   "skill": "pick",
   "obj": "cube"
  },
  {
   "skill": "place_at",
   "obj": "cube",
   "xy": "cell"
  },
  {
   "skill": "release"
  },
  {
   "skill": "home"
  }
 ],
 "randomize": {
  "cube": {
   "x": [
    0.2,
    0.26
   ],
   "y": [
    0.13,
    0.17
   ],
   "yaw_deg": [
    -180,
    180
   ],
   "mass_kg": [
    0.02,
    0.05
   ],
   "friction": [
    0.5,
    1.0
   ]
  },
  "cell": {
   "xy_choice": [
    [
     0.2,
     -0.045
    ],
    [
     0.2,
     0.0
    ],
    [
     0.2,
     0.045
    ],
    [
     0.245,
     -0.045
    ],
    [
     0.245,
     0.0
    ],
    [
     0.245,
     0.045
    ],
    [
     0.29,
     -0.045
    ],
    [
     0.29,
     0.0
    ],
    [
     0.29,
     0.045
    ]
   ]
  },
  "p1": {
   "xy_choice": [
    [
     0.2,
     -0.045
    ],
    [
     0.2,
     0.0
    ],
    [
     0.2,
     0.045
    ],
    [
     0.245,
     -0.045
    ],
    [
     0.245,
     0.0
    ],
    [
     0.245,
     0.045
    ],
    [
     0.29,
     -0.045
    ],
    [
     0.29,
     0.0
    ],
    [
     0.29,
     0.045
    ]
   ],
   "p_present": 0.5,
   "yaw_deg": [
    -20,
    20
   ],
   "size_scale": [
    0.88,
    1.03
   ]
  },
  "p2": {
   "xy_choice": [
    [
     0.2,
     -0.045
    ],
    [
     0.2,
     0.0
    ],
    [
     0.2,
     0.045
    ],
    [
     0.245,
     -0.045
    ],
    [
     0.245,
     0.0
    ],
    [
     0.245,
     0.045
    ],
    [
     0.29,
     -0.045
    ],
    [
     0.29,
     0.0
    ],
    [
     0.29,
     0.045
    ]
   ],
   "p_present": 0.5,
   "yaw_deg": [
    -20,
    20
   ],
   "size_scale": [
    0.88,
    1.03
   ]
  },
  "p3": {
   "xy_choice": [
    [
     0.2,
     -0.045
    ],
    [
     0.2,
     0.0
    ],
    [
     0.2,
     0.045
    ],
    [
     0.245,
     -0.045
    ],
    [
     0.245,
     0.0
    ],
    [
     0.245,
     0.045
    ],
    [
     0.29,
     -0.045
    ],
    [
     0.29,
     0.0
    ],
    [
     0.29,
     0.045
    ]
   ],
   "p_present": 0.5,
   "size_scale": [
    0.9,
    1.1
   ]
  },
  "p4": {
   "xy_choice": [
    [
     0.2,
     -0.045
    ],
    [
     0.2,
     0.0
    ],
    [
     0.2,
     0.045
    ],
    [
     0.245,
     -0.045
    ],
    [
     0.245,
     0.0
    ],
    [
     0.245,
     0.045
    ],
    [
     0.29,
     -0.045
    ],
    [
     0.29,
     0.0
    ],
    [
     0.29,
     0.045
    ]
   ],
   "p_present": 0.5,
   "size_scale": [
    0.9,
    1.1
   ]
  },
  "min_sep_m": 0.01
 },
 "limits": {
  "episode_s": 40,
  "max_objects": 6
 }
}
```

## D: `tasks/push_to_target/task.json`

```json
{
 "schema": "argon_task/1",
 "name": "push_to_target",
 "instruction": "push the cube onto the green square",
 "source": {"kind": "hand", "path": null, "model": null},
 "objects": [
  {"id": "cube", "role": "manipulated", "shape": "box", "size_m": [0.038, 0.038, 0.038], "color": "red",
   "mass_kg": 0.03, "friction": 0.5, "grasp": "top"},
  {"id": "target", "role": "target", "shape": "plate", "size_m": [0.08, 0.08, 0.002], "color": "green",
   "fixed": true, "collide": false}
 ],
 "goal": [{"pred": "at", "a": "cube", "xy": "target", "tol_m": 0.03}],
 "skeleton": [{"skill": "push_to", "obj": "cube", "xy": "target"}, {"skill": "home"}],
 "randomize": {"cube": {"r": [0.18, 0.26], "az_deg": [-40, 40], "yaw_deg": [-180, 180], "mass_kg": [0.02, 0.06],
                        "friction": [0.4, 0.8]},
               "target": {"r": [0.18, 0.28], "az_deg": [-45, 45]}, "min_sep_m": 0.0},
 "limits": {"episode_s": 40, "max_objects": 4}
}
```

## D: `tasks/stack_two_cubes/task.json`

```json
{
 "schema": "argon_task/1",
 "name": "stack_two_cubes",
 "instruction": "stack the red cube on the blue cube",
 "source": {"kind": "hand", "path": "deploy bench: two printed 38 mm cubes", "model": null},
 "objects": [
  {"id": "top", "role": "manipulated", "shape": "box", "size_m": [0.038, 0.038, 0.038], "color": "red",
   "mass_kg": 0.03, "friction": 0.6, "grasp": "top"},
  {"id": "base", "role": "target", "shape": "box", "size_m": [0.038, 0.038, 0.038], "color": "blue",
   "mass_kg": 0.03, "friction": 0.6}
 ],
 "goal": [{"pred": "on_top", "a": "top", "b": "base"}],
 "skeleton": [{"skill": "pick", "obj": "top"}, {"skill": "place_on", "obj": "top", "target": "base"},
              {"skill": "release"}, {"skill": "home"}],
 "randomize": {"top": {"r": [0.19, 0.29], "az_deg": [-50, 50], "yaw_deg": [-180, 180], "mass_kg": [0.02, 0.06],
                       "friction": [0.5, 1.0]},
               "base": {"r": [0.19, 0.29], "az_deg": [-50, 50], "yaw_deg": [-180, 180], "mass_kg": [0.02, 0.06],
                        "friction": [0.5, 1.0]}, "min_sep_m": 0.04},
 "limits": {"episode_s": 40, "max_objects": 4}
}
```

## D: `tasks/two_cubes_in_bowl/task.json`

```json
{
 "schema": "argon_task/1",
 "name": "two_cubes_in_bowl",
 "instruction": "put both cubes in the bowl",
 "source": {"kind": "hand", "path": "deploy bench: two printed 38 mm cubes + 110 mm bowl (sim uses a wider bowl so both fit)", "model": null},
 "objects": [
  {"id": "cube_a", "role": "manipulated", "shape": "box", "size_m": [0.038, 0.038, 0.038], "color": "red",
   "mass_kg": 0.03, "friction": 0.6, "grasp": "top"},
  {"id": "cube_b", "role": "manipulated", "shape": "box", "size_m": [0.038, 0.038, 0.038], "color": "brown",
   "mass_kg": 0.03, "friction": 0.6, "grasp": "top"},
  {"id": "bowl", "role": "container", "shape": "bowl", "size_m": [0.14, 0.03], "wall_m": 0.005, "color": "blue",
   "mass_kg": 0.25, "fixed": false}
 ],
 "goal": [{"pred": "inside", "a": "cube_a", "b": "bowl"}, {"pred": "inside", "a": "cube_b", "b": "bowl"}],
 "skeleton": [{"skill": "pick", "obj": "cube_a"}, {"skill": "place_in", "obj": "cube_a", "target": "bowl"},
              {"skill": "release"},
              {"skill": "pick", "obj": "cube_b"}, {"skill": "place_in", "obj": "cube_b", "target": "bowl"},
              {"skill": "release"}, {"skill": "home"}],
 "randomize": {"cube_a": {"r": [0.18, 0.29], "az_deg": [-55, 55], "yaw_deg": [-180, 180], "mass_kg": [0.02, 0.06],
                          "friction": [0.5, 1.0]},
               "cube_b": {"r": [0.18, 0.29], "az_deg": [-55, 55], "yaw_deg": [-180, 180], "mass_kg": [0.02, 0.06],
                          "friction": [0.5, 1.0]},
               "bowl": {"r": [0.21, 0.27], "az_deg": [-40, 40], "size_m": [[0.13, 0.16], 0.03]}, "min_sep_m": 0.03},
 "limits": {"episode_s": 60, "max_objects": 4}
}
```

# Appendix E: export config example (policy/export/d_ens3_search/config.json, fk arrays trimmed)


## E: d_ens3_search

```json
{
 "name": "d_ens3_search",
 "horizon": 15,
 "n_blocks": 3,
 "width": 512,
 "obs_dim": 13,
 "input_dim": 24,
 "fk": {
  "body_pos": [
   [
    0.0388353,
    -8.97657e-09,
    0.0624
   ],
   [
    -0.0303992,
    -0.0182778,
    -0.0542
   ],
   [
    -0.11257,
    -0.028,
    1.73763e-16
   ],
   [
    -0.1349,
    0.0052,
    3.62355e-17
   ],
   [
    5.55112e-17,
    -0.0611,
    0.0181
   ]
  ],
  "body_quat": [
   [
    3.56167e-16,
    1.22818e-15,
    -1.0,
    -4.14635e-16
   ],
   [
    0.5,
    -0.5,
    -0.5,
    -0.5
   ],
   [
    0.7071067811865475,
    -5.986128147598917e-17,
    -2.5805092014641316e-17,
    0.7071067811865475
   ],
   [
    0.7071067811865475,
    9.587217033245734e-16,
    -7.513127675070513e-16,
    -0.7071067811865475
   ],
   [
    0.017209108230571014,
    -0.017209108230571014,
    0.7068973380865913,
    0.7068973380865913
   ]
  ],
  "site_pos": [
   -0.0079,
   -0.000218121,
   -0.0981274
  ],
  "base_z": 0.0024
 },
 "q_lo": [
  -1.9198621771937616,
  -1.7453292519943224,
  -1.69,
  -1.6580628494556928,
  -2.7438472969992493,
  -0.17453297762778586
 ],
 "q_hi": [
  1.9198621771937634,
  1.7453292519943366,
  1.69,
  1.6580627293335335,
  2.841206309382605,
  1.7453291995659765
 ],
 "ensemble_coeff": 0.01,
 "tick_hz": 30,
 "action_space": "joint_delta",
 "train": "<dict, 13534 chars trimmed>",
 "members": [
  "members/1",
  "members/2"
 ],
 "member_sources": [
  "r9_smooth",
  "r11_smooth_s1",
  "r12_smooth_s2"
 ],
 "grasp_search": {
  "closed_below": -0.05,
  "blocked_mm": [
   -12.0,
   0.0
  ],
  "pattern_mm": [
   [
    0.0,
    12.0
   ],
   [
    12.0,
    0.0
   ],
   [
    0.0,
    -12.0
   ]
  ]
 },
 "obs_frame": "mjcf_base_xy"
}
```

## E: pick_place_stick_ens3

```json
{
 "name": "pick_place_stick_ens3",
 "task": "pick_place_stick",
 "task_spec": {
  "schema": "argon_task/1",
  "name": "pick_place_stick",
  "instruction": "pick up the screwdriver and put it on the green square",
  "source": {
   "kind": "hand",
   "path": "real arm Oct 25: wrist-camera screwdriver pick, fingertips ~3 mm above the table, jaws across the handle",
   "model": null
  },
  "objects": [
   {
    "id": "stick",
    "role": "manipulated",
    "shape": "stick",
    "size_m": [
     0.12,
     0.022
    ],
    "color": "yellow",
    "mass_kg": 0.04,
    "friction": 0.7,
    "grasp": "along"
   },
   {
    "id": "zone",
    "role": "target",
    "shape": "plate",
    "size_m": [
     0.1,
     0.1,
     0.002
    ],
    "color": "green",
    "fixed": true,
    "collide": false
   }
  ],
  "goal": [
   {
    "pred": "at",
    "a": "stick",
    "xy": "zone",
    "tol_m": 0.04
   }
  ],
  "skeleton": [
   {
    "skill": "pick",
    "obj": "stick"
   },
   {
    "skill": "place_at",
    "obj": "stick",
    "xy": "zone"
   },
   {
    "skill": "release"
   },
   {
    "skill": "home"
   }
  ],
  "randomize": {
   "stick": {
    "r": [
     0.19,
     0.27
    ],
    "az_deg": [
     -45,
     45
    ],
    "yaw_deg": [
     -180,
     180
    ],
    "size_scale": [
     0.9,
     1.1
    ],
    "mass_kg": [
     0.025,
     0.06
    ],
    "friction": [
     0.5,
     1.0
    ]
   },
   "zone": {
    "r": [
     0.19,
     0.29
    ],
    "az_deg": [
     -50,
     50
    ]
   },
   "min_sep_m": 0.0
  },
  "limits": {
   "episode_s": 40,
   "max_objects": 4
  }
 },
 "obs_layout": "argon_task/1",
 "K": 4,
 "yaw_sym": [
  2,
  4,
  0,
  0
 ],
 "horizon": 15,
 "n_blocks": 3,
 "width": 512,
 "obs_dim": 42,
 "input_dim": 76,
 "fk": {
  "body_pos": [
   [
    0.0388353,
    -8.97657e-09,
    0.0624
   ],
   [
    -0.0303992,
    -0.0182778,
    -0.0542
   ],
   [
    -0.11257,
    -0.028,
    1.73763e-16
   ],
   [
    -0.1349,
    0.0052,
    3.62355e-17
   ],
   [
    5.55112e-17,
    -0.0611,
    0.0181
   ]
  ],
  "body_quat": [
   [
    3.56167e-16,
    1.22818e-15,
    -1.0,
    -4.14635e-16
   ],
   [
    0.5,
    -0.5,
    -0.5,
    -0.5
   ],
   [
    0.7071067811865475,
    -5.986128147598917e-17,
    -2.5805092014641316e-17,
    0.7071067811865475
   ],
   [
    0.7071067811865475,
    9.587217033245734e-16,
    -7.513127675070513e-16,
    -0.7071067811865475
   ],
   [
    0.017209108230571014,
    -0.017209108230571014,
    0.7068973380865913,
    0.7068973380865913
   ]
  ],
  "site_pos": [
   -0.0079,
   -0.000218121,
   -0.0981274
  ],
  "base_z": 0.0024
 },
 "q_lo": [
  -1.9198621771937616,
  -1.7453292519943224,
  -1.69,
  -1.6580628494556928,
  -2.7438472969992493,
  -0.17453297762778586
 ],
 "q_hi": [
  1.9198621771937634,
  1.7453292519943366,
  1.69,
  1.6580627293335335,
  2.841206309382605,
  1.7453291995659765
 ],
 "ensemble_coeff": 0.01,
 "tick_hz": 30,
 "action_space": "joint_delta",
 "obs_frame": "mjcf_base_xy",
 "train": "<dict, 49065 chars trimmed>",
 "members": [
  "members/1",
  "members/2"
 ],
 "member_sources": [
  "pick_place_stick_v1",
  "pick_place_stick_s1",
  "pick_place_stick_s2"
 ],
 "obs_const": {
  "idx": [
   14,
   17,
   18,
   19,
   20,
   21,
   22,
   23,
   24,
   25,
   26,
   27,
   28,
   29,
   30,
   31,
   32,
   33,
   34,
   35,
   36,
   37,
   38,
   39,
   40,
   41
  ],
  "value": [
   1.0,
   0.0010051102144643664,
   0.0,
   1.0,
   0.10032973438501358,
   0.10032973438501358,
   0.002010220428928733,
   1.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0
  ],
  "tol": [
   0.0050000002374872565,
   0.005025577265769243,
   0.0050000002374872565,
   0.0050000002374872565,
   0.00664510705973953,
   0.00664510705973953,
   0.005051154876127839,
   0.0050000002374872565,
   0.0050000002374872565,
   0.0050000002374872565,
   0.0050000002374872565,
   0.0050000002374872565,
   0.0050000002374872565,
   0.0050000002374872565,
   0.0050000002374872565,
   0.0050000002374872565,
   0.0050000002374872565,
   0.0050000002374872565,
   0.0050000002374872565,
   0.0050000002374872565,
   0.0050000002374872565,
   0.0050000002374872565,
   0.0050000002374872565,
   0.0050000002374872565,
   0.0050000002374872565,
   0.0050000002374872565
  ]
 },
 "start_q": [
  -5.4989566396487706e-05,
  -0.0678621588756605,
  -0.28180667934776227,
  1.6580627293335335,
  0.0,
  0.9
 ],
 "eval": {
  "clean": {
   "success": 0.82,
   "n": 100,
   "first": 0,
   "task": "pick_place_stick",
   "file": "policy/out/evals/task_pick_place_stick_pick_place_stick_ens3_clean_100.json"
  },
  "realx0.5": {
   "success": 0.8,
   "n": 100,
   "first": 0,
   "task": "pick_place_stick",
   "file": "policy/out/evals/task_pick_place_stick_pick_place_stick_ens3_realx0.5_100.json"
  }
 }
}
```

# Appendix F: operator guides, findings, pitch (verbatim)


## F: `HARNESS.md`

# HARNESS: one command for the real2sim2real loop

```
python -m argon run --task <name | tasks/<t>/task.json | "an instruction"> [--video <session>] [--rounds N]
                    [--dry-run | --real] [--cpu] [--force <stage> ...]
python -m argon status [task]
```

`argon` is stdlib-only, so any Python 3.10+ can run it. Each stage runs as a subprocess in its component's own venv:
policy/.venv, real2sim/.venv, and the deploy venv. Override any of them with `ARGON_PY_POLICY`, `ARGON_PY_REAL2SIM` or
`ARGON_PY_DEPLOY`. The deploy venv defaults to `deploy/.venv`, then `.venv`, then `~/reachy-duplex/.venv-so101`.
On the Mac mini, set `ARGON_PY_DEPLOY=<the Python 3.12 venv>/bin/python`.

Every stage writes its result to `runs/<task>/manifest.json` (schema `argon_run/1`, written atomically; the dashboard
reads it) and its outputs and log to `runs/<task>/<round>/`. A finished stage is never redone. Rerun the same command
to resume. `--force <stage>` redoes that stage and every stage after it. Dry runs go to `runs/<task>-dry/`, so a
simulated run can never be read as a real one.

## Modes

| mode | flag | attempts | use |
|---|---|---|---|
| dry run | `--dry-run` | synthetic bench (deploy/so101d/synth.py, MuJoCo render, mock servo, ground truth) | rehearsal, CI, no hardware |
| plan | (none) | **printed**: argon stops at each stage that needs the arm and prints the exact bench command; rerun after it | laptop next to the bench |
| real | `--real` | run on the arm by argon (interactive: g to start, y/n to score) | **bench Mac only, owner present** |

## The stages (one status line each)

Each round r1, r2, ... runs these stages:

| # | stage | what | needs | dry-run time (laptop) |
|---|---|---|---|---|
| 1 | task | `tasks/<name>/task.json` (spec/task_spec.md); an instruction in quotes + `--video` goes to real2sim `scripts.task_from_video` (1 Gemini request) | - | 0 s |
| - | session | the attempt this round learns from. r1: `--video`, else a stub probe on the bench. r>1: the previous round's deploy attempt (**closing the loop**) | r1 live: the arm | 14-42 s |
| 2 | twin | real2sim `run_bench` on the session: Gemini render/critique/fix -> `scene.json`; live `r2s_live/1` in `real2sim/out/live/argon_<task>_<r>_<model>/` | Gemini key | ~100 s, 2-4 requests (0 for a synthetic task session) |
| 7 | close | reality (truth / operator y/n / perception) vs twin replay -> focus scene -> `sim_config.json` (loop/close_loop.py) | - | 1 s |
| 3 | data | bot traces in the MuJoCo twin around the focus scenes (loop/gen_traces.py), plus the task's Roblox crowd traces (`roblox/out/traces`, `meta.task`) | - | ~35 s (200 traces, 4 workers) |
| 4 | train | `policy/train.py` (r9_smooth recipe) on crowd traces + every round's twin traces, under a landlord lease | lease (or `--cpu`) | ~60 s on GPU |
| 5 | eval | twin eval of the new export and of the incumbent on this round's focus scenes, clean + realx0.5 (`argon/sim_eval.py`) | - | ~2 min (n=50 x 2 x 2) |
| 5 | gate | the new export replaces the incumbent only if mean(clean, realx0.5) is higher; r1's incumbent is the headline `d_ens3_search` | - | 0 s |
| 6 | deploy | the exact bench command for the gated export. Dry run: argon runs it on the synthetic bench. `--real`: argon runs it on the arm | arm (real) | 44 s (learned policies run to the 40 s timeout) |

**One round takes about 5 minutes.** Most of it is Gemini latency, the learned attempt and the eval.

Measured on Oct 25, 03:00-04:40 (laptop, RTX 5080 under a landlord lease, synthetic bench; `runs/*-dry/manifest.json`):

| run | round | session | twin | close | data | train | eval | deploy | total |
|---|---|---|---|---|---|---|---|---|---|
| cube_in_bowl | r1 | 14 s | 109 s (4 req; 29 s on the quota-capped rerun) | 0 s | 38 s | 62 s | 109 s | 43 s | ~5-6 min |
| cube_in_bowl | r2 | (r1's attempt) | 312 s (3 req, 5 Gemini retries) | 0 s | 34 s | 63 s | 158 s | 48 s | 10 min |
| pick_place_stick | r1 / r2 | 9 / 12 s | 6 / 7 s (0 req) | 0 s | 53 / 71 s | 111 / 138 s | 197 / 159 s | emitted | 6.3 / 6.4 min |
| stack_two_cubes | r1 / r2 | 11 / 12 s | 6 / 6 s (0 req) | 0 s | 133 / 79 s | 97 / 109 s | 182 / 162 s | emitted | 7.1 / 6.1 min |
| cube_in_bowl, cache-only task path (05:00, laptop shared) | r1 / r2 | 16 s / (r1's attempt) | 6 / 16 s (0 req) | 0 s | 80 / 92 s | 195 / 174 s | ~160 / 127 s | 47 / 46 s | ~16 min |
| pick_place_stick, cache-only, deploy RUN on the any-task bench (05:10, laptop shared, --workers 2) | r1 / r2 | 51 s / (r1's attempt) | 6 / 6 s (0 req) | 0 s | 101 / 107 s | 116 / 191 s | 257 / 235 s | 87 / 93 s | 10.3 / 10.5 min |

The gate results show the loop working. cube_in_bowl HOLDs both rounds: the retrained single member, at 0.80/0.60 and
then 0.62/0.52, stays below the headline d_ens3_search's 0.86/0.70 and then 0.66/0.54 on the same focus scenes. On
pick_place_stick, r2 scores 0.84/0.80 against r1's 0.12/0.04 on r2's scenes and PASSes. On stack_two_cubes, r1 scores
0.96/0.70 against stack_two_cubes_v1's 0.32/0.26 (PASS), and r2 scores 0.94/0.88 against r1's 0.50/0.32 (PASS).
All of these numbers are twin evals of synthetic sessions: n=50 per setting, clean / realx0.5.

## GPU: the train stage needs a landlord lease

A script cannot call the landlord MCP. When train has no lease, argon stops with BLOCKED and prints the lease step:

1. `landlord_overview()`. Check that squaredcube (the laptop, RTX 5080) is `trust: live` and that `unleased_used_gb` is 0.
2. `rent_apartment(pct=60, vram_gb=4, minutes=30, building="squaredcube", purpose="argon <task> <round>: train.py ...")`
3. Save the result as `runs/lease.json`, either the whole result or `{"lease_id": "...", "env": {...}}`. You can
   also pass `--lease <file>` or export the env vars.
4. Rerun the same command. Before every launch, argon checks that the lease is active in the ledger
   (`http://100.112.46.57:7788/api/state`) and refuses an expired one. train.py applies the VRAM cap itself.
5. `move_out(lease_id, outcome)` when you are done.

No GPU at all: `--cpu` (~10x slower; `--steps 2000` for a smoke run).

## Recipes

```
# rehearsal, no hardware, 2 rounds, ZERO Gemini requests (cache-only; cube_in_bowl runs on the task path)
python -m argon run --task cube_in_bowl --dry-run --rounds 2

# at the bench (Mac mini), owner present: probe -> twin -> data -> train -> gated deploy -> next round
export ARGON_PY_DEPLOY=~/argon-venv/bin/python     # Gemini roles default to main gemini-4-argon, small gemini-3.8-flash
python -m argon run --task cube_in_bowl --real --rounds 2

# laptop drives, bench runs the arm: argon prints each bench command and stops; rerun after the attempt is recorded
python -m argon run --task cube_in_bowl --rounds 2

# start from a recorded session (record_session.py or run_policy --record), e.g. the owner's video
python -m argon run --task cube_in_bowl --video deploy/runs/<ts>/attempt_01

python -m argon status            # where every task is: stage, twin gap, motion match, eval, gate, attempts
```

The bench command argon emits (gated export, recorded as the next round's session):
`$PY deploy/scripts/run_policy.py --policy <export> --attempts 1 --record --out runs/<task>/<r>/attempt_real --task "..."`

## Any task, not just cube_in_bowl

Pass `--task <name>` for any `tasks/<name>/task.json`, for example pick_place_stick, push_to_target,
two_cubes_in_bowl or stack_two_cubes. These tasks run on the generic path:

- **session**: `--video`. In a dry run, real2sim `synth_task_bench` instead renders a scripted attempt, with objects
  placed from the task's own distribution.
- **twin**: `run_bench --task-json`.
- **close**: writes `task_focus.json`. This is the task with `randomize` narrowed to +-2 cm and +-20 deg around the
  twin's object poses.
- **data**: `policy/task_traces.py` (generic expert): half from the prior, half from task_focus.json.
- **train**: `policy/train_task.py`.
- **eval**: `policy/task_eval.py` on task_focus.json, clean + realx0.5.
- **deploy**: `run_policy.py --task-json tasks/<name>/task.json` (task-general perception).

Since 56b8db4, the synthetic bench in run_policy renders any task (`--dry-run --task-json`; the stub is
`TaskScriptedPolicy`), so a dry run of any task runs its probe and its gated export on the synthetic bench, and the
next round learns from that attempt, exactly as on the arm. If an older checkout lacks that bench, argon falls back to
emitting the command and taking a scripted `synth_task_bench` attempt, labelled as one.

**Zero-Gemini dry runs.** For a synthetic task session, `--twin-gemini auto` (the default) fits the twin from oracle
boxes: synthetic truth projected through the true camera (`argon/oracle_boxes.py`), plus silhouettes and anchors, at
iteration 0 with no critique loop. It makes 0 requests and takes ~5 s. Use `--twin-gemini on` for the full Gemini loop.
Real sessions always use Gemini. The legacy cube_in_bowl twin always uses Gemini (detection + critiques, 2-4
requests), which is why a cache-only dry run of cube_in_bowl takes the task path (`--backend auto`). On both paths the cube incumbent is d_ens3_search (task_eval runs it on task scenes); other tasks start against `<task>_ens<N>`, else `<task>_v<N>`. Force the legacy path with
`--backend cube_in_bowl` (and `--no-cache-only`).

## Gemini: roles, cache-only, quota

real2sim and deploy use two roles (r2s/gemini.py): **main** (task from video, twin critique/fix; default
`gemini-4-argon`, `--main-model` / `R2S_MODEL_MAIN`) and **small** (boxes, pointing, JSON repair; default
`gemini-3.8-flash`, `--small-model` / `R2S_MODEL_SMALL`). argon passes both to every subprocess and the twin line names
the models that answered (`[Gemini: <model> <requests>+<cache hits>c ...]`). A 429 on main stops the stage with
BLOCKED (role, model, reset time) unless `--allow-fallback` (main -> small). **Dry runs are cache-only by default**
(`R2S_CACHE_ONLY=1`: zero new requests; a cache miss is BLOCKED, never a silent call); `--no-cache-only` spends quota
on purpose. Free tier: 20 requests/day/model, shared by every agent and the bench.

## Knobs

`--rounds N`, `--traces 200`, `--eval-n` (100; dry run 50), `--steps 10000`, `--workers 4`, `--main-model` /
`--small-model` / `--allow-fallback` / `--cache-only` / `--no-cache-only` (above), `--backend auto|cube_in_bowl|generic`,
`--twin-iters`/`--twin-calls` (dry run 2/4, else 4/7), `--gemini-budget 20` (uncached requests per run dir; the twin
stops with BLOCKED when it runs out), `--twin-gemini auto|on|off`, `--device cuda|cpu|mps` (mps = the Mac mini's GPU; no lease
needed), `--deploy-arg=--allow-unverified` (repeatable; passed to every run_policy.py attempt), `--seconds` (attempt
length), `-v` (echo every subprocess).

## Troubleshooting

- **BLOCKED: train needs a landlord GPU lease.** Do the lease step above, or use `--cpu`.
- **BLOCKED: r2 learns from r1's attempt on the arm.** Plan mode is waiting for the bench. Run the printed command on
  the bench and copy `runs/<task>/<r>/attempt_real/` back. Then rerun.
- **twin FAILED.** Read `runs/<task>/<r>/twin.log`. A daily-quota 429 stops real2sim cleanly (its best scene is
  kept). Responses are cached in `real2sim/out/cache/`, so a rerun costs nothing.
- **Stub probe "timeout" with `stale_cube_ticks` = every tick.** Perception never saw the cube: the bench.json HSV
  does not match the cube's colour. Live, run `deploy/scripts/perceive.py --tune`. On the synthetic bench, synth.py's
  cube colour must match bench.json.
- **The twin replay "DISAGREES (nan px)".** The attempt had no grasp, so there is no physics replay. Friction and
  mass then come from the prior (close_report.json says why).
- **gate HOLD.** The new export was not better on the focus scenes, so the incumbent is deployed. This is correct
  behaviour, not an error.
- **The bench Mac has no `policy/export/argon_<task>_<r>`.** Exports are not committed (policy/.gitignore keeps
  `export/*` local). Before you run a bench command on the Mac, copy the gated export there
  (`scp -r policy/export/<name> mac:argon-hack/policy/export/`), or commit it on purpose with `git add -f`
  (about 13 MB each).
- **BLOCKED: Gemini <role> model <m> is out of quota (429), resets in ...** Pass `--main-model` / `--small-model` with
  a model that has quota, or `--allow-fallback`. A 429 inside the critique loop instead keeps the best scene so far and
  the line ends with `[Gemini quota hit ...]`.
- **BLOCKED: cache-only and no cached Gemini answer for <tag>.** The dry run would have spent a request. Use a synthetic
  task session (oracle twin, 0 requests) or `--no-cache-only`.
- **Gravity compensation is ON by default.** `deploy/config/bench.json` sets `control.gravity_comp_k = 0.05`, and dry
  runs and the bench share it, so no extra `--deploy-arg` is needed. At the bench, refit k with
  `deploy/scripts/fit_sag.py` and edit bench.json. Without comp, the synthetic bench (which models the real arm's sag
  since 3be8dc1) ends attempts `held_after_refusals` within ~5 s. Measured 05:40 on the same exports and seeds:
  0/4 -> 3/4 goal_reached with k = 0.05 (cube r1 / r2, stick r2).
- **Out of memory on the laptop.** Run one argon at a time. Every stage is a child process that exits when it is done.
- **Logs.** Each stage writes `runs/<task>/<r>/<stage>.log`. The manifest records each stage's wall time, outputs
  and notes.

## F: `LOOP.md`

# The autonomous real2sim2real loop (cube into bowl, SO-101)

No human demonstration anywhere. The arm makes its own data. The twin is rebuilt from what the arm saw, and the
policy is retrained on bot traces in that twin.

```
 (1) real attempt ──record──> session ──(2) real2sim (Gemini render / critique / fix)──> twin + replay check
        ^                                                                                     │
        │                                                                     (3) close_loop: sim config, bot traces
        └──────────── (4) redeploy <── retrain (GPU, landlord lease) <────────────────────────┘
```

`PY=/c/Users/sarta/reachy-duplex/.venv-so101/Scripts/python.exe` (deploy). Run from the repo root.

| step | command | measured (laptop, dry run) |
|---|---|---|
| 1. probe / attempt | `$PY deploy/scripts/run_policy.py --policy stub --attempts 1 --record` (`--dry-run` here, live on the Mac) | 14 s (stub), 44 s (learned: runs to the 40 s timeout) |
| 2. real2sim | `cd real2sim && PYTHONPATH=. R2S_MODEL=gemini-3.5-flash .venv/Scripts/python -m scripts.run_bench ../deploy/runs/<ts>/attempt_01 --iters 4 --max-calls 7` | 102 s, 5 requests + 1 cache hit |
| 3. close + traces | `policy/.venv/Scripts/python loop/close_loop.py --session deploy/runs/<ts>/attempt_01 --r2s real2sim/out/live/<run> [--prev loop/rounds/<r>/sim_config.json]` | 29 s (200 traces, 4 workers, 0.55 s each) |
| 4a. retrain | printed by close_loop: `cd policy && .venv/Scripts/python train.py --name loop_<r> --traces ../roblox/out/traces ../loop/rounds/<r>/traces --steps 10000 --pos-sigma 0 --offset-deg 2 --aim-mm 10 4 --miss-weight 4 --chatter smooth --seed 0 --device cuda` with the lease env | ~65 s (41 s of training, RTX 5080 laptop) |
| 4b. check in the twin | `policy/.venv/Scripts/python loop/eval_focus.py --config loop/rounds/<r>/sim_config.json --policy policy/export/loop_<r> --n 100` | 43 s |
| 4c. redeploy | `$PY deploy/scripts/run_policy.py --policy loop_<r> --attempts 1 --record`, which is step 1 of the next round | as step 1 |

**One full round: about 5 minutes** (≈ 14-44 + 102 + 29 + 65 + 43 s). Most of it is Gemini latency, plus the attempt
itself on a learned policy. GPU: `landlord_overview`, then `rent_apartment(pct=60, vram_gb=4, minutes=30,
building="squaredcube", purpose=...)`. Launch train.py with the returned env (train.py refuses cuda without
`LANDLORD_VRAM_GB` and applies the VRAM cap itself), `attach_process` the printed pid, `move_out` when it is done.

## What each step does

1. **`run_policy.py --record`** writes each attempt as `deploy/runs/<ts>/attempt_NN/` in record_session's layout:
   `frames/NNNN.jpg` and `joints.jsonl` (t, frame, owner_deg, **cmd_owner_deg** = the guarded command sent that tick,
   plus phase and the perceived cube and bowl). It also writes `meta.json` (the joint map inline, fps, policy, outcome,
   bench perception sizes) and `camera.json` (ArUco). On the synthetic bench it adds `camera_true.json` and `truth.json`.
   Frames go through a bounded queue to a writer thread, so the 30 Hz tick never waits on disk. Dry-run tick time
   with and without `--record`: p50 22-24 ms, p90 31-33 ms either way (the MuJoCo render dominates). The writer
   takes 1.1 ms per frame, and 0 frames were dropped. In the dry run the jaws now stop at the cube's width while
   holding, as real jaws do, so the anchors see a real hold.
2. **real2sim** reads the attempt directly. The gripper commands make the grasp/release anchors the robot's own. The
   inline joint map wins over the config file's.
3. **`loop/close_loop.py`** compares the twin with reality. Reality means truth.json on the synthetic bench; live,
   the operator's y/n, else perception. It reports: did the twin's physics replay end in the bowl where reality did,
   and how far off it ended. It reports the twin's errors against what the robot believed and, on the synthetic
   bench, against ground truth. It then appends a **focus scene** to the sim config (`loop/rounds/<r>/sim_config.json`,
   with a "why" for every parameter):
   - cube and bowl xy and the bowl diameter come from perception;
   - the bowl height comes from bench.json;
   - the cube size comes from the twin (its jaw-gap anchor);
   - friction and mass come from the twin only if its replay agrees with reality, else from the prior;
   - weight 2 if the attempt failed.

   `loop/gen_traces.py` then runs the scripted bot (policy/scripted.py) in the MuJoCo twin (policy/sim.py). Half the
   scenes are drawn around the focus scenes, half from the Roblox prior. It writes the Roblox trace format
   (spec/policy_io.md), so train.py mixes these traces with `roblox/out/traces`.
4. **Retrain + check**: the same recipe as r9_smooth (one member of the headline d_ens3_search). Only the data
   changes. eval_focus runs the export in the twin on held-out focus scenes. This is the gate before redeploying.

## Results so far (all synthetic: dry-run bench, no hardware)

Round r1 (twin `real2sim/out/live/loop_dry_a01_g35flash`, from a stub attempt that succeeded):
- twin vs truth: cube 0.98 cm, bowl 2.8 cm, camera 1.1 cm, cube size 0.0 mm (anchor), bowl diameter +22 mm,
  bowl height 7 cm (true 5). Perception vs truth: cube 0.1 cm, bowl 0.14 cm. That is why positions come from perception.
- replay: the twin grasped and lifted (8.6 cm) but did **not** end in the bowl (9.7 cm off where reality left the
  cube; trajectory error 31 px). It disagrees with reality, so the traces use the prior friction and mass.
- 200 traces, 156 successful bot episodes; loop_r1 trained in 41 s.
- twin eval (real noise, held-out seeds): **focus scenes 0.08 (r9_smooth) -> 0.19 (loop_r1)**, n = 100 each;
  prior scenes 0.43 -> 0.43, n = 200, no regression. The focus scenes (25 mm cube, 5 cm bowl, this bench's layout)
  are much harder than the prior for the old policy. Most failures are still "never lifted".
- redeployed loop_r1 (dry run): cube in the bowl per truth; the policy runs to the 40 s timeout (learned exports never
  signal done).

Round r2 (twin `loop_dry_r1_a01_g35flash`, from loop_r1's own attempt, which succeeded):
- twin vs truth: cube 0.64 cm, bowl 2.6 cm, bowl diameter +2 mm, cube size 0.0 mm; IoU cube .71, bowl .69.
- replay: the twin **never grasped** (lift 0, 21.6 cm off, 90 px). It disagrees again, so friction and mass come from
  the prior. This is real2sim's known knife-edge sim grasp (FINDINGS.md).
- 200 traces (155 ok); loop_r2 trained on Roblox + r1 + r2 traces in 42 s.
- twin eval on r2's focus scenes (both attempts, n = 100, real noise): **r9_smooth 0.10, loop_r1 0.17, loop_r2 0.26**;
  d_ens3_search (the headline: 3 seeds + grasp search) 0.16. Prior scenes: loop_r2 0.42 (r9_smooth 0.43, n = 200).
- round wall time: attempt 44 s + real2sim 99 s + close/traces 28 s + retrain ~60 s + eval 42 s = **4.5 min**.

## Not automatic yet (honest list)

- **One command per step, not one command per round.** Nothing chains (1)-(4), and live attempts need the owner at
  the bench (g to start, y/n to score). The steps are run by hand or by an agent.
- **The twin's physics is the weak link.** The replay disagreed with reality in round 1, so friction and mass fall back
  to the prior. The dry-run cube is kinematic (it follows the closed jaws exactly), so a physics replay cannot fully
  match it. On the real arm this check is meaningful. real2sim fits one attempt at a time; it does not yet fit several
  attempts jointly.
- **Gemini's bowl is off by 2-3 cm and 2 cm in diameter**, so close_loop takes positions and bowl size from perception
  and only physical parameters from the twin.
- **The perception bias** (perceived minus twin) is reported in sim_config.json but not injected into the traces
  (train.py's `--aim-mm` already relabels the grasp aim; doing both would double-count).
- **The retrain is from scratch** (no fine-tuning) and is a single member, not the 3-seed ensemble with grasp search
  that d_ens3_search is. Redeploy decisions use eval_focus; nothing compares against the headline export automatically.
- **Bot success in the twin is about 75%** (grasps are knife-edge in MuJoCo); failed bot episodes are written and dropped by train.py.
- Live recording is untested on the real arm (this laptop has no hardware). The writer thread and the queue are the
  same code; LiveCamera.grab returns a copy.

## F: `deploy/RUNBOOK.txt`

```
SO-101 BENCH RUNBOOK: cube into bowl (argon-hack/deploy). Owner present for every step that touches the arm.
All commands run from C:\Users\sarta\argon-hack in Git Bash.  PY=/c/Users/sarta/reachy-duplex/.venv-so101/Scripts/python.exe
(LeRobot 0.6.1). The Gemini-only step uses real2sim/.venv. Practice first with no hardware:
  $PY deploy/scripts/run_policy.py --dry-run --policy stub --attempts 3    (synthetic bench, should print truth_in_bowl true)
  $PY deploy/tests/test_stack.py                                         (guard / roll / torque-path checks, all "ok")

0. BEFORE PLUGGING IN  (10 min)
 - Print deploy/runs/aruco_board_letter.png ($PY deploy/scripts/make_board.py) at 100 % / actual size.
   Measure the marker row width (outer edge of marker 0 to outer edge of marker 3) and set
   bench.json board.marker_length_m = width / 4.75 and separation_m = marker_length_m / 4.
 - Ruler: cube edge -> perception.cube.size_m; bowl rim height -> bowl.rim_height_m; bowl wall -> bowl.wall_m.
 - Mount the camera HIGH and fixed: >= 40 deg down onto the table 15-35 cm in front of the base, with base, bowl
   and cube in view. Set camera.index (try 0/1) in bench.json. Do not touch the camera after step 2.
 - The calibration file is C:/Users/sarta/ego16-storage/lerobot_calibration/robots/so_follower/follower_1.json
   (~/.cache/... does not exist on this machine). NEVER run lerobot-calibrate during the session: every offset depends on it.
 - Close anything holding COM3 or the camera (old python processes).

1. PLUG IN: arm power + USB. $PY -c "import serial.tools.list_ports as l; print([p.device for p in l.comports()])" ->
   set robot.port. The arm stays LIMP until step 5 (steps 2-4 only read).

2. CAMERA (no Gemini; arm limp; 10 min)
   $PY deploy/scripts/calib_camera.py --capture
   - i x 8-12: hold the board TILTED 20-45 deg at different places/distances (intrinsics).
   - Place the board flat, top marker row toward the robot, parallel to the base front face. Measure:
     dist_from_base_front_m (base front face -> top edge of marker 0), dist_from_base_front_right_m (same at marker 3),
     left_edge_y_m (robot-y of marker 0's left edge; robot's right is negative). Enter them in bench.json, then
     p (placed view) and c. Re-run offline after edits: $PY deploy/scripts/calib_camera.py --from deploy/runs/<ts>_calib
   PASS: board reprojection < 1.5 px, elevation >= 40 deg; in overlay.jpg the base footprint (cyan) sits on the base.
   Optional Gemini cross-check (one request per frame; 20/day/model on the free key): after step 3b,
   R2S_MODEL=gemini-4-argon real2sim/.venv/Scripts/python.exe deploy/scripts/calib_from_arm.py deploy/runs/<ts>_session
   -> held-out keypoint error and camera distance vs the ArUco camera (expect < 4 cm).

3. JOINT OFFSETS: verify, never assume (torque OFF, read-only; 15 min)
 a) Live mirror: roblox\.venv\Scripts\python roblox\tools\joint_server.py --source leader --port 8767 (own console;
    in Studio set Workspace TwinServer = http://127.0.0.1:8767, TwinDrive = live), then
    $PY deploy/scripts/mirror.py --overlay
    Move the limp arm: folded, upright, gripper level, pan +-45, jaw open/closed. The twin must match the real pose.
    In the overlay, note which candidate's outline hugs the arm, ESPECIALLY the gripper orientation (the two
    candidates differ by 88 deg on wrist_roll). q quits.
 b) $PY deploy/scripts/record_session.py --seconds 25   (move the limp arm through the workspace; the jaw near the
    table once), then $PY deploy/scripts/verify_offsets.py deploy/runs/<ts>_session --fit
    -> mean IoU per candidate (synthetic test: right 0.52 vs wrong 0.42) and offsets_check.jpg. Set "active" in
    deploy/config/joint_map.json to the winner (this settles wrist_roll).
 c) $PY deploy/scripts/read_pose.py    then type:  folded / closed / open / touch <x> <y> (6-8 spots) / fit / done
    touch = FIXED-jaw tip on the table at x = 0.048 + ruler distance from the base front face, y = left(+)/right(-) of
    the centreline; spread x 0.15-0.32, y -0.15..0.15, and vary the wrist (steep and shallow approach).
    fit -> per-candidate tip error, a "bench_fit" candidate (pan/lift/elbow/wrist_flex, roll from the active one) and
    a table plane. PASS: tip rms <= 5 mm. Then set active = bench_fit, guard.plane_z = the printed value, and
    "verified": true on that candidate. Until then every live run needs --allow-unverified and runs at <= 0.3 speed.

4. PERCEPTION  (5 min)  $PY deploy/scripts/perceive.py   (click to print HSV; adjust perception.*.hsv; s saves a frame)
   Put the cube at 3 ruler spots and the bowl at one; the yellow cross must sit on the cube and the yellow ring on
   the rim. On a saved frame: $PY deploy/scripts/perceive.py --image <frame> --truth cx cy bx by bowl_d
   Expected (synthetic Monte Carlo, ruler sigma 2 mm, yaw 1 deg): cube 4 mm median / 7 mm p90, bowl centre 3-4 / 8 mm,
   bowl diameter 1-2 mm. PASS: <= 8 mm.

4b. TASK-GENERAL PERCEPTION (any tasks/<name>/task.json; no HSV ranges)  (2 min, 1 Gemini request)
   GEMINI_API_KEY in the env (or GEMINI_ENV_FILE=<.env>); R2S_MODEL_SMALL (default gemini-3.8-flash) boxes the objects at setup; R2S_MODEL_MAIN (default gemini-4-argon) re-detects lost objects.
   $PY deploy/scripts/perceive.py --task tasks/cube_in_bowl/task.json --setup-out deploy/runs/setup_<task>.json
   One request boxes every task object on the first frame; GrabCut + a per-object HSV histogram model (fitted on that
   frame against the rest of the scene) then track at 30 Hz (5.4 ms/frame for 3 objects). Outlines: each object's
   fitted 3D silhouette; thick = seen this frame. Lost > 1 s (not in the jaws) -> one asynchronous re-detect
   (rate-limited 1 / 8 s, at most 6 per run: the free tier is ~20 requests/day/model). Sticks whose colour model covers one end (screwdriver handle) are tracked from
   that end + the box. Policy runs: run_policy.py ... --task-json tasks/<name>/task.json
   [--perception-setup deploy/runs/setup_<task>.json]  -> per attempt "auto_success" = argon_tasks.check on the
   perceived final state; the operator's y/n overrides it ("success", "success_source").
   Synthetic accuracy (deploy/scripts/perception_task_eval.py, oracle boxes, 6 scenes x 5 poses): cube 1.3 mm /
   1.4 deg, bowl 1.2 mm (inner d +0.3 mm), screwdriver 1.6 mm (p90 5 mm) / 0.6 deg, 30/30 seen, 5.7 ms/frame.
   With Gemini boxes (gemini-3-flash-preview) + a perturbed ArUco camera: cube 2.4 mm / 3.1 deg, bowl 3.9 mm,
   screwdriver 1.7 mm / 2.1 deg. Tests: $PY deploy/tests/test_taskperception.py

4c. GEMINI QUOTA (free key, ~20 requests/day/model; on Oct 25 gemini-4-argon has no billing, 3.8-flash, 3.7-flash and
   3.5-flash are spent until midnight PT, gemini-3-flash-preview has a few left). The bench needs ONE request in total:
     export R2S_MODEL_SMALL=gemini-3-flash-preview      # owner's fallback for the small role (default gemini-3.8-flash)
     # 1. ONE request: box the task objects on the real camera (objects placed, arm at the start pose, out of view)
     $PY deploy/scripts/perceive.py --task tasks/cube_in_bowl/task.json --setup-out deploy/runs/setup_cube_in_bowl.json
        -> check the outlines sit on every object (q quits); the setup frame is saved next to it. If an outline is off,
        move the object a little and rerun (the answer for an identical frame is cached and costs nothing).
     # 2. ZERO requests from here on: every run replays that setup, re-detection is off (--redetects 0 is the default
     #    with --perception-setup), and R2S_CACHE_ONLY=1 makes any request that would leave the machine fail instead
     export R2S_CACHE_ONLY=1
     $PY deploy/scripts/run_policy.py --policy <name> --attempts 1 --speed 0.3 --task-json tasks/cube_in_bowl/task.json          --perception-setup deploy/runs/setup_cube_in_bowl.json
     $PY deploy/scripts/perceive.py --task tasks/cube_in_bowl/task.json --setup-in deploy/runs/setup_cube_in_bowl.json
   Proof (synthetic frame, Oct 25): with R2S_CACHE_ONLY=1, step 1 (cache hit) and a run_policy dry run on that frame
   with --perception-setup left deploy/runs/_gemini_cache/calls.jsonl unchanged (0 requests), meta max_redetects 0,
   0 stale ticks. A setup is valid while the camera and the lighting stay put; redo step 1 after either changes.
   No quota at all: run_policy without --task-json uses the bench.json HSV perception (cube_in_bowl only).

4d. SAG + LEARNED EXPORTS (read before any learned policy on the arm)
   The exports output joint DELTAS on the q they see. The real arm sits 5-12 mm below its command (gravity sag); a
   policy that sees the MEASURED q re-adds the sag every tick and walks into the table (guard holds it). Dry runs,
   seed 11, real sag: d_ens3_search 0/12 measured q; 10/12 with gravity comp k .05; cube_in_bowl_v1 2/10 -> 7/10
   with comp (no sag: 9/12, 8/10). Two fixes, combinable:
     --policy-q commanded   the policy sees its last command (no calibration; resets to measured if > 4 deg off)
     --gravity-comp K       command + K * gravity torque (<= 3 deg/joint, before the guard); K from:
        $PY deploy/scripts/armd.py            (owner present)   then   $PY deploy/scripts/fit_sag.py
        -> prints K (8 small armd moves, ends at rest); or set control.gravity_comp_k in bench.darwin.json
   Defaults in code: policy_q COMMANDED (dry runs, real sag: d_ens3 10/12, cube_in_bowl_v1 7/10, same as comp;
   stick_v1 + comp 7/10), gravity comp off (bench.json control.policy_q / gravity_comp_k).

4e. TIC-TAC-TOE (Argon vs the human)
   Print deploy/assets/ttt_grid_letter.png at 100 % (check the 100 mm bar); tape it with the marked corner at base
   x 0.1775, y -0.0675, the arrow toward the robot. Robot = brown cubes from the supply mark x 0.23, y 0.15; human =
   blue discs. One Gemini MAIN request per robot turn (R2S_MODEL_MAIN, default gemini-4-argon; R2S_ALLOW_FALLBACK=1
   allows the small model if Argon has no quota); a misread is re-asked once, then local minimax (logged).
     $PY deploy/scripts/tictactoe.py --first robot --policy stub --policy-q commanded --gravity-comp 0.05 \
         --publish-port 8790 [--allow-unverified --speed 0.3]
   Dry run (no hardware, no Gemini): $PY deploy/scripts/tictactoe.py --dry-run --first robot
   One board read on a saved frame: $PY deploy/scripts/tictactoe.py --frame-only frame.jpg --history r1c1:X

5. FIRST POLICY RUN, REDUCED SPEED  (owner's hand near the keyboard)
   Optional mirror: joint_server on 8767 as in 3a, then add --mirror.
   $PY deploy/scripts/run_policy.py --policy stub --attempts 1 --speed 0.3 [--allow-unverified] [--mirror]
   - Torque-on: the OWNER SUPPORTS THE WRIST, torque comes on holding the present pose, the arm moves slowly to the
     sag-safe start pose (owner frame lift -78 / wrist_flex 86; FK >= 105 mm clear under both joint maps).
   - Place cube + bowl, press g.   s or SPACE = stop (hold, slow retreat to start, hold)   p = pause   q = quit.
     Agent supervision: echo '{"seq": 1, "cmd": "stop"}' > deploy/runs/control.json   (increase seq per command)
   - A guard refusal HOLDS (torque on). 60 refused ticks in a row end the attempt as held. Nothing drops torque.
   - Then the trained policy: --policy <name under policy/export/> --speed 0.3, then 0.6, then 1.0 after two clean runs.

6. SCORING, 10 TRIALS
   $PY deploy/scripts/run_policy.py --policy <name> --attempts 10 --speed <s>
   Spots (base frame, metres): cube x in {0.18, 0.24, 0.30} x y in {-0.10, -0.04}; bowl (0.22, +0.11) for trials 1-5,
   (0.28, +0.06) for 6-10. After each attempt: y = cube in bowl, n = not. Note the failure class in the session notes:
   no grasp / dropped / missed bowl / guard hold / load stop. Results: deploy/runs/<ts>_live/{summary.json, ticks.jsonl}.

7. END: the arm holds at the start pose. The owner supports the wrist, presses T twice -> torque off (or q to leave it
   holding). Stop joint_server. Commit any changed config/*.json with explicit paths.
```

## F: `DEMO_DAY.md`

# Demo day, Sun Oct 25 (SHACK15, judging from 9 AM)

Bench = Mac mini `ssh squaredcube@100.123.146.17`, repo `~/argon/repo`, `PY=~/argon/.venv/bin/python`, run from the repo.
Detail for every bench step: deploy/RUNBOOK.txt. Whole loop: HARNESS.md. Pitch: PITCH.md.

## Before leaving (owner, ~8:25-8:40)
- [ ] Print done (~08:25): take the plate as-is. Board (white/black, arrow toward the robot), 2 brown 38 mm cubes, blue
      bowl 110 mm. Matte, no ironing: check the markers are crisp.
- [ ] Gemini billing or hackathon credits on the key. Without it there is NO gemini-4-argon and NO gemini-3.8-flash today
      (free quota spent; only gemini-3-flash-preview has a few calls). Test: one tiny request to gemini-4-argon.
- [ ] Bring: follower arm + 12 V brick, USB hub, wrist cam (on the claw), scene camera + phone tripod/clamp, Mac mini +
      its power, ethernet/hotspot, the screwdriver. Leave the leader (7.4 V servos, only a 12 V supply).

## Roblox publish (owner, 10 min; click-by-click at the top of roblox/FINDINGS.md)
- [ ] Trace server + tunnel: `roblox\.venv\Scripts\python roblox\tools\joint_server.py --port 8770 --key <secret>`, then
      cloudflared quick tunnel to it; same key in TwinConfig.TRACE_KEY.
- [ ] Studio: File > Publish to Roblox under YOUR account (the meshes were uploaded by it). Game Settings: Allow HTTP
      Requests + Studio API ON, Phone/Tablet ON, Public, Max Players 8. Publish again.
- [ ] In game chat: `/traceurl https://<tunnel>`. Play the tutorial once on a real phone (never tested on a device).
- [ ] dashboard/config.json: `game_url` = https://www.roblox.com/games/<PlaceId> (QR uses it).
- [ ] Move tonight's test traces out of the counters (Claude does this: roblox/out/traces/human_* -> roblox/out/dev_traces/).

## Bench, in order (Claude runs the commands over ssh; owner present for anything that moves)
1. [ ] Power + USB. `$PY deploy/scripts/preflight.py --tests` -> 0 FAIL (follower EEPROM = follower_1.json, 12 V).
2. [ ] Scene camera mounted HIGH (>= 40 deg down), plugged into the Mac. Start `~/argon/camserver.command` (Terminal;
       Allow camera). Pick its index from `curl 127.0.0.1:8792/list`; set bench.darwin.json camera.index. Never move it after 3.
3. [ ] Camera calibration with the printed board (RUNBOOK 2): `$PY deploy/scripts/calib_camera.py --capture`
       (8-12 tilted views, then flat at the measured spot). PASS: reprojection < 1.5 px, elevation >= 40 deg.
4. [ ] Joint offsets (RUNBOOK 3): mirror check (wrist roll! tonight's evidence: the active map's roll is ~90 deg off),
       then `read_pose.py` touch fit -> active = bench_fit, verified. PASS: tip rms <= 5 mm.
5. [ ] Gravity sag: comp is ON by default (k 0.05). With armd running, `$PY deploy/scripts/fit_sag.py` (8 small moves) prints k: put it in bench.darwin.json control.gravity_comp_k.
6. [ ] Perception setup, ONE Gemini request (RUNBOOK 4c): objects placed, arm at start pose.
       `$PY deploy/scripts/perceive.py --task tasks/cube_in_bowl/task.json --setup-out deploy/runs/setup_cube_in_bowl.json`
       Check the outlines. Then `export R2S_CACHE_ONLY=1` for everything after (0 requests).
7. [ ] Scripted probe at 0.3 speed, recorded (this is the robot's own "real" episode for Argon's twin):
       `$PY deploy/scripts/run_policy.py --policy stub --attempts 1 --speed 0.3 --record --task-json tasks/cube_in_bowl/task.json --perception-setup deploy/runs/setup_cube_in_bowl.json [--allow-unverified]`
8. [ ] Policy: `--policy d_ens3_search --speed 0.3`, then 0.6, then 1.0 after two clean runs. 10 scored trials (RUNBOOK 6).
       Fallback, always works: the scripted stub.
9. [ ] Argon twin of the probe (needs gemini-4-argon): `python -m argon run --task cube_in_bowl --video <probe attempt dir>`
       or `real2sim: python -m scripts.run_bench <attempt dir>`. Pin dashboard real2sim_run to that run.
10. [ ] TIC-TAC-TOE vs the robot (centrepiece; RUNBOOK 4e). Grid: print deploy/assets/ttt_grid_letter.png at 100 %
       (check the 100 mm bar), tape its marked corner at base x .1775 / y -.0675. Robot's piece on the supply mark x .23 / y .15.
       `python deploy/scripts/tictactoe.py --first robot --policy stub --publish-port 8790 --allow-unverified --speed 0.3 --piece-size <x,y,z of the robot piece, m> --robot-desc "<e.g. red bottle caps>" --human-desc "pennies and nickels (coins)"`
       (Argon reads the board + picks the move each turn: R2S_MODEL_MAIN=gemini-4-argon; untested live; dry games won/drew.)
       Pieces: dry runs used 38 mm cubes; the printed 40 mm discs (P2S, later) are untested.
       Keep --policy stub (scripted IK place). Learned place_on_cell_v1 / _ens3 are weak (0.40 / 0.51 in the twin): do not demo them.
11. [ ] Optional second task: screwdriver -> `tasks/pick_place_stick`, policy `pick_place_stick_ens3` (starts from its own
       home pose automatically), its own perception setup (one request).

## Honest numbers to quote (fill the [brackets] in PITCH.md only with today's measurements)
- Policy in MuJoCo, cube in bowl: d_ens3_search 94 % clean / 57 % with real-world noise (200 scenes); more play ->
  better robot curve 53 -> 93 % over 50 -> 873 game episodes.
- Practice bench (kinematic mock) is NOT evidence for the real arm: its grasp model and sag are known-wrong; only the
  real trials count.
- Bots solve every task in the twin (0.88-0.96); task policies 0.45-0.87; game success check = Python's on 854 checks.

## F: `PITCH.md`

# Pitch: 3 minutes, no slides

Fill the [brackets] with tomorrow's real numbers. Never say a number that wasn't measured that day.

## 0:00 - 0:20  The problem (stand next to the arm)
"Robots learn from demonstrations, and demonstrations cost a person at a robot, by the hour.
Roblox has tens of millions of people playing every day. We connect the two."
(Move the follower arm by hand, torque off. The Roblox twin on screen moves with it.)

## 0:20 - 1:10  Argon builds the twin (point at the dashboard, REAL2SIM panel)
"This morning we recorded this arm doing the task once. Gemini 4 Argon watched that video and the joint log.
The camera is fixed by this printed marker. Argon found the cube and the bowl, used the moment the gripper closed and
opened to pin them in 3D, and built the scene in Roblox.
Then it renders the twin, puts it over the real video, critiques the difference, and fixes it. Every round.
The gap number here is measured on frames the critic never saw: [today's run: gap first -> best, motion px first -> best].
A round that makes any object's overlap worse is rejected, even if the physics score improves."
(Quote only today's run. The Oct 24 synthetic ".72 -> .96" came from a keep rule that has since been tightened.)
"It also tunes the physics. The cube slides as far in Roblox as it did on the real table: [error X mm -> Y mm]."

## 1:10 - 1:50  People play (hold up the QR code)
"That twin is a published Roblox game. Scan it. You drive the arm on your phone and put the cube in the bowl.
Every round you play becomes training data. [N] rounds so far today, [M] players."
(A judge plays. The episode counter on the dashboard ticks up.)

## 1:50 - 2:40  The real robot (back to the arm)
"This policy was trained only on game play. It never saw a real demonstration."
(Run it: cube into bowl.)
"In a second simulator it succeeds [89 %] of the time. On this table it was [X / 10] this afternoon.
Argon is the eyes and the planner, about one call a second. The small policy is the hands, at 30 Hz."

## 2:40 - 3:00  The product
"Any robot company uploads a video of its robot doing a task. Argon builds the game, players generate the data,
and a policy comes back. More play, better robot." (Point at the scaling curve.)

## Fallbacks, in order
1. The real-arm policy fails: show the run log and say it straight: "[X/10] earlier; the camera estimate was off by N mm here."
   Then run the scripted baseline, or move the arm by hand with the twin mirroring it. (No leader: its 7.4 V servos
   can't take the only supply we have, 12 V.)
2. Argon is slow or rate-limited: show the run finished before the slot (same artifacts, timestamped) and start a fresh one live.
3. Roblox publishing or the tunnel is down: play in Studio on the laptop. The episodes still log locally.

## Likely Q&A
- **Why Roblox, not Isaac or MuJoCo?** Because players are already there and it runs on phones. The physics holds up:
  joint tracking within 0.06 deg of MuJoCo, friction-only grasp 4/4, push distances within 5-12 mm.
- **Sim-to-real gap?** The policy is state-based (joint angles plus object positions from a calibrated camera),
  so the Roblox graphics never have to look real. The physics gap is what Argon's loop calibrates.
- **What does Argon add over a fixed pipeline?** It reads arbitrary real video, works out the scene and the camera,
  judges its own render against reality, and fixes it. Long context covers the whole video plus the joint log.
- **Is crowd data any good?** Every episode is checked by physics in the game (success flag) and can be filtered.
  The training set mixes bot players with deliberate mistakes and human play; the curve shows success rising with episodes.
- **Why not have Argon control the arm directly?** We measured about 1.3-5 s per call. Control needs 30 Hz.
- **Other robots?** The twin builder takes any MJCF/URDF; the SO-101 is the first template.
- **Why a marker, not Argon, for the camera?** We tried calibrating from Argon's points on the arm: off by 86-89 cm on
  a synthetic check (its wrist/elbow points were 50-70 px off). Once the lens focal length is known, jaw tips + base
  recover it to 5.3 cm. The marker is 0.2 cm, so the marker sets the camera and Argon's points are a cross-check.
- **What breaks?** Depth from a single low camera (fixed with a high camera, plus grasp and release as anchors),
  and joint-zero calibration on real arms (we fit it on the bench).

## F: `PLAN.md`

# Gemini 4 Argon SF hackathon (Sun Oct 25 2026, SHACK15) — plan

## Pick: Idea 2 with Idea 1's split inside it. Track 2 ("render, watch, critique, fix").
**Pitch:** "Show Gemini your robot doing a task once. It builds a Roblox twin, checks it against the real video
until they match, and publishes it as a game. People play; their play trains the robot."
User: a robotics team that needs task data and can't afford teleop hours. Argon = scene builder + critic + data QC
(slow, ~1-5 s/call, measured); small trained policy = fast low-level control.

## 3-minute demo
1. 0:00 Real SO-101 + leader arm: teleop a pick-and-place (cube -> bowl) for ~15 s, high front camera + joint log.
2. 0:20 Argon real2sim, live dashboard: camera fitted from the arm's own joints (overlay), objects placed, render vs
   real overlay iterating; gap metric drops each round (spike: object IoU 0.40 -> 0.75, held-out, not self-graded).
3. 1:20 QR code -> published Roblox game; judges play on phones; episode counter ticks up.
4. 2:10 Policy trained on today's game play runs on the real arm (Argon points at the cube through the calibrated camera).
5. Always-works fallback: leader arm drives real follower and Roblox twin side by side (SSE 60 Hz, ~10 ms).

## Spike evidence (Oct 24) — see real2sim/FINDINGS.md, roblox/FINDINGS.md
- Roblox twin: FK 0.001 mm, joints 0.01-0.06 deg rms vs MuJoCo, push within 5-12 mm, friction-only grasp 4/4,
  silhouette 95-98 % vs MuJoCo, teleop pick-carry-place in Play mode, 30 Hz episode logging.
- Real2sim on real pixels: held-out arm keypoints 10.8 px (jaw tips); object IoU 0.40 -> 0.75 over critique rounds.
  Depth along a low camera's line of sight stays ~7 cm off -> mount camera high (>= 40 deg down) + use
  grasp/release joint angles as 3D anchors.

## Must fix / open
- Argon access: free key = 0 quota. Use hackathon credits or enable billing; check daily quota first (loop needs ~36 calls).
- Loop: keep-best + stop rule, bounded thinking, grasp/release anchors, `container` boxes.
- Game: touch controls for phones; public URL (tunnel) + Allow HTTP Requests; import meshes; clean Studio (2 s hitch every 10 s).
- Joint frame: verify owner's arm vs official MJCF zeros with the live mirror (known offsets ~11.5 deg lift, +76 roll vs corpus).
- Policy: state-based (joints + object pose) so the Roblox render never has to look like a camera image.
- Check the event's rules on pre-built code; present the twin/recipe as existing tooling if needed.

## F: `real2sim/README.md`

# real2sim

Gemini watches one real LeRobot SO-101 episode and builds a MuJoCo twin in `spec/scene_spec.md` format, then a
render -> critique -> patch loop refines it. Results and numbers: `FINDINGS.md`.

```
uv venv .venv --python 3.11
uv pip install --python .venv/Scripts/python.exe mujoco opencv-python numpy pandas pyarrow av huggingface_hub google-genai scipy pillow
export PYTHONPATH=.            # run everything from real2sim/
export R2S_MODEL=gemini-4-argon   # default gemini-robotics-er-2-preview; one-line switch
HF_HUB_DISABLE_XET=1 .venv/Scripts/python.exe -m r2s.data 0     # episode 0: states + decoded front frames -> data/
.venv/Scripts/python.exe -m scripts.s1_point                     # Gemini keypoints on 30 frames (cached in out/points/<model>/)
.venv/Scripts/python.exe -m scripts.s1_calib                     # camera fit + held-out reprojection -> out/calib/
.venv/Scripts/python.exe -m scripts.s2_objects                   # detection -> scene_spec iter 0 -> out/loop/scene_iter0.json
.venv/Scripts/python.exe -m scripts.s4_loop 5 <run_name>         # critique loop -> out/loop/<run_name>/
.venv/Scripts/python.exe -m scripts.s4b_eval <run_name>          # re-score all iterations -> metrics_table.md
.venv/Scripts/python.exe -m scripts.s5_latency <model> 0 none    # pointing latency/accuracy per thinking budget
.venv/Scripts/python.exe -m scripts.s6_physics <scene.json>      # replay actions with physics (grasp does not work yet)
```

The API key is read at runtime from `GEMINI_API_KEY` or `C:/Users/sarta/roblox-studio-mcp/.env`; every call is
logged (latency, tokens, model, tag) to `out/gemini_calls.jsonl`. Daily-quota 429s fail fast; RPM 429s back off.

## v2: one command, bench-ready (Oct 24 evening)

```
export PYTHONPATH=. R2S_MODEL=gemini-4-argon            # the only switch for tomorrow
.venv/Scripts/python.exe -m scripts.run_bench ../deploy/runs/<ts>_session      # a record_session.py output
.venv/Scripts/python.exe -m scripts.run_bench public:0                          # the spike's public episode
.venv/Scripts/python.exe -m scripts.synth_bench                                 # synthetic bench session (pick + place,
.venv/Scripts/python.exe -m scripts.run_bench data/bench/synth_pick_place       #   true poses in truth.json) and its run
```
Flags: `--run NAME` (default `<session>_<model>_<time>`), `--iters 6`, `--patience 2`, `--max-calls 12`,
`--budget-s 1800`, `--anchors auto|hard|soft`, `--focal-px F` (known focal for the keypoint camera), `--no-physics`, `--ref` (one extra request: independent detection
on the eval frames with `R2S_REF_MODEL`, default gemini-3-flash-preview), `--camera <CameraModel json>`, `--task`.
Env: `R2S_THINKING` (thinking budget, default 2048; `none` = model default; a model that rejects a budget is retried
without one), `R2S_NO_CACHE=1`.

Requests per run: 1 keypoint batch (all ~14 frames in ONE request; 0 for `public:0`, which reuses the spike's
ER-2 points) + 1 detection (first + last frame in one request) + 1 per critique iteration (all keyframes and the
physics image in one request). Every response is cached in `out/cache/<sha256(model, config, all parts)>.json`:
an identical rerun costs nothing. A daily-quota 429 stops the loop cleanly and keeps the best scene.

What happens (code in `r2s/`):
1. `anchors.py`: grasp/release from the joint log alone. A hold = gripper stopped part-closed after closing,
   flat >= 0.4 s (and commanded more closed than measured, when `leader_deg`/actions exist). The object sits at
   the narrowest point of the jaw gap (`physics.jaw_gap`, the twin's finger boxes; the tip midpoint can be ~1 cm
   off) and is as wide as the gap: a position anchor at the grasp, a release point, and a size anchor.
2. Camera. Bench: the ArUco CameraModel (`<session>/camera.json`, else `deploy/config/camera.json`, which must be
   the calibration of THIS mount); frames are remapped to an ideal pinhole so spec camera and renders are exact.
   Always also (Gemini showcase, reported side by side): keypoints-only fit and keypoints+anchors fit (the object's
   detected pixel at rest / after release must be the projection of the anchors: table-level points that pin the
   height-vs-pitch trade-off the arm keypoints leave open), and, when the focal is known (ArUco K or
   `--focal-px`), keypoints (known f)+anchors: jaw tips + base only, focal fixed. With the focal free the keypoint
   camera trades focal for distance (synthetic, gemini-3-flash points: 86-89 cm from truth at f pinned to the
   1200 px bound); with it fixed: 5.3 cm, held-out jaw 4.1 px. Camera used by the loop: ArUco if present, else
   keypoints (known f)+anchors with --focal-px, else keypoints+anchors.
3. `detect.py`: scene_spec iteration 0 from one detection request; objects matched to colour masks (bench.json
   HSV, else a window around Gemini's colour) for independent scoring.
4. `patch.py`: hard constraints re-applied after every patch. Gripped object: width within +-20 % of the jaw gap;
   with `--anchors auto` (default) and image/anchor agreement <= 2.5 cm, its depth along the camera's line of
   sight is fixed to the grasp anchor (2 cm lateral slack) and the container must hold the release point.
   Disagreement > 2.5 cm (public:0: 3.5-4.3 cm, its joint offsets are unknown) -> anchors reported, not enforced,
   and the physics replay puts the object at its grasp anchor (FK frame). Patches are JSON, validated field by
   field (`PATCH_RULES`), the patched scene is validated against the spec (`validate_scene`).
5. `evaluate.py` (independent; the critic never sees the eval frames): object IoU and centroid px vs colour
   masks, held-out keypoint px, anchor errors, truth errors (synthetic), physics. score = mean IoU - 0.005 *
   centroid px - 0.004 * trajectory px. Keep-best per object: the whole patch and each object's part alone are
   scored; the best variant is kept only if it beats the best scene by > 0.003.
6. `physics.py`: the measured joint trajectory replayed on the Roblox twin (official dynamics, the Roblox voxel
   collision boxes); gripper commanded shut during holds. Object track vs the colour-mask track in the video:
   median px per phase (before / hold / after), trajectory error = mean of the phase medians. The critic tunes
   friction and mass. Frame note: twin_mjcf's spec origin is the MJCF base origin, real2sim/deploy's is the base
   footprint centre (+16.8 mm in x); physics.py converts.
7. Stop: critic says done with no patch, `--patience` iterations without gain, `--max-calls`, `--budget-s`, quota.

### Live output (dashboard contract), `out/live/<run>/`

All JSON is written to a temp file and renamed (`os.replace`): a reader never sees a partial file. Paths inside the
JSON are relative to the run directory.

`run.json` (rewritten at every stage and iteration):
```
schema "r2s_live/1", run, status ("starting"|"running"|"done"|"failed"), stage ("anchors"|"keypoints"|"detect"|
"evaluate"|"loop"|"critique N"|"done"), started, updated, elapsed_s, session, session_kind ("public"|"bench"),
video, model, task, thinking_budget, fps, n_frames, (session_kind "synthetic" for generated sessions)
iters: [{iter, file: "iter_N.json", score, kept, best_so_far, thumb: "iter_N_overlay_0.jpg"}],
(score: higher is better, NOT bounded to [0, 1]: it goes negative when the physics replay fails)
best_iter, stop_reason, calls {requests, cache_hits, errors, latency_s},
anchors [...], keypoints {...}, calibration {keypoints|keypoints+anchors|keypoints (known f)+anchors|aruco: {f_px, position, heldout_median_px,
anchor_reproj_px, dist_to_aruco_cm?, dist_to_truth_cm?}}, camera_source, track {object id: mask name}, links,
eval_frames, critic_frames, constraints, anchor_consistency, detection, baseline {metrics, images},
v1_best (public:0 only), when done: best_scene "scene_best.json", summary "summary.json"; when failed: error
```
`iter_N.json`: `iter, kind ("initial"|"critique"), metrics, kept, best_iter, why, images {overlay_0.., physics,
scene}, constraint_notes, patch (validated), applied, rejected, critic {latency_s, cached, thinking_tokens,
total_tokens, discrepancies, done}, t_eval_s, t_iter_s, t_wall_s, calls`. Images are absent on an iteration whose
patch produced no valid scene.
`metrics`: `score; objects {id: {iou, centroid_err_px, ref_center_err_px, n_frames}}; anchors [{object,
grasp_xy_err_cm, container, release_to_container_centre_cm, release_inside}]; kp_heldout_median_px;
kp_heldout_jaw_px; truth {id: {xyz_err_cm, xy_err_cm}, camera_pos_err_cm} (synthetic); physics {traj_err_px,
traj_err_px_before|hold|after, sim_max_lift_m, sim_end_xyz, sim_ends_in_container, sim_end_to_release_cm,
sim_end_truth_err_cm?, object_placed_at?}`.
Images: `iter_N_overlay_J.jpg` (1280x480: real keyframe with the sim silhouettes outlined | sim render; J indexes
`critic_frames`), `iter_N_physics.jpg` (real frame with real track green / sim track red | physics sim at that
frame), `iter_N_scene.json` (the candidate scene), `iter_0_baseline_*` (v1-style: keypoint camera, no anchors).

## v3 (Oct 25): any task, from one video: `task_from_video`

```
export PYTHONPATH=.
export R2S_MODEL_MAIN=gemini-4-argon        # default; reasoning: task from video, critique loop, scene detection
export R2S_MODEL_SMALL=gemini-3.8-flash     # default; keypoints, boxes on a frame (R2S_MODEL = legacy main override)
.venv/Scripts/python.exe -m scripts.task_from_video <session_dir | public:N | video.mp4 | frame.jpg>     [--instruction "..."] [--name NAME] [--iters 4] [--max-calls 8] [--no-loop] [--allow-fallback] [--frames]
.venv/Scripts/python.exe -m scripts.synth_task_bench [--task task.json] [--place id=x,y,yaw ...]  # synthetic session
```
Writes `../tasks/<name>/task.json` (argon_task/1, validated by `argon_tasks.validate`) and, when the input has a
joint log, `../tasks/<name>/scene.json` (scene_spec v0, ids = task ids, `task_shape` / `role` kept) plus the usual
live output `out/live/<run>/` (r2s_live/1; run.json also carries `task_name`, `task_json`, `model_small`, and
`calls.by_model`). Request/answer/validation log and boxes: `out/tasks/<name>/`. Last stdout line = result JSON.

Models are roles (r2s/gemini.py, same in deploy/so101d/gem.py). A PerDay / limit-0 429 on the main role raises
QuotaError: exit 1 with a clear message, never a silent stand-in; `--allow-fallback` (R2S_ALLOW_FALLBACK=1) retries
on the small model and records `fallback_from`. Every record logs the model that served it. `R2S_CACHE_ONLY=1`:
zero requests (a cache miss fails); `--answer file.json`: offline answer.

1. ONE long-context main-role request: the whole episode as a video (a session's frames encoded to mp4 at <= 10 fps;
   `--frames` = 2 fps images instead), its first and last frame (for boxes), the whole joint log as a 5 Hz table
   (joints + jaw midpoint by FK, base frame) and the grasp / release anchors as measured facts (jaw width at the
   grasp = the object's width; where it was picked and let go). A single frame needs `--instruction`.
2. Answer: the task in the closed vocabulary + boxes of every object in the first and last frame. Validation errors
   (argon_tasks + colour words / metre sanity / "something was grasped but nothing is manipulated") go back to
   Gemini (<= `--retries 2`); unparsable JSON is re-asked with the parse error; missing randomize ranges are filled
   locally (positions are measured by the scene fit, not guessed). The grasped object's width is pinned to the
   jaw-gap anchor when Gemini is > 25 % off. The format example is a different task (marker on a coaster): an
   example equal to the bench leaks its sizes.
3. `run_bench --task-json [--task-boxes] --no-keypoints`: the task's objects replace the free-form detection (boxes
   from step 2 when drawn on the same session, else one small-role detection request); each object's real mask is
   an appearance model fitted on the first frame (GrabCut in the box, then only the task colour's hue, -> HSV
   histogram vs the rest of the frame; deploy/so101d/appearance.py), not bench.json colour ranges; iteration-0 poses
   come from the real arm's silhouette fitters (deploy/so101d/taskperception.py: hull over yaw for boxes, rim circle
   for bowls; a stick whose colour model covers one end, a screwdriver's handle, is posed from that end + the box,
   and its length is fitted to the box). Part-tracked objects are left out of the IoU score and their pose is frozen
   (their real mask is one end). Grip width constrains only the dims across the jaws. Bowls render as open cylinders.
   Randomize ranges are widened to contain the fitted poses.
Requests per task: 1 main (+ <= 2 re-asks) + 0-1 small + 1 main per critique iteration.

Results (synthetic sessions with truth; deploy/so101d/synth_task.py renders any task; real camera = perturbed ArUco):
| input | model (stand-in for main) | task inferred | scene error vs truth (iter 0 -> best) | requests, wall |
|---|---|---|---|---|
| tonight bench session (brown 38 mm cube -> blue 110 mm bowl, yellow screwdriver), no instruction, keyframe prompt | gemini-3.7-flash | "put the brown cube in the blue bowl", cube 38 mm, bowl 110/30, screwdriver obstacle + not_moved | cube 3.1 mm, bowl 4.1 mm, screwdriver 2.5 cm (before the handle model); replay lifts the cube 11 cm at best iter | 4, 715 s |
| screwdriver -> green square session, no instruction, whole-video prompt | gemini-3-flash-preview | "place the screwdriver on the green mat", stick 0.16x0.025 (truth 0.20x0.025, box fit 0.193), mat 0.10x0.10, cube obstacle + not_moved | screwdriver 2.0 mm / 1.7 deg, mat 3.7 mm, cube 3.7 mm; replay lifts it 10 cm | 1 + 3, 44 s |
| same scene as a bare mp4 (no joint log, no instruction) | gemini-3-flash-preview | "place the screwdriver on the green mat", stick 0.18x0.02, mat 0.10x0.10x0.005, block 0.04 | (no loop without a joint log) | 1, 6 s |
| public:0 (LeRobot phi_so101 ep 0, keypoint camera, no ArUco), no instruction | gemini-3-flash-preview | "put the red cube in the cardboard box", cube 24 mm, box modelled as a 12x8 cm open "bowl" | no truth; eval IoU cube .92 / box .40 at iter 0; the old rule kept iter 1 (cube IoU .31 for the replay); the keep rule rejects it, best = iter 0 | 1 + 1, 27 s |

Keep rule (r2s/keep.py, Oct 25 later): a candidate is kept only if the score improves AND no object's held-out IoU
drops > 0.10 below its best so far AND (anchors not enforced) no container's IoU drops > 0.02 AND the camera term
(+1 px) and, with anchors enforced, the anchor terms (+0.5 cm, release still inside) do not get worse; otherwise
`REJECTED (iou_regression | anchor_regression | camera_regression)` in iter_N.json ("verdict", "verdict_reasons").
`python -m scripts.rescore_keep [run ...]` re-judges recorded runs offline. Changed verdicts: public:0 task run iter 1
(cube IoU .92 -> .31) now REJECTED, best = iter 0 (re-run with R2S_CACHE_ONLY=1: 0 requests, same verdict);
synthetic tonight iter 1 (cube .89 -> .76) REJECTED; synth_g3flash iter 1 (cube .72 -> .56) REJECTED; screwdriver
run b iters 2-3 REJECTED (anchor_regression: the critic moved the screwdriver 0.9-1.5 cm off its grasp anchor).

Not done / open: no run on gemini-4-argon (429, no billing) or gemini-3.8-flash (its free-tier day quota was used up
by 03:33); a part-tracked object's physics track
is the handle's centroid, not the stick's centre.

## F: `real2sim/FINDINGS.md`

# real2sim spike findings (Oct 24 2026)

Data: `BrutalCaesar/phi_so101_cubes_cylinder_recovery_v1` @ `cf44c09`, episode 0 (718 frames), `front` camera
(fixed, low, facing the arm; pick red cube, drop in cardboard box). Official `so101_new_calib.xml` takes the dataset's
degrees directly; shoulder_pan sign +1 (jaw-tip error 10.8 px vs 31.8 px with -1).

## 1. Camera from the robot
`gemini-robotics-er-2-preview` points at arm keypoints in 19 frames; 11 fit the camera (f = 477 px, fovy 53.4 deg),
8 held out. Held-out error: jaw tips median 10.8 px, base 10.1, elbow 16.6, wrist 20.1, shoulder 32.6, all 13.9
(p90 55). Fitting joint offsets too moved them < 2.3 deg and did not help. Rendered arm overlays the real arm in all
19 frames (`out/calib/calib_montage.jpg`).

## 2-4. Detection + 5 critique iterations
Metrics never come from the critic: IoU and centre errors use colour masks of frames the critic never sees; "ref" is
a separate `gemini-3-flash-preview` detection; "cube to grasp point" compares the cube with where the jaws closed
(from joint angles only).

| run | iter | cube IoU | box IoU | cube ref px | box ref px | cube to grasp point |
|---|---|---|---|---|---|---|
| both | 0 (single pass) | 0.40 | 0.41 | 10.4 | 22.6 | 6.6 cm |
| 3.8-flash | 5 | 0.72 | 0.75 | 2.8 | 21.0 | 7.8 cm |
| 3.7-flash | 3 (best) | 0.75 | 0.83 | 0.6 | 8.5 | 7.5 cm |
| 3.7-flash | 5 | 0.73 | 0.78 | 1.7 | 17.8 | 7.7 cm |

- The 2D gain is real and repeats across two critics (objects were ~1.4x too large; box 15x18x7 -> ~11x8.5x4.5 cm,
  cube 3.5 -> 2.5 cm).
- The loop oscillates and the critic declared "done" early: needs keep-best and a stop rule.
- 3D did not improve: the camera is 16 cm high looking down ~14 deg, so depth along the line of sight is weakly
  constrained; a jaws-grip-cube-at-mid-height constraint fits keypoints equally well (14.1 vs 13.9 px) with a camera
  6 cm higher. Keypoints alone cannot pin depth.

## 5. Latency
| call | model | median |
|---|---|---|
| 6-keypoint pointing | ER-2, default thinking | 5.4 s (n=19) |
| jaw-tip pointing | 3.5-flash, thinking_budget=0 | 1.3 s (1.1-1.6), ~8 px vs FK |
| critique | 3.8-flash / 3.7-flash | 161 s / 128 s (2-10k thinking tokens; 3.8 had 503s) |

Gemini cannot sit in a control loop (~1 Hz at best). It fits as planner / scene builder over a fast policy.

## 6. Physics
Replaying the recorded actions in MuJoCo never grasps the cube, even with the cube placed exactly at the grasp
point (the gripper pushes it). Sim grasping is unsolved.

## For Argon tomorrow
1. `R2S_MODEL=gemini-4-argon`. Check the key's daily quota first (free tier was 20 req/day/model; pipeline needs
   ~30 pointing + 1 detection + 5 critiques; batch frames per request).
2. Fix depth with a second constraint: grasp/release frames as 3D anchors (joint angles give the cube and box
   positions), or calibrate a second camera and triangulate. On our own bench: mount the camera high (>= 40 deg down).
3. Keep-best + real stop rule. 4. Bound thinking budget. 5. Spec gets optional `"container": true` on `box`.

Outputs: `out/calib/`, `out/loop/<run>/{montage.jpg,metrics_table.md,scene_iter*.json,critique_iter*.json}`,
`out/gemini_calls.jsonl`, `out/latency/`, `out/physics/`. Commits 43728e1, 308a22a, 2ca9e00, 9040eb6, f0a690a, b918cb2.

# v2 (Oct 24 evening): one-command bench loop. Commits 0f5e5eb, e076884
`cd real2sim; PYTHONPATH=. python -m scripts.run_bench <session_dir|public:0> --model $R2S_MODEL`
(`python -m scripts.synth_bench` makes a synthetic pick-and-place with ground truth). Live output for the dashboard:
`out/live/<run>/` (run.json schema r2s_live/1, iter_N.json, iter_N_overlay_J.jpg 1280x480 real|sim, iter_N_physics.jpg).
Tested on gemini-3-flash / 3.5-flash / 3.7-flash; gemini-4-argon untested (quota 0). Switching model = `R2S_MODEL` only.

| run | IoU cube / container | 3D error | trajectory px |
|---|---|---|---|
| synthetic, v1-style (keypoint camera, no anchors) | .45 / .64 | cube 0.9 cm, bowl 4.2 cm | 73, never grasped |
| synthetic, v2 best (iter 3, ArUco camera + anchors) | **.96 / .69** | **0.1 / 1.2 cm** | **13**, ends in bowl |
| ep0, v1 spike best | .75 / .83 | 8.1 cm from grasp anchor | 81, misses box |
| ep0, v2 best (iter 2, 3.7-flash) | **.78** / .50 | 4.3 cm | **16**, ends in box |

Cost: 1 keypoint batch + 1 detection + 1 request per iteration (5-7 per run, all cached on disk). 212-241 s per run;
per iteration 16-40 s with a fast critic, 60-110 s with 3.7-flash.

Still failing:
- **Camera from keypoints alone: 86-89 cm off on synthetic** (Gemini's wrist/elbow/shoulder points 50-70 px wrong;
  with 4 px noise the same fit is within 1.4 cm). Free focal trades against distance; with focal fixed to the ArUco
  value, jaw tips + base + anchors land 5.3 cm off (4.1 px held-out jaw), reported alongside (`--focal-px` without a
  board; commit 17a4181). The camera source is the ArUco board; Argon's keypoints are a cross-check.
- ep0 joint log vs camera disagree 3.5-4.3 cm, so anchors are not enforced there; ep0 box IoU below v1 (.50 vs .83).
- Sim grasp is knife-edge (works at friction 0.4-0.7, fails at 1.0). Bowl renders as a solid cylinder (IoU cap ~.7).
- Frame origins: real2sim and deploy use the base footprint centre; Roblox SceneLoader and policy/sim.py use the MJCF
  base origin (16.8 mm apart in x). Convert at the boundary.
- Synthetic sessions now say session_kind "synthetic" and run.json carries `video` (17a4181).

## F: `roblox/FINDINGS.md`

# OWNER: publish the game (demo checklist, ~10 min)

The place `roblox/so101_twin.rbxl` (saved 03:40 with the real meshes in ReplicatedStorage/SO101Meshes and the synced
game scripts) is ready. Do this in that one Studio window, in edit mode (no playtest running).

1. **Trace server + tunnel** (where every round lands), two terminals in `C:\Users\sarta\argon-hack\roblox`:
   `.venv\Scripts\python tools\joint_server.py --port 8770 --key <long-random-secret> --out out`
   `C:\Users\sarta\bin\cloudflared.exe tunnel --url http://127.0.0.1:8770`  -> note the `https://<words>.trycloudflare.com` URL.
   Traces arrive as `roblox/out/traces/*.jsonl` (human_* = cube_in_bowl rounds, game_<level>_* = every other level).
2. **Key into the game:** Explorer -> ServerStorage -> TwinConfig (double-click) -> set `TRACE_KEY = "<same secret>"`,
   Ctrl+S. (Or edit `roblox/src/ServerStorage/TwinConfig.lua` and run `node tools/sync.mjs`; a later sync overwrites
   edits made only in Studio.) TRACE_URL can stay empty: step 7 sets it in game.
3. **Publish:** File -> Publish to Roblox -> "Create new experience" -> Name `SO-101 Twin: Teach the Robot`,
   Creator = **your own account** (the same "Me" that imported the meshes, or the mesh assets will not load), Genre
   any -> Create. Wait for "Publish succeeded".
4. **Game Settings** (Home tab -> Game Settings):
   - Security: **Allow HTTP Requests = ON**, **Enable Studio Access to API Services = ON** (DataStores: profiles,
     stars, leaderboards, "rounds today") -> Save.
   - Basic Info -> Playable Devices: **Computer, Phone, Tablet ON** -> Save.
   - Permissions -> Playability: **Public** -> Save.
   - Places -> (the start place) -> ... -> Edit -> Max Players: **8** (one station per player; TwinConfig.MAX_STATIONS = 8) -> Save.
   Then File -> Publish to Roblox once more (Alt+P).
5. **Game link:** Home -> Game Settings -> Basic Info shows the experience; or in the command bar (View -> Command Bar)
   run `print(game.PlaceId)`. The link is `https://www.roblox.com/games/<PlaceId>`. Put it in
   `dashboard/config.json` -> `"game_url"` (or start the dashboard with `--game-url <link>`) and make the QR from it.
6. **Smoke test on a phone:** open the link in the Roblox app, PLAY (10 s tutorial), one level. Expect the result card
   to say "This round is now training data" and the joint_server terminal to print `trace game_tutorial_lift_...`.
7. **Point the live servers at the tunnel:** in the running game, open chat and type
   `/traceurl https://<words>.trycloudflare.com` (owner only; stored in a DataStore, all servers pick it up within 30 s).
   `/traceurl` alone shows the URL and upload counters; `/traceurl off` stops recording.
8. Optional: upload a photo of the real arm (Creator Hub -> Decals), then in game chat `/realarm rbxassetid://<id>
   caption` to replace the "LIVE soon" card. The pipeline can also serve `{"real_arm": {...}}` at `<trace url>/status`.
9. Optional data when nobody plays: Explorer -> Workspace -> Attributes: `TwinBots = 2` (two bot stations cycling all
   levels, recorded as source "scripted"), or TwinConfig.BOT_STATIONS. Leave it 0 for the demo so players get stations.

If the tunnel dies: restart cloudflared and repeat step 7 with the new URL (rounds played meanwhile are queued per
server, up to 200 parts, and retried).

# Roblox SO-101 twin spike findings (Oct 24 2026)

Verdict: faithful enough for real2sim render-compare and crowd teleop. Model: official TheRobotStudio
`so101_new_calib.xml` (`model/so101_official/`). MuJoCo comparisons use the same collision boxes and the same
targets at the same frame times.

| Check | Result |
|---|---|
| FK, 30 random configs | 0.001 mm kinematic; Run mode with gravity 0.17 mm mean / 0.41 mm max vs commanded |
| Dynamics (6-joint sinusoids + step) | joint 0.01-0.06 deg rms (0.63 max) vs MuJoCo; EE 1.3 mm rms / 5.5 max (lag at speed). robot-academy spring recipe: 100 mm rms |
| Push (4x4x2 cm block flicked) | friction 0.3/0.6/1.0: Roblox 118/90/80 mm vs MuJoCo 112/98/82; mass 0.03/0.1/0.3 kg: 90/95/99 vs 98/95/99; final positions within 5-12 mm |
| Grasp, friction only (3 cm, 20 g, mu 0.6) | 4/4 lifted 7 cm, slip < 0.1 mm over 2 s (MuJoCo 4/4). Also 2/2 at mu 0.3/50 g and mu 0.6/200 g; mu 0.15/100 g Roblox 1/2, MuJoCo 0/2 |
| Render-compare | robot silhouette vs MuJoCo 95-98 % overlap, < 1 px offset (2 poses x 2 cameras) |
| Live streaming | SSE 60-62 msg/s, sample age p50 9 ms / p90 18 ms; WebSocket 60 msg/s 0.8 ms RTT; polling capped ~8 Hz on live servers -> use SSE |
| Teleop | scripted operator in Play mode (real player present) picked, carried 13 cm, placed; gripper error 1-2 mm; 30 Hz episodes logged |

## What it took
- robot-academy inertia-box recipe fails here: SO-101 motor armature is 100-1000x link inertia and leaks into other
  axes (base joint 4x too heavy). Fix: `RobotKit/JointSim.lua` runs MuJoCo's servo model (full mass matrix + bias)
  in Luau at 480 Hz; Roblox hinge motors follow it each frame; links stay real bodies for contact; torque cap.
- Twin parts must be server-owned (`SceneLoader.serverOwn`), else the joining client takes ownership and the arm thrashes.
- Collision = 4 mm voxel boxes of the meshes (648 parts), mirrored in the MuJoCo comparison model.

## Blockers for the owner
1. Published game needs Game Settings -> Security -> Allow HTTP Requests + a public URL (tunnel), not 127.0.0.1.
2. Meshes: Studio Import 3D -> `meshes/*.obj` (13 files) -> `ReplicatedStorage/SO101Meshes`, names unchanged; builder
   fits them. Until then Play mode shows the voxel arm (edit mode renders real meshes via `tools/load_meshes.mjs`).
3. Studio freezes ~2 s every ~10.5 s in every mode even with an empty script (`spikes/hitch_probe.luau`): environment
   / plugin. Freezes the live twin. Try a clean Studio (minimal plugins) before the demo.
4. Real keyboard path unverified (bridge key presses don't reach the Play client): press W/A/S/D, Q/E, Space once.
5. Bridge commands land in a random Studio window when several are open: `tools/twin_bridge_server.mjs` (port 8766)
   gives this window its own channel. Keep only one Studio open on demo day.

## Run
- Place: `so101_twin.rbxl` (`rojo build default.project.json -o so101_twin.rbxl`; `node tools/sync.mjs` pushes src/).
- Workspace attribute `TwinDrive` = "teleop" (default) | "live"; `TwinServer` = stream URL (default http://127.0.0.1:8765).
- Stream/episode server: `.venv\Scripts\python tools\joint_server.py` (/q /since /sse /ws, POST /episode /set ->
  `out/episodes/`).
- Key files: `src/ReplicatedStorage/RobotKit/JointSim.lua`, `Twin/{SceneLoader,ArmIK,EpisodeLogger,LiveLink}.lua`,
  `src/ServerScriptService/TwinMain.server.lua`, `src/StarterPlayerScripts/TwinTeleop.client.lua`.
- Checks: `ref/{fk_check,dyn_check,push_check,grasp_check,render_compare,live_check,episode_replay}.py`, results in `results/`.

Commits: 6b1deee, b839ce0, a145854, 5eb5f05, 5fdd6aa, 1ea8817, 4798e43.

# The game (Oct 25 2026): multi-task levels, phone-first

What a player sees: menu (PLAY, 10 s tutorial first; level cards with stars and best times; DAILY challenge; real-arm
card; "rounds played today" counter; gripper colours unlocked by stars; leaderboard: all / this level / daily) ->
round (timer, instruction, points, streak badge, joystick + UP/DOWN + twist + GRAB; tutorial: coach bubble + ghost
hand on the control to use; yellow outline = what to pick, green = where it goes) -> result card (stars pop in,
points with speed / clean-grab / combo breakdown, unlocks, "this round is now training data") -> NEXT / RETRY / LEVELS.
Juice: grab click + sparkles, swish + splash on entering the bowl, thud + camera shake on a drop, "oof" off the
table, chime + confetti on success (pitch rises with the streak), last-10-s ticks, haptics where the device has them.

- Levels = `tasks/*/task.json` (argon_task/1). `py -3 roblox/tools/gen_levels.py` regenerates
  `Twin/TaskData.lua`; `Twin/Levels.lua` adds the built-in tutorial (lift a cube) and titles/icons/par times. A new
  task.json is a new level after regenerating + syncing. Round length = limits.episode_s (clamped 20-75 s).
- Scenes: `Twin/LevelGen.lua` samples task.randomize (r/az/yaw, size_scale, size_m, mass, friction, min_sep);
  `SceneLoader` builds box/bowl/plate/stick (square-section bar drawn as a screwdriver: a lying cylinder rolls away
  and slips in the jaws)/lying cylinder/sphere; `collide:false` objects are painted zones.
- Judge: `Twin/TaskCheck.lua` = exact port of `argon_tasks.check` (200 shared vectors pass); `Twin/Round.lua` holds
  it for 0.5 s, emits juice events, logs grasps/drops/clean, fail_reason + fail_detail. Score: `Twin/Score.lua`
  (100 + speed bonus + 50 clean, combo x1.5/x2/.. cap x3; stars: 1 success, 2 under par, 3 under par + clean).
- Bots: `Twin/SkillOperator.lua` runs any task skeleton (FK picks the wrist roll so the jaws close across the
  object; placement servos the held object). Run mode over seeds: cube_in_bowl ~85 %, stack ~85 %, stick ~90 %,
  push ~80 %, two cubes ~60 %, tutorial 100 %. Headless: `node tools/runmode.mjs spikes/collect.luau
  '{"n":4,"minutes":10,"levels":"all"}'`; in game: TwinConfig.BOT_STATIONS / workspace attribute TwinBots.
- Traces: policy_io jsonl + meta.task (sampled sizes), task_name, level, obs_objects, tutorial, novice, daily, points,
  stars, grasps, drops, clean, fail_reason, fail_detail; ticks add `objects` (K x 9 task_spec slots), `obj_pose`,
  `grip_closed` (joint_server.py resamples them; cube_in_bowl keeps cube_pose/bowl_xy).
- Persistence (pcall'd): DataStore TwinPlayers_v1 (stars, best, points, rounds, skin), TwinGlobal_v1 (rounds per day),
  OrderedDataStores lb_points_v1 / lb_level_<id>_v1 / lb_daily_<day>_v1.
- Tests: `luau tests/levels_test.luau` (registry, sampling, no free wins, scoring: 4446 checks),
  `luau tests/taskcheck_test.luau` (vectors: 200 cases), `luau tests/touchmap_test.luau`,
  `.venv/Scripts/python tests/joint_server_trace_test.py`; Play mode: `KEEP_TWIN=1 PLAY=1 node tools/runmode.mjs
  spikes/play_smoke.luau '{"levels":["tutorial_lift","cube_in_bowl"]}'`; bots: `spikes/levels_bot.luau`.
  (luau = C:\Users\sarta\g4studio\backend\bin\luau.exe, run from roblox/.)
(Publishing: see the OWNER checklist at the top of this file.)

## F: `tasks/README.md`

# tasks/: one `task.json` per task (argon_task/1)

Contract: [spec/task_spec.md](../spec/task_spec.md). Executable form: [argon_tasks/](../argon_tasks) (pure numpy:
`load_task`, `validate`, `sample_scene`, `encode_obs`, `check`, `check_window`). The same file drives the MuJoCo twin,
the bot expert, trace generation, training, eval, the real arm (deploy/so101d) and the Roblox level.

## Add a task (no code)

1. Write `tasks/<name>/task.json` (by hand, or `real2sim` task_from_video from a video). Every object needs a fixed
   `xy` or a `randomize` entry (`r` + `az_deg`, optional `yaw_deg` | `yaw_rel_deg`, `size_scale` | `size_m`,
   `mass_kg`, `friction`). A place/push goal can name a target object instead of literal xy
   (`{"pred": "at", "a": "cube", "xy": "zone", "tol_m": 0.03}` with a `"role": "target"` plate, `"collide": false`).
2. `policy/.venv/Scripts/python -m argon_tasks validate tasks/<name>`
3. Expert in the twin (target >= 0.8): `policy/.venv/Scripts/python policy/expert.py --task tasks/<name> --n 100`
4. Traces: `policy/.venv/Scripts/python policy/task_traces.py --task tasks/<name> --n 4000 --workers 3`
   (~0.8-1.4 s per episode per worker; at least 3000 successful traces are needed, see the table)
   (-> `policy/out/task_traces/<name>/`)
5. Train (GPU only under a landlord lease; launch with the lease env, `attach_process` the printed pid):
   `cd policy && .venv/Scripts/python train_task.py --task ../tasks/<name> --traces out/task_traces/<name> --name <name>_v1 --steps 30000 --miss-weight 4 --device cuda`
6. Eval: `cd policy && .venv/Scripts/python task_eval.py --policy export/<name>_v1 --n 100 --noise clean` and
   `--noise real --noise-scale 0.5`
7. Real arm: `deploy/so101d/policy.py` `load_policy("<name>_v1")` + `task_obs(pol, q, object_poses)`; the export's
   config.json carries `task`, `task_spec` and `obs_frame` (`mjcf_base_xy`; deploy shifts footprint-frame poses).

## Conventions worth knowing

- Observation: `q` (6) + K = `limits.max_objects` slots `[x, y, z, sin yaw, cos yaw, sx, sy, sz, present]` in
  `objects` order. cube_in_bowl reduces to the 13-value policy_io v0 vector bit for bit (`argon_tasks.legacy_obs`,
  tested against policy/data.py and policy/runtime.py).
- Expert skills come from the skeleton only. The wrist rolls so the jaws close across the object's narrow side;
  placement servoes the held object (offset measured at lift); `place_in` spreads several objects in one container.
- Exports clamp observation dims that never varied in training (config `obs_const`; e.g. a target zone's yaw, fixed
  sizes) back to their training value when they stray more than 5 training stds. A focus scene that rotated the zone
  by 20 deg took pick_place_stick_v1 from 0.87 to 0.00 before (300-sigma input), 0.52 after; own scenes are unchanged
  (policy/tests/test_task_exports.py). To make a policy actually use such a dim, randomize it in task.json.
  Exports also declare `start_q` (every trace starts from scripted.home_q()).
- MuJoCo stick = square section with side = diameter: a round one slips out of the voxel-box fingers (0/8 grasps
  vs 8/8). Lying cylinders get rolling friction (condim 6).

## Numbers (MuJoCo twin, n = 100 held-out scenes, seeds 10000+)

Expert = policy/expert.py (the bot that writes the traces); "real" for the expert = 1 cm xy bias on every object +
2 deg joint offsets (its re-grasp aims at the same biased spot, hence the drop). Policy = K-slot 512x3 MLP
(policy/train_task.py), eval policy/task_eval.py: clean, and real x0.5 (object bias 5 mm, z 1.5 mm, 2 mm jitter, joint
offsets 1 deg, 1 tick latency, stale pose while held). Success = argon_tasks.check held 1 s, or at the end.

| task | expert clean | expert real | successful traces (incl. 10 % val) | policy export | clean | real x0.5 |
|---|---|---|---|---|---|---|
| cube_in_bowl | 0.94 | 0.46 | 3583 | cube_in_bowl_v1 | 0.80 | 0.72 |
| (legacy, same scenes) | | | Roblox traces | d_ens3_search via legacy_obs | 0.95 | 0.84 |
| stack_two_cubes | 0.88 | 0.35 | 5015 | stack_two_cubes_v1 | 0.66 | 0.51 |
| push_to_target | 0.89 | 0.56 | 1726 | push_to_target_v1 | 0.45 | 0.34 |
| pick_place_stick | 0.93 | 0.57 | 1769 | pick_place_stick_v1 | 0.87 | 0.72 |
| | | | 1769 x 3 seeds | pick_place_stick_ens3 | 0.82 | 0.80 |
| two_cubes_in_bowl | 0.96 | 0.34 | 3637 | two_cubes_in_bowl_v1 | 0.73 | 0.57 |
| place_on_cell (tic-tac-toe) | 0.74 * | | 2952 | place_on_cell_v1 | 0.40 | 0.35 |
| | | | 2952 x 3 seeds | place_on_cell_ens3 | 0.51 | 0.37 |

\* place_on_cell expert over its 4000 trace episodes (not the n=100 eval seeds). Placement error: median 6.9 mm, 90 %
within the 12 mm tolerance. A piece was knocked in 10 % of episodes, and 9 % were grasp misses from the supply strip.
For the v1 policy, among episodes where the cube was placed: median error 9.7 mm, 66 % within 12 mm, and a piece moved
in 7/100 episodes. Its biggest failure is never lifting the cube (33/100).

Data is the lever. Clean success by number of successful traces (90 % train, 10 % val):
- cube_in_bowl: 364 traces 0.11, 1802 traces 0.58, 3583 traces 0.80, 5356 traces 0.82.
- stack_two_cubes: 1684 traces 0.41, 3344 traces 0.61, 5015 traces 0.66.
- two_cubes_in_bowl: 1820 traces 0.50, 3637 traces 0.73.
- push_to_target does not scale: 1726 and 3471 traces both give 0.45. A 1024x4 net gives 0.39. It underfits
  (train L1 = val L1), and the policy stalls mid-push between the expert's re-push modes (3 pushes per episode on
  average).

For cube_in_bowl the legacy d_ens3_search (3 seeds, runtime grasp search, the Roblox trace set) is still better
on the task's own scenes, and it is what deploy runs today.

Symmetry-aware yaw features (config `yaw_sym`) went in after a probe at 360 episodes showed 0.16 against 0.11
without them. Everything after that uses them.

## F: `dashboard/README.md`

# dashboard

The page projected during the demo: REAL -> REAL2SIM -> GAME -> POLICY, one 1920x1080 stage scaled to the window.

```
cd dashboard
uv venv .venv --python 3.11 && uv pip install --python .venv/Scripts/python.exe aiohttp segno opencv-python-headless numpy
.venv/Scripts/python server.py --mock                 # rehearsal: every panel animated from data on disk
.venv/Scripts/python server.py --game-url <url>       # live
# open http://127.0.0.1:8800 full screen (F11)
```

Live camera: start the deploy loop with `--publish-port 8791` (deploy/scripts/run_policy.py). The dashboard never opens
the webcam; it pulls `/frame.jpg` and `/status.json` from the deploy process.

| panel | reads |
|---|---|
| REAL | deploy publisher `http://127.0.0.1:8791/{frame.jpg,status.json}` (annotated frame, cube/bowl estimates) |
| REAL2SIM | newest `real2sim/out/live/<run>/`: `run.json`, `iter_N.json` (`metrics.score` -> gap = 1 - score, `images.overlay_0` = real \| render side by side, `kept`) |
| GAME | first (meta) line of `roblox/out/traces*/*.jsonl`: rounds played = `source: human` rounds on every level (`human_*` = cube in bowl, `game_<level>_*` = other levels and the tutorial; level from `meta.level`/`meta.tutorial`, none = cube in bowl), per-level counts, leaderboard across levels; bots (`scripted`, `attract`) listed separately when `count_bots`; `traces_tmp` parts younger than 20 s = playing now. Test: `tests/test_game.py` |
| POLICY | one policy (`headline_eval` in config.json pins it; else the arm's policy; else the newest policy with both evals): its newest `export_<p>_clean_*` and `export_<p>_real[x<k>]_*` evals, episodes from `policy/export/<p>/config.json`, curve from `policy/out/scaling.json` (`{"points": [{"episodes", "success"}]}`, clean success) |
| real arm | newest real (not `--dry-run`) `deploy/runs/<ts>_*/ticks.jsonl` (tailed), `meta.json`, `summary.json`; success = operator y/n, else camera. `--show-dry-runs` also shows simulated dry runs, under a "SIM DRY RUN" header |

Everything in `config.json` (port, game URL, paths). In `--mock` the camera plays real2sim/data/ep0, real2sim replays the
g37flash spike, players/arm attempts/scaling curve are invented and the page says MOCK DATA.

# Appendix G: file manifest (every source file, first docstring / header comment)

| file | lines | purpose |
|---|---|---|
| `argon/__init__.py` | 8 | argon: one command for the real2sim2real loop (task -> twin -> data -> train -> eval/gate -> deploy -> close). python -m argon run --task cube_in_bowl --dry-run --rounds 2 python -m argon status Stdlib only: every stage runs as a subprocess in its component's own venv (policy/, real2sim/, deploy). HARNESS.md is the operator guide; runs/<task>/manifest.json (schema argon_run/1) is what the dashboard reads. |
| `argon/__main__.py` | 3 |  |
| `argon/cli.py` | 132 | python -m argon run/status (HARNESS.md). |
| `argon/close_round.py` | 43 | loop/close_loop.py's decision (reality vs twin -> focus scene -> sim config) for one argon round, written into the round directory instead of loop/rounds/. Runs in the policy venv: policy/.venv/Scripts/python -m argon.close_round --session <attempt dir> --r2s real2sim/out/live/<run> --round r2 --out runs/<task>/r2 [--prev runs/<task>/r1/sim_config.json] |
| `argon/oracle_boxes.py` | 52 | Oracle task boxes for a SYNTHETIC session (truth.json + camera_true.json): the pixel box of every task object in the first and the last frame, in run_bench --task-boxes form. With them (and --no-keypoints --iters 0) real2sim builds the twin from its silhouette fitters and the joint-log anchors with zero Gemini requests: the dry-run twin when the Gemini quota is gone. Real sessions never use this (no truth); they get the task-aware detection request. python -m argon.oracle_boxes <session dir> <out boxes.json> |
| `argon/pipeline.py` | 673 | The round loop. One Run per task directory (runs/<task>[-dry]/), one manifest (argon_run/1), cached stages. Per round r (r1, r2, ...), each stage cached in runs/<task>/<r>/ and in the manifest: session the attempt the round learns from: r1 = --video <session>, else a stub probe (dry run: synthetic bench; --real: the bench), r>1 = the previous round's deploy attempt (stage 7, closing the loop) twin real2sim run_bench on that session -> scene.json (live r2s_live/1 under real2sim/out/live/<run>) close reality vs twin -> focus scene -> sim_config.json (loop/close_loop.py's decisions) data bot trac |
| `argon/sample_places.py` | 22 | Object placements for a synthetic task session drawn from the task's own distribution (argon_tasks.sample_scene), printed as real2sim synth_task_bench --place arguments (deploy footprint frame: MJCF base xy - 16.8 mm in x). policy/.venv/Scripts/python -m argon.sample_places <task.json> <seed> |
| `argon/sim_eval.py` | 75 | Twin eval of one export on a round's focus scenes (loop/eval_focus.py's scenes and held-out seeds), for several noise settings in one process pool. Runs in the policy venv: policy/.venv/Scripts/python -m argon.sim_eval --config runs/<task>/<r>/sim_config.json --policy policy/export/<name> --noise clean real:0.5 [--n 50] [--workers 4] --out runs/<task>/<r>/eval_<name>.json noise "real:0.5" = policy/eval.py's "real" set (pos, offset, latency, stale) with the pos/offset sigmas x0.5. Prints one JSON line: {policy, n, results: {"clean": success, "realx0.5": success}, outcomes, wall_s}. |
| `argon/tasks.py` | 323 | Task resolution (spec/task_spec.md, argon_task/1) and the per-task backend. --task accepts: a task name (tasks/<name>/task.json), a path to a task.json, or an instruction in quotes (perception's task_from_video on --video writes the task.json). cube_in_bowl runs today's code paths (loop/, policy/, deploy/) and falls back to the spec's own cube_in_bowl example when tasks/cube_in_bowl/task.json does not exist yet. |
| `argon/util.py` | 112 | Paths, interpreters, atomic JSON, subprocess runner with a log file. |
| `argon_tasks/__init__.py` | 20 | argon_tasks: the executable form of spec/task_spec.md (argon_task/1). Pure numpy, no torch, no MuJoCo. import argon_tasks as at task = at.load_task("tasks/cube_in_bowl") # validated dict (ValueError lists every problem) scene = at.sample_scene(task, np.random.default_rng(0)) # scene_spec v0 instance, objects resting on the table x = at.encode_obs(task, q, at.poses_from_scene(scene)) # (6 + 9K,) policy observation res = at.check(task, state) # {"ok", "per_predicate": [...]} at.legacy_obs(task, x) # cube_in_bowl only: the 13-value policy_io v0 vector CLI: python -m argon_tasks validate <task.jso |
| `argon_tasks/__main__.py` | 38 |  |
| `argon_tasks/check.py` | 195 | The one success checker (MuJoCo twin = truth, Roblox game = score, real arm = perception estimate). state = { "objects": {id: {"xyz": [3], "yaw_deg": float / "quat": [w, x, y, z], optional "v": linear speed m/s, optional "size_m" (task convention; default the task's)}}, "gripper": {"open": bool, optional "site_xyz": [3] (gripper site), optional "holding": [ids]}, (optional) "initial": {id: {"xyz": [3]}}, (for not_moved) } Absent objects (not in state["objects"]) fail every predicate that names them, unless the goal has "optional": true (then it holds; per_predicate says "absent"). Predicates ( |
| `argon_tasks/scene.py` | 200 | Scene sampling and the K-slot policy observation. Geometry conventions (shared by the twin, the game and the checker): slot size [sx, sy, sz] = the object's extents in its own frame (x along its yaw): box/plate [x, y, z]; cylinder upright [d, d, h], lying "axis": "x" [h, d, d], "axis": "y" [d, h, d]; sphere [d, d, d]; bowl [d, d, h] (d = inner diameter at the rim); stick [length, d, d] (lies along x). An object resting on the table has its centre at z = sz / 2 (scene_spec xyz is the centre). Legacy cube_in_bowl obs (spec/policy_io.md, 13 values) = q, cube slot xyz, cube sx, bowl slot xy, bowl  |
| `argon_tasks/spec.py` | 151 | argon_task/1 vocabulary, loading and validation (spec/task_spec.md is the contract; this is its executable form). Argument layout of the JSON entries (every name below is a key of the entry): goal: inside {a, b} / on_top {a, b} / at {a, xy, tol_m} / near {a, b, d_m} / lifted {a, h_m} / grasped {a} upright {a, tol_deg} / yaw {a, deg, tol_deg} / not_moved {a, tol_m} / stacked {objs: [bottom, ..., top]} skeleton: pick {obj} / place_in {obj, target} / place_on {obj, target} / place_at {obj, xy} / push_to {obj, xy} release / home / wait {s}; any skill may add "approach": "top" / "side" Any goal may |
| `argon_tasks/tests/__init__.py` | 0 |  |
| `argon_tasks/tests/boundary_cases.py` | 179 | Hand-built check() cases for every predicate in spec/task_spec.md, each with its intended outcome ("want"): pairs 0.5 mm / 0.5 deg either side of every tolerance, yaw wrap-around at +-180 and symmetry periods, rotated containers/bases, missing inputs, implicit gripper_open / settled. Written into vectors.json by test_argon_tasks.py --write-vectors, which asserts check() gives every "want" first. |
| `argon_tasks/tests/test_argon_tasks.py` | 296 | argon_tasks unit tests: validation, scenes, K-slot obs (+ byte-compat with policy_io v0), every predicate, and the shared test vectors (vectors.json, what the Luau port must reproduce). policy/.venv/Scripts/python -m pytest argon_tasks/tests -q policy/.venv/Scripts/python argon_tasks/tests/test_argon_tasks.py --write-vectors # regenerate vectors.json |
| `dashboard/adapters.py` | 548 | Read-only adapters: each turns one writer's files into the small dict the page draws. Every adapter is tolerant: a missing directory, a half-written file or an unknown key gives `{"ok": False, ...}` or skips that item, never an exception. Files are re-read only when their mtime changes. |
| `dashboard/mock.py` | 156 | --mock: animate all four panels from data already on disk, so the demo can be rehearsed without hardware. real2sim replays the g37flash spike loop one iteration at a time; the game adds an episode every few seconds on top of the real trace counts (invented player names); the policy panel shows the real headline eval and, until policy/out/scaling.json exists, an invented scaling curve; the real-arm status cycles through scripted attempts; the camera plays the recorded real SO-101 episode (real2sim/data/ep0) with the cube detected by colour. Everything synthetic is labelled MOCK on the page. |
| `dashboard/server.py` | 199 | Demo dashboard: one page, four panels (REAL -> REAL2SIM -> GAME -> POLICY), polled JSON. cd dashboard && .venv/Scripts/python server.py # live: reads the repo's output files cd dashboard && .venv/Scripts/python server.py --mock # rehearsal: animates every panel from data on disk open http://127.0.0.1:8800 Config: dashboard/config.json (game URL, port, source paths). --config / --port / --game-url override it. Endpoints: / (page), /api/state (all panels), /api/camera.jpg, /api/qr.svg, /img/r2s/<i>/<kind>. |
| `dashboard/shots.py` | 61 | Real-time screenshots of the running dashboard over Chrome DevTools (headless --virtual-time-budget screenshots run timers ahead of painting, so they can show a badge from one view over the image of another). .venv/Scripts/python shots.py http://127.0.0.1:8800/ shots/live 2.0 4.6 7.4 10.2 # seconds after load |
| `dashboard/static/index.html` | 528 | page: #0d0d0d; --panel: #1a1a19; --panel-2: #222220; --ring: rgba(255,255,255,0.08); ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781; --grid: #2c2c2a; --axis: #383835; s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #9085e9; good: #0ca30c; --good-text: #2fc42f; --bad: #d03b3b; --warn: #fab219; font: "Segoe UI Variable Display", "Segoe UI", Inter, system-ui, -apple-system, sans-serif; |
| `dashboard/static/runs.html` | 114 | page: #0d0d0d; --panel: #1a1a19; --panel-2: #222220; --ring: rgba(255,255,255,0.08); ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781; --grid: #2c2c2a; good: #0ca30c; --good-text: #2fc42f; --bad: #d03b3b; --warn: #fab219; --sim: #7aa7ff; font: "Segoe UI Variable Display", "Segoe UI", Inter, system-ui, -apple-system, sans-serif; mono: ui-monospace, "Cascadia Mono", Menlo, Consolas, monospace; |
| `dashboard/tests/test_game.py` | 67 | GAME panel counting over a fixture folder with every trace naming the game writes. cd dashboard && .venv/Scripts/python tests/test_game.py |
| `deploy/scripts/armd.py` | 172 | Arm daemon: connect once, hold the follower, execute small supervised motions from HTTP commands. Why a daemon: RealRobot() turns torque OFF for a moment while it primes the hold, so reconnecting from a new script while the arm is out in the air drops it. One process owns the bus for the whole session. python deploy/scripts/armd.py [--port 8793] (owner present; torque ON from start until /release) curl -s 127.0.0.1:8793/state curl -s -d '{"cmd":"xyz","xyz":[0.20,0,0.18]}' 127.0.0.1:8793/cmd fingers-down IK to the jaw midpoint (base frame, m) curl -s -d '{"cmd":"joints","owner":[...6]}' 127.0.0 |
| `deploy/scripts/calib_camera.py` | 126 | Camera calibration WITHOUT Gemini: printed ArUco board (scripts/make_board.py) at a ruler-measured spot. Live (camera only, the arm is not touched): python deploy/scripts/calib_camera.py --capture window keys: i = save an INTRINSICS view (hold the board TILTED 20-45 deg at different places/distances; 8-12 views; the more of the image the board covers across views, the better) p = save the PLACED view (board flat at its measured spot, nothing on top, arm out of the way) c = compute + save config/camera.json q = quit Offline (re-run on saved images): python deploy/scripts/calib_camera.py --from  |
| `deploy/scripts/calib_from_arm.py` | 107 | Camera from the arm itself (the real2sim pipeline): Gemini points at jaw tips / wrist / elbow / shoulder / base on ~15 frames of a recorded session, FK of the logged joints (through the joint map) gives the 3D points, a PnP + focal fit gives the camera. Held-out frames give the reprojection error. If config/camera.json (ArUco) exists, the same keypoints are also scored through the ArUco camera for every joint-map candidate (a second, camera-fixed check). Runs in the real2sim venv (google-genai, mujoco 3.17); costs one Gemini request per frame (free tier: 20/day/model). set R2S_MODEL=gemini-4-a |
| `deploy/scripts/camserver.py` | 64 | Camera frame server, launched from Terminal.app so macOS camera permission applies. GET /frame/<i>.jpg, /list. |
| `deploy/scripts/fit_sag.py` | 156 | Fit the arm's gravity-sag compliance k (rad/Nm) for robot.GravityComp, from small supervised moves through armd. The arm must be held by deploy/scripts/armd.py (owner present; armd owns the bus and does every safety check: calibrated limits, table clearance along the whole path, abort-and-hold on tracking lag or sustained load). This script only asks armd for fingers-down jaw positions and reads back what armd measured: per move, the commanded owner joints (armd's q - err), the measured ones, and FK of both. Model: the servos deflect by dq = -k * tau_gravity(q_cmd) on shoulder_lift / elbow_fle |
| `deploy/scripts/look_yellow.py` | 24 | Wrist-camera frame from camserver (127.0.0.1:8792), find the largest yellow blob (screwdriver handle): centroid, axis angle, size. Saves an annotated jpg. |
| `deploy/scripts/make_board.py` | 42 | Printable ArUco board (US Letter, 300 dpi, 100 % scale) matching config/bench.json "board". python deploy/scripts/make_board.py -> deploy/runs/aruco_board_letter.png Print at 100 % ("actual size", no fit-to-page). Then MEASURE with a ruler: the width of the top marker row from the left edge of marker 0 to the right edge of marker 3 (nominal 190 mm) and put marker_length_m = width / 4.75 into bench.json (separation stays 0.25 x marker length). Tape it flat; matte paper avoids glare. |
| `deploy/scripts/make_plate_3mf.py` | 97 | One H2C plate with every bench print: two-colour ArUco board, blue 110 mm bowl, two red 38 mm cubes. python deploy/scripts/make_prints.py && python deploy/scripts/make_plate_3mf.py -> deploy/assets/print/bench_plate_h2c.3mf Open it in Bambu Studio (GUI), pick the H2C, and map filaments 1-4 = white, black, red, blue to the AMS slots. Per-part filament assignments live in Metadata/model_settings.config (board = one object with a white and a black part). Layout (bed mm): board x 57-267 y 150-310, bowl centre (100, 70) outer d 120, cubes at x 195 / 253, y 30; >= 20 mm between parts, everything ins |
| `deploy/scripts/make_prints.py` | 120 | 3D-printable bench objects matched to the training distribution (policy/sim.py sample_scene) + the ArUco board. python deploy/scripts/make_prints.py -> deploy/assets/print/*.stl - cube_38mm.stl: red. Training cubes are 25-50 mm, 20-100 g; ~40-50 % infill gives ~40 g. - bowl_d130_h30.stl: blue. Sim bowl = straight-walled open cylinder, inner diameter at the rim 100-160 mm, 30 mm tall, 5 mm wall (SceneLoader.makeBowl / policy/sim.py make_spec). 130 mm inner, 30 mm tall, 5 mm wall, 3 mm floor. - aruco_board_white.stl + aruco_board_black.stl: ONE object, two parts (load both, "load as single objec |
| `deploy/scripts/make_stones.py` | 46 | Curling stones for the bench (P2S): 40 mm discs, 14 mm tall, 1.5 mm bottom chamfer (glides instead of catching), 2 mm top chamfer, shallow 20 mm top dimple (a grasp/visual centre). Print 100 % infill, flat side down. python deploy/scripts/make_stones.py -> deploy/assets/print/stone_40mm.stl |
| `deploy/scripts/make_ttt_grid.py` | 84 | Printable tic-tac-toe grid for the bench: US Letter, 300 dpi, PRINT AT 100 % (no fit-to-page). 3x3 cells of 45 mm (tasks/place_on_cell/task.json: cell centres x .200/.245/.290, y -.045/0/.045, robot base frame), 3 mm lines, a 20 mm check bar to verify the print scale, a corner mark for the grid's outer corner nearest the robot on its right (r0c0 corner: base x .1775, y -.0675) and an arrow toward the robot. python deploy/scripts/make_ttt_grid.py [--out deploy/assets/ttt_grid_letter.png] |
| `deploy/scripts/make_ttt_pieces.py` | 61 | Tic-tac-toe set that doubles as curling stones (one plate, three filaments: board white, O teal, X red). - Board: 3x3 x 45 mm cells (tasks/place_on_cell), 2 mm plate + 1.2 mm raised lines (3.2 mm tall). - Pieces: 40 mm disc x 10 mm, 1.5 mm bottom chamfer (glides as a curling stone), 1 mm top chamfer, plus a 3 mm raised symbol: O = ring r 9-14 mm, X = two 28 x 5 mm bars. 13 mm tall in all; the jaws grasp the round base from any angle. python deploy/scripts/make_ttt_pieces.py -> deploy/assets/print/ttt_{board,O,X}.stl |
| `deploy/scripts/mirror.py` | 90 | Live mirror: the real arm's joints (read-only bus, torque OFF, owner moves it) -> MJCF radians via the joint map -> the Roblox twin (roblox/tools/joint_server.py --source leader, Workspace TwinDrive = "live") and, with --overlay, a camera window with the arm silhouette of each joint-map candidate drawn on the live image. 1) roblox\.venv\Scripts\python roblox\tools\joint_server.py --source leader --port 8767 (separate console; in Studio set Workspace attributes TwinServer = http://127.0.0.1:8767 and TwinDrive = live) 2) .venv-so101/Scripts/python.exe deploy/scripts/mirror.py --overlay [--candid |
| `deploy/scripts/perceive.py` | 165 | Perception check / HSV tuning on the live camera or an image. Prints cube_xyz, bowl_xy, bowl_d at ~5 Hz and draws the masks, the deprojected cube centre and the fitted rim circle reprojected (a wrong rim_height or camera shows as a circle that does not sit on the rim). Click a pixel to print its HSV. Keys: s = save frame, q = quit. python deploy/scripts/perceive.py (live camera from bench.json, camera model config/camera.json) python deploy/scripts/perceive.py --image deploy/runs/<calib>/placed.png --no-window --truth 0.25 -0.05 python deploy/scripts/perceive.py --task tasks/<name>/task.json [ |
| `deploy/scripts/perception_accuracy.py` | 115 | Expected perception accuracy on the bench, by Monte Carlo on the synthetic bench (MuJoCo renders, real code path). Each trial: the board lies at its TRUE spot; the calibration uses the owner's MEASURED numbers, perturbed by ruler errors (placement sigma 2 mm, yaw 1 deg if one distance is measured / 0.4 deg with both, print-scale 0.3 %); image noise 2 grey levels. Then cube + bowl at random workspace spots are rendered and perceived with the estimated camera. Two intrinsics cases: focal-only from the placed view (one image) and known intrinsics (calibrateCamera views). Also fits `centroid_heigh |
| `deploy/scripts/perception_task_eval.py` | 139 | Task-general perception (so101d.taskperception) vs ground truth on the synthetic bench, no hand-tuned colours. For each scene: objects at random poses, setup on one frame (boxes from the oracle detector = projected truth, or Gemini with --gemini, one request per scene), then the objects are moved to new random poses (and the arm is swept through the view) and tracked with the same appearance models. Reports xy / yaw errors per object and ms per frame. python deploy/scripts/perception_task_eval.py [--task tasks/<name>/task.json / --bench tonight] [--scenes 6] [--moves 4] [--gemini] [--out deplo |
| `deploy/scripts/preflight.py` | 167 | Bench preflight: is everything ready for a live run? READ-ONLY (never writes a motor register, never moves). python deploy/scripts/preflight.py [--policy d_ens3_search] [--tests] PASS / WARN / FAIL per check, exit code 1 on any FAIL. Checks: config overlay, follower bus (6 motors, EEPROM calibration == calibration file, voltage, temperature, torque, error flags), bus not held by another process, camera (frames, size, rate), camera calibration file vs camera size, board, joint map, policy export loads and acts, optional deploy tests. |
| `deploy/scripts/read_pose.py` | 181 | Torque-OFF bench readout: verifies / fits the joint map with no motion command. Read-only on the bus. The owner moves the limp arm by hand; each capture is the median of the last second (refused if a joint moved > 0.5 deg). Type a command + Enter in this console, or write {"seq": n, "cmd": "..."} to --control-file. folded arm folded against its mechanical stops (corpus: lift -103.7, elbow +96.5 in the model frame) closed / open gripper fully closed (jaws touching) / fully open touch <x> <y> FIXED-jaw tip touching the table at base-frame x, y (metres). x = 0.048 + ruler distance from the base F |
| `deploy/scripts/record_session.py` | 104 | Record joints + camera frames at 30 Hz for camera/offset calibration. Default is HAND-GUIDED: torque off, read-only bus, the owner moves the limp arm through its workspace for ~20 s (no motion command exists in this mode). .venv-so101/Scripts/python.exe deploy/scripts/record_session.py --seconds 20 [--mirror] --mode teleop --leader-port COM4 --leader-id leader_1 leader arm drives the follower (owner present; uses the guard, speed 0.5); needs the leader's own LeRobot calibration file. Output: deploy/runs/<ts>_session/{frames/0000.jpg..., joints.jsonl (t, owner_deg, frame), meta.json} Use it for |
| `deploy/scripts/run_policy.py` | 274 | Run the cube-into-bowl policy on the SO-101 (or a mock), 30 Hz, guarded, every tick logged. Dry run (no hardware, no camera): synthetic bench rendered by MuJoCo from a high camera, camera calibrated from a synthetic ArUco view (so calibration error propagates like on the bench), mock servo, kinematic cube: python deploy/scripts/run_policy.py --dry-run --policy stub --attempts 5 Dry run on a recorded / still image (perception on real pixels, mock robot, cube never moves): python deploy/scripts/run_policy.py --dry-run --image deploy/runs/<calib>/scene.jpg --camera-model deploy/config/camera.json |
| `deploy/scripts/synth_session.py` | 48 | Make a synthetic recorded session (frames + owner-deg joints) whose TRUE joint map is a chosen candidate, plus the true camera as camera.json, to test record -> verify_offsets / calib_from_arm offline. python deploy/scripts/synth_session.py --truth simfix_official_v3 [--frames 150] |
| `deploy/scripts/tictactoe.py` | 347 | Tic-tac-toe against the SO-101: Argon reads the board and picks the move, the arm places a brown cube there. python deploy/scripts/tictactoe.py --dry-run [--games 1] [--first robot/human] [--seed 0] synthetic bench: the grid is rendered on the table, the "human" (a script: minimax or random) drops blue discs, the board is read by the oracle (0 Gemini calls), the arm plays every move with run_policy's Session python deploy/scripts/tictactoe.py --frame-only img.jpg [--camera deploy/config/camera.json] [--history r1c1:X,...] one Argon board read + move on a saved scene image (prints model, board, |
| `deploy/scripts/verify_offsets.py` | 128 | Gemini-free joint-map check: rendered arm silhouette (each joint-map candidate, through the ArUco camera) vs the moving arm in a recorded session (background subtraction against the session's median frame). python deploy/scripts/verify_offsets.py deploy/runs/<ts>_session [--fit] [--frames 25] Prints mean IoU per candidate (higher = better; a wrong wrist_roll or pitch offset shows as a clear drop), writes <session>/offsets_check.jpg (contours per candidate on sample frames) and, with --fit, candidate "silhouette_fit" (Nelder-Mead on pan/lift/elbow/wrist_flex/roll offsets from the best candidate |
| `deploy/scripts/wristcam_appearance_eval.py` | 176 | 2D appearance robustness of the task perception on REAL wrist-camera frames (deploy/runs/real_wristcam_oct25/, the Oct 25 bench screwdriver grasp: Innomaker 1280x720, real room light, white bubble wrap + pink damask cloth, black jaws). No Gemini: one setup box per run comes from the reference mask of the setup frame. Reference per frame (the yardstick, checked by eye on the overlay sheet): the handle, i.e. pixels clearly yellow in Lab (b* > 141, a* < 135: the pink cloth and the red annotation strokes have a* > 135), closed, largest component; the thin metal shaft is not in it (no colour signal |
| `deploy/so101d/appearance.py` | 253 | Per-object colour models fitted automatically from one labelled frame (no hand-tuned thresholds). fit(img, mask, space): a 3D colour histogram of the object's pixels vs the rest of the frame (other objects, arm, table, board), smoothed, turned into P(object / colour) and looked up per pixel. The threshold is the one that best reproduces the fitting mask on the fitting frame (IoU sweep). Colour spaces: "lab" (default): L 8 x a* 32 x b* 32 bins, a*/b* at 4 units over [64, 192): pale objects under room light (a cream screwdriver handle next to a salmon-pink device on a pink cloth) separate in a*/ |
| `deploy/so101d/bench.py` | 60 | Shared bench helpers: config loading, session folders, read-only joint logging, robot-silhouette rendering. |
| `deploy/so101d/camera.py` | 147 | Frame sources with one call: grab(q_model) -> BGR image (q only matters for the synthetic source). |
| `deploy/so101d/gem.py` | 149 | Minimal Gemini REST client for the bench (no google-genai dependency; stdlib only, macOS and Windows). Key: GEMINI_API_KEY in the environment, else the line `GEMINI_API_KEY=...` of the file named by GEMINI_ENV_FILE (default: the same .env real2sim reads, when it exists). The key goes in a request header, never in a URL or a log. Models are roles (same names as real2sim/r2s/gemini.py): "small" = R2S_MODEL_SMALL (default gemini-3.8-flash) boxes the task objects on one frame at setup; "main" = R2S_MODEL_MAIN (default gemini-4-argon; R2S_MODEL legacy override) does the lost-object re-detection rea |
| `deploy/so101d/guard.py` | 150 | Workspace + rate guard. ONE enforcement point for every target sent to the arm, including retreats. Rules (model frame, metres / radians): - joint limits of the official model, minus a margin; - per-tick step cap relative to the LAST SENT target (and to the measured pose, so a lagging arm is not wound up); 1 deg/tick inside the slow zone (fingers within `slow_zone_m` of the table); - wrist_roll is an orientation: a commanded roll is resolved to the short way round, and the net excursion from the attempt's start pose is capped (motion back toward the start is always allowed); - table: arm links |
| `deploy/so101d/jointmap.py` | 60 | LeRobot so_follower degrees (this arm's calibration) <-> MJCF radians (official so101_new_calib.xml). model_deg = sign * owner_deg + offset_deg (five arm joints) model_rad = closed_rad + gripper/100 * (open_rad - closed_rad) (gripper, LeRobot 0 = closed, 100 = open) Numbers live in deploy/config/joint_map.json. `active` names one of `candidates`; each candidate carries `measured` per joint and a top-level `verified` that only the bench procedure (scripts/read_pose.py + the mirror / overlay check) may set. Live runs refuse an unverified map unless the operator passes --allow-unverified (reduced |
| `deploy/so101d/kb.py` | 52 | Non-blocking single-key console input with the msvcrt interface (kbhit / getwch / getwche) on every platform. Windows: msvcrt itself. macOS / Linux: the terminal is put in cbreak mode (keys arrive without Enter, Ctrl-C still interrupts) and restored at exit. Returns None when stdin is not a terminal (e.g. started over a non-tty ssh), so callers fall back to the control file; `require=True` makes that a hard error for operator-facing runs. |
| `deploy/so101d/kin.py` | 102 | SO-101 kinematics on the official so101_new_calib.xml (the model real2sim, roblox and the policy all use). Frame = scene_spec base frame: origin at the base footprint centre on the table, x forward, y left, z up (MJCF base frame shifted by BASE_ORIGIN_MJCF). Joint vector q is MJCF radians in the order of JOINTS. Link geometry for the guard = convex-hull vertices of each collision mesh (exact for the min-height and max-radius tests, which are attained at extreme points). Importing this module never opens hardware. |
| `deploy/so101d/operator.py` | 72 | Operator channel: keyboard (Windows console, non-blocking) AND a control file, polled every tick. The control file exists because an agent supervising a run cannot type into the process. Keys / commands: s or SPACE STOP: hold, then retreat slowly to the start pose and hold there (torque stays on) p pause (hold where it is) g go / resume / start the next attempt y / n score the attempt success / failure q end the session (arm holds at the start pose, torque ON) T torque off: ONLY with the owner supporting the arm; asks for T again within 3 s Control file JSON: {"seq": <int, increase for each ne |
| `deploy/so101d/perception.py` | 292 | Fixed-camera perception: camera model, ArUco board calibration (no Gemini), colour detection of the cube and the bowl, deprojection onto the table. Base frame = scene_spec frame (metres). Importing never opens a camera. Board convention (printed by scripts/make_board.py): OpenCV GridBoard, origin = outer top-left corner of marker 0, board x = along the top marker row (page right), board y = down the page, z into the page. Place the sheet flat, printed side up, TOP marker row nearest the robot and parallel to the base's front face; reading the page from the far side of the table, page-right = r |
| `deploy/so101d/policy.py` | 250 | Policy API (spec/policy_io.md) + a scripted IK stand-in + temporal ensembling. A trained export lives in policy/export/<name>/ with runtime.py defining `Policy.load(dir)` and `act(obs) -> (15, 6)` (absolute MJCF-radian targets at 30 Hz starting at the next tick). obs = {q (6,), cube_xyz (3,), cube_size (1,), bowl_xy (2,), bowl_d (1,)}. `load_policy("stub")` returns ScriptedPolicy, which is useful on its own: a perception- driven pick-and-place through damped-least-squares IK on the official model (no learning), the bench baseline. |
| `deploy/so101d/publish.py` | 126 | Optional camera publisher for the demo dashboard (run_policy.py --publish-port N; off by default). Two processes cannot both own a Windows webcam, so the deploy loop shares what it already has. A background thread, at `hz`, takes the newest frame (pushed by the control loop with `offer`, or read from a LiveCamera's own grab thread between attempts), draws the cube and bowl colour detections on it, JPEG-encodes it, and serves: GET /frame.jpg newest annotated frame GET /status.json {t, attempt, tick, phase, cube_ok, bowl_ok, cube_xyz, bowl_xy, frames} It never blocks the loop: `offer` stores ref |
| `deploy/so101d/record.py` | 84 | Per-attempt session recording for real2sim (run_policy.py --record). Each attempt becomes a folder in deploy/scripts/record_session.py's layout, which real2sim/scripts/run_bench.py reads: frames/NNNN.jpg, joints.jsonl (t, frame, owner_deg, cmd_owner_deg, tick, phase, est), meta.json, camera.json [+ camera_true.json, truth.json on the synthetic bench]. `cmd_owner_deg` is what the robot was told to do this tick (owner frame), so real2sim's grasp / release anchors come from the robot's own gripper commands ("commanded more closed than measured" = blocked by the cube). The 30 Hz loop only puts (fr |
| `deploy/so101d/robot.py` | 213 | Robot back-ends with one interface, all in the OWNER frame (LeRobot degrees, gripper 0-100): read() -> (6,) owner deg write(target (6,) owner deg) loads() -> dict or None hold() goal := measured pose, torque stays ON close(torque_off=False) RealRobot opens LeRobot 0.6.1 SOFollower on the bench (import is lazy: importing this module never touches a port). Torque-on is primed: torque off -> read present -> Goal := present -> configure() (which re-enables torque), so the arm does not jump. Nothing here disables torque except close(torque_off=True), which only an explicit operator command reaches. |
| `deploy/so101d/runner.py` | 321 | The 30 Hz control loop. One tick: read joints (owner deg) -> model rad -> grab frame -> perceive (cube/bowl, FK-attached cube while gripped) -> policy chunk (every `replan_every` ticks) -> temporal ensemble -> guard.project (step caps, roll, table, reach) -> load check -> write (owner deg) -> log -> operator poll -> sleep to the next tick Failure handling, by design: - guard refusal : the previous target is re-sent (arm HOLDS, torque on); after `max_consecutive_refusals` the attempt ends "held" and the arm stays put until the operator says go/stop. - load stop : goal := measured pose (torque o |
| `deploy/so101d/synth.py` | 164 | Synthetic bench for --dry-run and the perception accuracy study: MuJoCo renders the official SO-101, a table, a brown cube, a blue bowl and (optionally) the printed ArUco board, from a camera given as a CameraModel. The world also plays the cube kinematically (it follows the jaws while they are closed around it, drops to the table when they open). Works with plain MjModel XML (no MjSpec), so it runs in .venv-so101 (mujoco 3.13) as well as newer versions. |
| `deploy/so101d/synth_task.py` | 269 | Synthetic bench for ANY argon_task/1 task: MuJoCo renders the official SO-101, the table, the printed ArUco board and every task object (box, plate, cylinder, sphere, straight-walled bowl, stick; a screwdriver = coloured handle + grey shaft) at known poses, from a CameraModel. Manipulated objects are mocap bodies that follow the jaws while they are closed around them (kinematic, as synth.World does for the cube). Ground truth for perception and real2sim tests. Plain MjModel XML (no MjSpec): runs in deploy's .venv-so101 (mujoco 3.13) and in real2sim's venv. |
| `deploy/so101d/taskobj.py` | 135 | Geometry of argon_task/1 objects (spec/task_spec.md): extents, resting height, yaw symmetry, silhouette points, policy observation slots, scene_spec mapping and colour names. Shared by deploy perception, the synthetic bench and real2sim (which imports deploy/so101d). numpy only. |
| `deploy/so101d/taskperception.py` | 622 | Task-general fixed-camera perception for any argon_task/1 task (spec/task_spec.md), no hand-tuned colour ranges. Setup (once, on a camera frame): a detector boxes every task object (GeminiDetector: one request for all objects; OracleDetector: projected ground truth on the synthetic bench; or boxes passed in), GrabCut turns each box into a pixel mask, and an appearance model (appearance.Appearance: HSV histogram vs the rest of the frame, threshold fitted on that frame) is built per object. The setup serialises to JSON (save_setup / load_setup): a bench session can be replayed without another re |
| `deploy/so101d/taskscripted.py` | 222 | Scripted skeleton player for any argon_task/1 task (dry runs on the synthetic bench, a stand-in for a learned task export): it reads the K-slot observation runner.Session builds with policy.task_obs (deploy footprint frame) and plays the task's skeleton (pick, place_in, place_on, place_at, push_to, release, home, wait) as jaw-midpoint waypoints, with ScriptedPolicy's IK (fingers down) and joint-space ramps. No per-task code. cfg = {"task_spec": task, "obs_frame": "footprint"}, so runner feeds it slots and load_policy's frame shift is not involved. The wrist roll stays at the start pose's: the  |
| `deploy/so101d/ttt.py` | 232 | Tic-tac-toe against the arm: board geometry (tasks/place_on_cell/task.json), rules, perfect-play minimax, the Argon board read (one MAIN-role Gemini request per robot turn: scene frame with the cells drawn on it + a top-down crop of the grid + the game history -> {board, move, why}) and its local validation. Cells are indexed 0-8 = r0c0, r0c1, r0c2, r1c0, ..., r2c2: row i along x (r0 nearest the robot), column j along y (c0 = robot's right, y -0.045). Marks: the robot plays "X" (its cubes), the human "O" (discs) unless told otherwise. |
| `deploy/tests/test_gravity.py` | 93 | Gravity sag (dry-run MockRobot) and the optional real-arm gravity compensation. Run: python deploy/tests/test_gravity.py |
| `deploy/tests/test_load.py` | 19 | Present_Load decoding: signed (LeRobot-decoded) and raw sign-magnitude readings give the same magnitude. |
| `deploy/tests/test_stack.py` | 171 | Offline checks of the safety-relevant behaviour (no hardware). Run: python deploy/tests/test_stack.py |
| `deploy/tests/test_start_pose.py` | 37 | Attempts start where the policy was trained to start: task exports from the expert home pose, legacy exports from the bench start pose, any export from its config "start_q" when it declares one. |
| `deploy/tests/test_task_policy.py` | 70 | argon_task/1 exports on the deploy side: load_policy loads a task export (its own runtime.py), wraps it in the frame shift, task_obs builds the K-slot obs from deploy-frame poses, and present slots (only) are shifted. Run: python deploy/tests/test_task_policy.py |
| `deploy/tests/test_taskperception.py` | 101 | Task-general perception on the synthetic bench (no hardware, no Gemini: the oracle detector boxes projected truth). Run: python deploy/tests/test_taskperception.py |
| `loop/close_loop.py` | 231 | Close the real2sim2real loop for one policy rollout: real attempt (deploy run_policy --record) + the twin real2sim built from it (real2sim run_bench) -> an updated sim config for trace generation -> bot traces in the MuJoCo twin -> the retrain / redeploy commands. policy/.venv/Scripts/python loop/close_loop.py --session deploy/runs/<ts>/attempt_01 --r2s real2sim/out/live/<run> [--prev loop/rounds/<r-1>/sim_config.json] [--round r1] [--n 200] [--workers 4] [--no-gen] [--policy d_ens3_search] What it decides (close_report.json says why for each item): - reality: did the cube end in the bowl? syn |
| `loop/eval_focus.py` | 60 | Closed-loop eval of a policy export in the MuJoCo twin on the loop's FOCUS scenes (scenes drawn around the real attempts' twins, loop/gen_traces.focus_scene), with held-out seeds: the check before a retrained policy is redeployed. Same episode code as policy/eval.py (only the scene source differs). policy/.venv/Scripts/python loop/eval_focus.py --config loop/rounds/<r>/sim_config.json --policy policy/export/<name> [--n 40] [--noise real] [--workers 4] |
| `loop/gen_traces.py` | 132 | Bot traces in the MuJoCo twin (policy/sim.py + the scripted player of policy/scripted.py), written in the Roblox trace format (spec/policy_io.md), so policy/train.py reads them next to roblox/out/traces. policy/.venv/Scripts/python loop/gen_traces.py --config loop/rounds/<r>/sim_config.json --n 200 [--workers 4] [--out loop/rounds/<r>/traces] [--tag <r>] Scenes come from the loop's sim config (loop/close_loop.py): with probability focus_frac a scene is drawn around one of the twins real2sim built from real attempts (its cube size / mass / friction, its cube and bowl positions, a few cm of spre |
| `policy/data.py` | 284 | Roblox game traces (spec/policy_io.md jsonl) -> (observation, 15-step action chunk) training samples. Per episode: 1. resample to a uniform 30 Hz grid (Studio frame jitter makes recorded ticks 0.4-2.5 ticks apart); 2. collapse dwells: ticks where the operator is idle (end-effector command and gripper label unchanged, gripper settled) and the arm is either at its label or not moving (DART-displaced, moving arms keep their recovery label) beyond the first KEEP_STATIC of each run are dropped. The scripted players sit still for up to 4 s (lift/retreat timeouts when the wrist hits its limit, close/ |
| `policy/ensemble.py` | 49 | Combine exports trained with the same recipe (different seeds) into one export whose act() averages their chunks, optionally with the runtime re-grasp search (runtime.GraspSearch). .venv/Scripts/python ensemble.py --name final --members s0 s1 s2 [--grasp-search] The first member's weights/norm sit at the top of export/<name>/ (so the export loads like any single policy), the others under members/<k>/; config.json lists them under "members". |
| `policy/eval.py` | 159 | Closed-loop eval in the MuJoCo twin (the cross-simulator test): N random scenes from the trace generator's distribution (sim.sample_scene, seeds 10000+i), policy at 30 Hz. .venv/Scripts/python eval.py --policy export/<name> [--n 100] [--noise clean/real] [--exec ensemble/k5/k15] .venv/Scripts/python eval.py --scripted [--n 100] [--noise clean/real] # baseline: the trace player itself noise "real" (what the real arm will add; components pos, offset, latency, stale can be given alone, e.g. pos+stale): per-episode cube xy bias N(0, 1 cm) + z N(0, 3 mm) + 2 mm per-tick jitter, bowl xy bias N(0, 1  |
| `policy/expert.py` | 528 | Generic bot expert for any argon_task/1 task in the MuJoCo task twin (sim.TaskTwin): it runs the task's skeleton (pick, place_in, place_on, place_at, push_to, release, home, wait) with the scripted player's machinery (scripted.Station: end-effector target integrated from key axes, 3 IK iterations per 60 Hz frame, DART perturbation on the executed command only, so the label q_target never carries it) and Operator's phase logic (approach, descend, close, lift, ... re-grasp on a miss). No per-task code: every number comes from the task's objects and the scene. Generalisations over scripted.Operat |
| `policy/gpu.py` | 113 | Torch (GPU) versions of data.Batcher.sample, runtime.fk_site/features and the ee label, so training is not bound by numpy sampling. Same semantics as the numpy reference (checked by `python gpu.py`). |
| `policy/legacy_to_task.py` | 55 | Legacy cube_in_bowl traces (spec/policy_io.md: cube_pose 7, bowl_xy 2 per tick) -> argon_task/1 task traces for tasks/cube_in_bowl (task_data.py format: "objects" {cube: 7, bowl: [x, y, h/2, 1, 0, 0, 0]}, "grip_open", "skill"). .venv/Scripts/python legacy_to_task.py --src ../roblox/out/traces ../loop/rounds/r1/traces ../loop/rounds/r2/traces [--out out/task_traces/cube_in_bowl_legacy] Episode meta keeps the legacy fields and adds task, scene objects' role (cube manipulated, bowl container). |
| `policy/roblox/PolicyEval.luau` | 142 | Closed-loop policy eval in the Roblox twin (Run mode). The policy runs in policy/serve.py; every 30 Hz tick each station POSTs its observation (spec/policy_io.md) and sets the reply as its JointSim targets. Stations are the trace-collection Twin.Station (unchanged): same arm, JointSim, table, cube and bowl as the traces; no operator, no teleop path, no trace upload. cd roblox; node tools/runmode.mjs ../policy/roblox/PolicyEval.luau @../policy/out/roblox_eval/args.json 1800 args: {server, n (stations), horizon_s, scenes: [{i, objects}] (from policy/roblox_eval.py, same scenes as eval.py)} Succe |
| `policy/rollout.py` | 71 | Render policy rollouts in the MuJoCo twin: mp4 + a frame strip per episode, and a per-tick text log (gripper site vs cube, gripper command). .venv/Scripts/python rollout.py --policy export/<name> --eps 0 3 7 [--noise real] [--tag v1] |
| `policy/runtime.py` | 264 | Standalone runtime for an exported cube-in-bowl policy (numpy only, no torch). from runtime import Policy pol = Policy.load("policy/export/<name>") chunk = pol.act(obs) # (15, 6) absolute joint targets, MJCF radians, for the next 15 ticks at 30 Hz target = pol.step(obs) # (6,) temporally ensembled target for the next tick (call once per tick) pol.reset() # new episode (clears the ensembling buffer) obs (spec/policy_io.md): {"q": 6 measured joints (MJCF rad; order shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper), "cube_xyz": 3 (m, robot base frame), "cube_size": float ( |
| `policy/scaling.py` | 69 | Data-scaling curve: MuJoCo success vs number of training episodes for one recipe ("more play -> better robot"). .venv/Scripts/python scaling.py --prefix sc_ --sizes 50 100 200 400 all --noise clean realx0.5 [--n 200] Reads out/evals/export_<prefix><size>_<noise>_ensemble_<n>.json (written by eval.py for exports trained with `train.py --name <prefix><size> --max-episodes <size>`), writes out/scaling.json and out/scaling.png. |
| `policy/scripted.py` | 276 | Python port of the Roblox trace player, for the MuJoCo baseline: Twin.Station's teleop path (end-effector target integrated from key axes, ArmIK 3 damped Gauss-Newton iterations per frame, DART perturbation on the executed command) and Twin.ScriptedOperator (approach, descend, close, lift, transport, lower, release, retreat, settle; re-grasp on a miss). Frame rate 60 Hz like Studio; the servo targets go to the MuJoCo twin's position actuators. This is a baseline only. It never produces training data. |
| `policy/serve.py` | 70 | HTTP policy server for closed-loop runs in the Roblox twin (policy/roblox/PolicyEval.luau) or anything else that can POST JSON. One temporally-ensembled policy instance per station. .venv/Scripts/python serve.py --policy export/<name> [--port 8770] POST /reset {"station": "S1"} -> {"ok": true} POST /act {"station": "S1", "obs": {q, cube_xyz, cube_size, bowl_xy, bowl_d}} -> {"q": [6 joint targets for the next tick], "ms": float} POST /log {...} -> appended to out/roblox_eval/<policy>.jsonl (episode results) |
| `policy/sim.py` | 283 | MuJoCo twin of one Roblox trace-collection station (roblox/src/ReplicatedStorage/Twin/Station.lua): the official SO-101 with the Roblox voxel collision boxes (roblox/ref/twin_mjcf.py), the table, a cube and a bowl (disc + 16 wall staves, as SceneLoader.makeBowl builds it). This is the cross-simulator eval: policies trained on Roblox traces run here. Frames: spec frame (robot base frame, origin on the table surface) everywhere in this module's API; the MJCF frame is the spec frame shifted down by BASE_Z. |
| `policy/task_data.py` | 294 | Task traces (argon_task/1, policy/task_traces.py or legacy_to_task.py) -> (K-slot observation, 15-step action chunk) training samples. The per-episode cleaning is data.py's (uniform 30 Hz grid, dwell collapse, IK chatter cut/smooth); only the observation changes: q (6) + K slots x [x, y, z, sin yaw, cos yaw, sx, sy, sz, present] in task["objects"] order (argon_tasks.encode_obs layout), absent slots all zero. Augmentation (Batcher.sample, as data.Batcher but per object): present objects' xy bias N(0, pos_sigma) (z 0.3x), a lifted object's pose is live (0.4), stale = its last pose on the table ( |
| `policy/task_eval.py` | 166 | Closed-loop eval of an argon_task/1 export in the MuJoCo task twin (sim.TaskTwin): N scenes from the task's own distribution (argon_tasks.sample_scene, seeds 10000+i), policy at 30 Hz with temporal ensembling. .venv/Scripts/python task_eval.py --policy export/<t>_v1 [--task ../tasks/<t>] [--n 100] [--noise clean/real] [--noise-scale 0.5] [--workers 4] noise "real" (eval.py's model, per object; components pos, offset, latency, stale can be '+'-joined): per-episode object xy bias N(0, 1 cm) + z N(0, 3 mm) and 2 mm per-tick xy jitter, per-joint zero offset N(0, 2 deg) (the arm reports q + offset  |
| `policy/task_runtime.py` | 173 | Standalone runtime for an exported argon_task/1 policy (numpy only, no torch, no repo imports). from runtime import Policy pol = Policy.load("policy/export/<task>_v1") chunk = pol.act(obs) # (15, 6) absolute joint targets, MJCF radians, for the next 15 ticks at 30 Hz target = pol.step(obs) # (6,) temporally ensembled target for the next tick (call once per tick) pol.reset() # new episode obs: {"q": 6 measured joints (MJCF rad), "slots": (K, 9)} or the flat (6 + 9K,) vector of argon_tasks.encode_obs. A slot is [x, y, z, sin yaw, cos yaw, sx, sy, sz, present] in config.json "task_spec" objects o |
| `policy/task_traces.py` | 85 | Bot traces for any argon_task/1 task: the generic expert (policy/expert.py) in the MuJoCo task twin, one jsonl per episode in the task trace format (spec/policy_io.md extended by spec/task_spec.md): {"meta": {episode_id, source "scripted", player "mujoco_task_expert", task, seed, scene, success, check, attempts, fail_reason, ticks, duration_s, dr}} then per 30 Hz tick {t, q, q_target (the label), q_exec (DART-perturbed command, never a label), ee_cmd, objects: {id: [x, y, z, qw, qx, qy, qz]}, grip_open, phase, skill} policy/.venv/Scripts/python policy/task_traces.py --task tasks/<t> --n 400 [- |
| `policy/tests/test_task_exports.py` | 72 | Task exports reproduce their training-time eval, and never-varied inputs cannot push them out of distribution. policy/.venv/Scripts/python policy/tests/test_task_exports.py (~1 min; --all ~3 min) 1. export -> task_eval on fixed seeds reproduces the eval recorded in its config.json ("eval", written by task_eval.py --record): first 20 seeds within 3 sigma of the recorded rate and episode-for-episode identical to the recorded file (MuJoCo is deterministic per seed). Default: the screwdriver demo exports; --all: every task export. 2. obs_const: a target zone at 20 deg yaw (constant 0 in training)  |
| `policy/train.py` | 229 | Train the state-based cube-in-bowl policy on Roblox game traces and export it (policy/export/<name>/). .venv/Scripts/python train.py --name mlp_v1 [--traces ../roblox/out/traces] [--tags bot1024 ...] [--max-episodes 100] [--steps 20000] [--device cpu/cuda] Model: residual MLP (observation -> 15x6 chunk of joint-target deltas from the observed q), L1 loss on normalised deltas. |
| `policy/train_task.py` | 181 | Train a state-based policy for one argon_task/1 task on its traces (policy/task_traces.py) and export it. .venv/Scripts/python train_task.py --task ../tasks/<t> --traces out/task_traces/<t> --name <t>_v1 [--steps 20000] [--batch 512] [--chatter smooth] [--miss-weight 4] [--pos-sigma 0.01] [--device cuda] Same model and loss as train.py (residual MLP -> 15 x 6 joint-target deltas, L1 on normalised deltas); observation = argon_tasks K-slot vector (task_data.py). The export (policy/export/<name>/) holds weights.npz, norm.json, runtime.py (= task_runtime.py) and config.json with "task", "task_spec |
| `real2sim/r2s/__init__.py` | 0 |  |
| `real2sim/r2s/anchors.py` | 115 | Grasp / release anchors from the joint log alone (no camera, no Gemini). A HOLD is a stretch where the gripper stopped part-way closed after a closing motion: the jaws are blocked by an object. (grasp_xyz / release_xyz below are the jaw-TIP midpoint; the object sits at the narrowest point of the jaw gap, gap_*_xyz, which can be ~1 cm away from it; object_point() turns that into the object centre.) The plateau must sit clearly above the empty-closed reading and below fully open, stay flat for >= 0.4 s, and (when commanded targets exist) the command must be more closed than the measurement. The  |
| `real2sim/r2s/calib.py` | 127 | Camera calibration from the robot itself: Gemini 2D keypoints vs MuJoCo FK of the logged joint states. Unknowns: camera extrinsics (rvec, tvec) + focal f (square pixels, principal point at the image centre, no distortion), optionally per-joint offsets (corpus deg -> model deg) and the shoulder_pan sign. Robust least squares (soft_l1, 15 px scale). The two jaw tips are unlabelled (Gemini is not reliable about which jaw is fixed), so each frame takes the cheaper of the two assignments. |
| `real2sim/r2s/data.py` | 82 | One public LeRobot SO-101 episode: joint states (corpus degrees -> MJCF radians) and decoded front-camera frames. Dataset: BrutalCaesar/phi_so101_cubes_cylinder_recovery_v1 @ cf44c09 (LeRobot v3.0, so_follower, 30 fps). Corpus audit (reachy-duplex docs/wg/so101_corpus_audit.md): mid-range degrees, lift/elbow offsets 0, wrist_roll level reads -67.7 (to_corpus offset +67.7). The 'front' key was verified from pixels: a fixed camera facing the arm. |
| `real2sim/r2s/detect.py` | 121 | Object detection (one request, first + last frame) -> scene_spec iteration 0, mask matching, anchor assignment. |
| `real2sim/r2s/evaluate.py` | 161 | Independent per-iteration metrics. Nothing here comes from the critic; the critic never sees the eval frames. - obj IoU / centroid px: rendered silhouette vs the colour mask of the real (rectified) frame, eval frames only. - ref px: rendered bbox centre vs a separate Gemini detection (another model) on the eval frames (optional). - kp held-out px: held-out Gemini keypoints vs FK projected through the scene camera. - anchors (3D, camera-free): carried object xy vs the grasp anchor; release point inside the container footprint. - truth (synthetic sessions only): object xyz and camera position er |
| `real2sim/r2s/gemini.py` | 221 | Thin Gemini client: one call = one jsonl log line (model, role, latency, tokens, prompt tag). Backs off on 429/5xx. Models are ROLES, every call site picks one: main = R2S_MODEL_MAIN (default gemini-4-argon; R2S_MODEL is the legacy override): the reasoning calls (task from the whole video + joint log, the twin critique / fix loop, scene detection). small = R2S_MODEL_SMALL (default gemini-3.8-flash): per-frame pointing / keypoints, boxes on a frame, JSON repair. A daily-quota / zero-quota 429 on the MAIN role fails loudly (QuotaError) so a demo never runs on the wrong model; only R2S_ALLOW_FALL |
| `real2sim/r2s/keep.py` | 65 | Keep rule for the critique loop: a candidate scene replaces the best one only if 1. the combined score improves by more than EPS, AND 2. no object's held-out IoU falls more than IOU_DROP below the best IoU it has had in any kept scene, AND 3. when the anchors are NOT enforced (the joint log and the camera disagree), no container's IoU falls more than CONTAINER_DROP below the current best scene's: physics alone must not buy a worse-looking container, AND 4. the camera / anchor terms do not get worse: held-out keypoint px (+1 px slack) and, when the anchors are enforced (joint log and camera agr |
| `real2sim/r2s/kpcal.py` | 165 | Camera from the robot itself, two ways, both reported: keypoints Gemini points at jaw tips / wrist / elbow / shoulder / base on ~14 frames, all in ONE request; FK of the logged joints gives the 3D points; PnP + focal fit (r2s.calib.Problem). keypoints (known f) jaw tips + base only, focal fixed from the camera's intrinsics (ArUco K or --focal-px). With the focal free, keypoints trade focal for distance (synthetic: 25-89 cm off); with it fixed, 3-5 cm. keypoints+anchors the same, plus table-level correspondences from the grasp / release anchors: the object's pixel at rest (Gemini detection) mus |
| `real2sim/r2s/live.py` | 76 | Live run output for the dashboard: out/live/<run>/{run.json, iter_N.json, images}. Every JSON file is written to a temp file and renamed (os.replace), so a reader never sees a half-written file. Schema: README.md "Live output". |
| `real2sim/r2s/measure.py` | 71 | Measurements that do NOT come from Gemini: colour masks of the red cube and the cardboard box, a background- subtraction mask of the moving arm, and IoU helpers. Thresholds were set once on frame 100 (HSV samples) and are fixed for every iteration. |
| `real2sim/r2s/metrics.py` | 101 | Per-iteration metrics that do not come from the critique loop's own judgement. - obj IoU / centroid error: rendered object silhouette vs a colour-threshold mask of the real frame (measure.py), on frames the critic never sees. - ref centre error: rendered object bbox centre vs an independent Gemini detection (another model, other frames). - keypoint error: held-out Gemini keypoints (from step 1) vs FK projected through the spec camera. - arm IoU: rendered moving-arm silhouette vs background subtraction, on held-out frames. - grasp / release (3D, camera-free): the cube must sit where the jaws cl |
| `real2sim/r2s/patch.py` | 209 | Critic patches: schema, validation, application, and the hard constraints from the anchors. A patch is JSON: {"discrepancies": [{"target": str, "issue": str}], "object_patches": [{"id": str, "dxyz_m": [dx, dy, dz] / null, "size_m": [..] / null, "yaw_deg": num / null, "color": "#rrggbb" / null, "friction": num / null, "mass_kg": num / null}], "done": bool} Every field is checked against PATCH_RULES; a bad field is dropped (and reported), not the whole patch. After the patch the scene is re-constrained (objects on the table, anchors) and checked against the scene spec (validate_scene); a scene t |
| `real2sim/r2s/physics.py` | 209 | Physical match: replay the recorded joint trajectory in MuJoCo with the scene's objects as free bodies and compare the simulated object motion with the object's track in the video (colour mask centroid). Robot model: the Roblox twin (roblox/ref/twin_mjcf.py): official SO-101 kinematics and dynamics, collision meshes replaced by the same voxel boxes the Roblox builder uses, robot-robot contacts off, an object's friction governs its contacts. The official mesh model never held the cube in the spike (the jaws pushed it away); the twin's boxes do (roblox/results/grasp_mu*.json). Frames: twin_mjcf  |
| `real2sim/r2s/robot.py` | 209 | SO-101 in MuJoCo (official so101_new_calib.xml): FK keypoints, scene building from a scene spec, rendering. Frames. The scene spec frame is the MJCF base frame shifted so that the origin is the centre of the base footprint on the table: spec = mjcf - BASE_ORIGIN_MJCF. x forward (the arm reaches +x at q=0), y left, z up. Joints. The dataset is in LeRobot mid-range degrees ("corpus" frame after the audit's roll offset). The official model's zero is also mid-range, so model_rad = deg2rad(corpus_deg + JOINT_OFFSET_DEG); JOINT_OFFSET_DEG is a calibration (fitted / checked in calib.py), gripper 0-10 |
| `real2sim/r2s/scene.py` | 86 | Scene-spec construction (iteration 0 from one Gemini detection pass), rendering and comparison images. |
| `real2sim/r2s/session.py` | 193 | One recorded episode, whatever its source: frames, joint trajectory in MJCF radians, task text, an optional fixed camera calibration (ArUco), and colour masks for object tracking. Sources - public LeRobot episode (r2s.data): `Session.public(0)`; joints = dataset degrees through the spike's calibration. - bench session from deploy/scripts/record_session.py (or deploy/scripts/synth_session.py, scripts/synth_bench.py): `<dir>/{frames/0000.jpg..., joints.jsonl (t, frame, owner_deg[, leader_deg]), meta.json}`. Owner degrees go through deploy/config/joint_map.json (the candidate named in meta.json[" |
| `real2sim/r2s/taskscene.py` | 138 | Task-guided scene building for run_bench (--task-json): the task's objects (ids, shapes, sizes, roles from task.json) replace the free-form detection, the boxes Gemini drew in the first / last frame while writing the task replace the detection request, and every object's real mask comes from an appearance model fitted on the first frame (deploy/so101d/appearance: GrabCut in the box -> HSV histogram vs the rest of the frame), not from bench.json colour ranges. Iteration-0 poses come from the same silhouette fitters the real arm's perception uses (so101d.taskperception: hull over yaw for boxes / |
| `real2sim/scripts/__init__.py` | 0 |  |
| `real2sim/scripts/probe_point.py` | 22 | Probe: ask Gemini to point at arm keypoints on a few frames, draw them. |
| `real2sim/scripts/rescore_keep.py` | 63 | Re-judge recorded critique-loop runs under the keep rule in r2s/keep.py (no Gemini, no rendering: the iteration metrics in out/live/<run>/iter_N.json are replayed in order). python -m scripts.rescore_keep [run ...] (default: every out/live/* run with critique iterations) Prints one row per critique iteration: recorded verdict -> new verdict (+ reasons). Caveat: after the first changed verdict the later candidates were patches of the scene the old rule kept, so their rows are counterfactual. |
| `real2sim/scripts/run_bench.py` | 498 | Real2sim v2, one command: recorded episode -> calibrated camera -> scene_spec -> render / critique / patch loop with keep-best, a stop rule and a physical-match check, streamed to out/live/<run>/ for the dashboard. python -m scripts.run_bench <session_dir / public:0> [--model gemini-4-argon] [--run NAME] [--iters 6] [--patience 2] [--max-calls 12] [--budget-s 1800] [--no-physics] [--ref] [--camera camera.json] [--task "..."] [--anchors auto/hard/soft] Gemini requests per run: 1 keypoint batch (0 for public:0, cached spike points) + 1 detection (2 frames) + 1 per critique iteration (+1 with --r |
| `real2sim/scripts/s1_calib.py` | 77 | Step 1b: fit the front camera from Gemini keypoints + FK; report held-out reprojection error; save overlays. |
| `real2sim/scripts/s1_point.py` | 44 | Step 1a: Gemini points at arm keypoints on calibration + held-out frames. Cached in out/points/<model>/<frame>.json. |
| `real2sim/scripts/s2_objects.py` | 93 | Step 2+3: Gemini detects the task objects on one frame -> deproject onto the table through the fitted camera -> scene_spec iteration 0 -> render (side-by-side + outline overlay). |
| `real2sim/scripts/s4_loop.py` | 162 | Step 4: render -> Gemini critique -> patch -> re-render, N iterations, with independent metrics per iteration. Usage: R2S_MODEL=<critic model> python -m scripts.s4_loop [n_iter] [run_name] The reference detection (independent check) uses R2S_REF_MODEL (default gemini-3-flash-preview) on frames the critic never sees; it is made once and cached. |
| `real2sim/scripts/s4b_eval.py` | 46 | Re-score every saved iteration of a loop run with the current metric set; write metrics_full.json + a table. |
| `real2sim/scripts/s5_latency.py` | 51 | Step 5: latency + accuracy of a single pointing call (jaw tips) per model and thinking budget. Accuracy reference: the calibrated camera + FK projection of the two jaw tips (step 1), not Gemini. Usage: python -m scripts.s5_latency <model> [budget ...] (budget 'none' = model default) |
| `real2sim/scripts/s6_physics.py` | 105 | Step 6 (optional): physical match. Replay the episode's commanded joint trajectory (action) through the MJCF position servos with the spec objects as free bodies; compare the simulated cube path, projected through the spec camera, against the real cube's colour-mask centroid in every frame. Usage: python -m scripts.s6_physics <scene.json> [friction] [mass_kg] |
| `real2sim/scripts/synth_bench.py` | 129 | Synthetic BENCH session in deploy/scripts/record_session.py format, with a pick-and-place in it (deploy's synth_session.py has the arm only): the official SO-101 picks a red cube and drops it in a blue bowl, rendered by deploy/so101d/synth.World from a high camera (kinematic cube: it follows the jaws while they hold it). python -m scripts.synth_bench [--out data/bench/synth_pick_place] [--cube 0.23 -0.07] [--bowl 0.21 0.12] Writes frames/, joints.jsonl (t, frame, owner_deg, leader_deg: the commanded gripper closes further than the measured one during the hold, as on the real arm), meta.json, c |
| `real2sim/scripts/synth_task_bench.py` | 139 | Synthetic BENCH session for any argon_task/1 task (record_session layout + ground truth keyed by task object id): the official SO-101 picks the task's first manipulated object and puts it in / on its goal target (or moves it to an `at` goal), rendered by deploy/so101d/synth_task.TaskWorld (white arm, printed ArUco board, every task object). python -m scripts.synth_task_bench [--task path/task.json / --bench tonight] [--out data/bench/synth_<name>] [--place obj=x,y ...] [--seed 0] Writes frames/, joints.jsonl (owner_deg + leader_deg; the commanded gripper closes further than the measured one wh |
| `real2sim/scripts/task_from_video.py` | 485 | One real video (or a first frame + a sentence) -> tasks/<name>/task.json (argon_task/1) + tasks/<name>/scene.json. python -m scripts.task_from_video <session_dir / public:N / video.mp4 / frame.jpg> [--instruction "..."] [--name NAME] [--model M] [--no-loop] [--iters 4] [--max-calls 8] [--run RUN] [--retries 2] [--frames] [--allow-fallback] [--answer answer.json] 1. ONE long-context request on the MAIN model role (r2s.gemini: R2S_MODEL_MAIN, default gemini-4-argon): the WHOLE episode as a video (a session's frames are encoded to mp4; --frames sends 2 fps frames instead), its first and last fram |
| `roblox/ref/dyn_check.py` | 45 | Dynamic check: replay smooth multi-joint sinusoids + a step in Roblox Run mode and in MuJoCo (same targets, same frame times, MJCF actuator gains kp 998.22 / kv 2.731 / forcerange 3.35, damping 0.6, frictionloss 0.052, armature 0.028, gravity 9.81). .venv/Scripts/python ref/dyn_check.py [jointsim/spring] |
| `roblox/ref/episode_replay.py` | 51 | Replay a logged Roblox teleop episode (out/episodes/<name>.jsonl, 30 Hz actions) in the MuJoCo twin and compare object trajectories. Usage: .venv/Scripts/python ref/episode_replay.py [episode.jsonl] [--scene scenes/default.json] |
| `roblox/ref/fk_check.py` | 88 | Kinematic check: Roblox SO-101 vs MuJoCo FK for random joint configs. .venv/Scripts/python ref/fk_check.py kinematic # edit mode, composed joint frames .venv/Scripts/python ref/fk_check.py physics [gravity] # Run mode, hinge + PD springs, settle 1 s Compares the gripperframe site, both jaw-tip points and every body origin. In physics mode it also compares at the joint angles Roblox actually reached (separates geometry error from PD tracking error). |
| `roblox/ref/grasp_check.py` | 76 | Grasp: can the SO-101 twin pick up a 3 cm cube with friction only? Same joint trajectory in Roblox Run mode and in the MuJoCo twin: open, descend around the cube, close (servo saturates at 3.35 N m), lift 7 cm, hold 2 s. .venv/Scripts/python ref/grasp_check.py [--friction=0.6] [--mass=0.02] [--repeats=3] |
| `roblox/ref/ik.py` | 65 | Small IK helpers for the SO-101 (MuJoCo FK): place the gripper with its fingers pointing straight down. |
| `roblox/ref/live_check.py` | 64 | Live-drive check: run spikes/live_drive.luau (twin follows tools/joint_server.py over SSE or polling) and measure stream rate, sample age and the lag from the streamed target to the twin's measured joints. .venv/Scripts/python ref/live_check.py [sse/poll] [seconds] Needs tools/joint_server.py running on :8765 and HttpService enabled in the place. |
| `roblox/ref/push_check.py` | 114 | Push physics: the arm flicks a block across the table along a fixed joint trajectory, in Roblox Run mode and in the MuJoCo twin (identical collision boxes, same targets at the same frame times). Compares the block's final pose and path for several friction / mass settings. .venv/Scripts/python ref/push_check.py |
| `roblox/ref/render_compare.py` | 114 | Check the Roblox render path against MuJoCo: same scene_spec camera, same joint angles, robot silhouette overlap. .venv/Scripts/python ref/render_compare.py scenes/default.json --cam side --q 0.2,-0.4,0.6,1.0,0,0.4 Renders the MuJoCo twin (official meshes) with the spec camera, renders the Roblox twin with tools/render.py (needs tools/load_meshes.mjs first for real meshes), segments the yellow/black robot in both and reports IoU and the silhouette centroid offset in pixels. Writes shots/compare_<cam>.png (MuJoCo / Roblox / overlay). |
| `roblox/ref/replay.py` | 82 | Run a joint trajectory in Roblox Run mode (spikes/traj_replay.luau) and replay the exact same targets in MuJoCo. |
| `roblox/ref/so101.py` | 28 | Shared MuJoCo helpers for the SO-101 twin checks (official TheRobotStudio so101_new_calib model). |
| `roblox/ref/twin_mjcf.py` | 104 | MuJoCo twin of the Roblox scene: the official SO-101 with its collision meshes replaced by the exact voxel boxes the Roblox builder uses (model/so101_boxes.json), plus table and objects from a scene_spec. Contact conventions match roblox/src/ReplicatedStorage/Twin/SceneLoader.lua: - robot-robot contacts off (Roblox collision group "Robot" does not self-collide); - an object's friction governs all of its contacts (geom priority 1 here, FrictionWeight 100 in Roblox); - spec frame = MJCF frame shifted up by BASE_Z (base mesh bottom at z = -0.0024). |
| `roblox/spikes/collect.luau` | 9 | Trace collection run (Run mode): node tools/runmode.mjs spikes/collect.luau '{"n":4,"minutes":10,"seed":1,"tag":"bot"}' Every game level (task_spec tasks, Twin.SkillOperator): add "levels":"all" (or a list of level ids). Needs tools/joint_server.py (receives /trace). Returns Twin.Collector stats. |
| `roblox/spikes/contact_probe.luau` | 41 | Which robot parts touch at a pose? (Run mode) Builds the arm, drives it with JointSim to args.q, then lists collision-part pairs from different bodies whose geometry overlaps (GetPartsInPart) and the collision group state. |
| `roblox/spikes/fk_probe.luau` | 61 | FK probe. Builds the SO-101 and reports site / body-origin positions (MJCF frame, m) for each joint vector in _G.SPIKE_ARGS.qs. mode = "kinematic" (edit mode, tools/run.mjs): poses bodies directly by composing joint frames. mode = "physics" (Run mode, tools/runmode.mjs): RobotKit.JointSim servo drive at the model gains, gravity args.gravity (studs/s^2, default 0), settle args.settle seconds, then read; also returns the measured joint angles so geometry and tracking errors can be separated. |
| `roblox/spikes/hitch_probe.luau` | 12 | Frame-hitch probe (Run mode): timestamps of frames longer than 50 ms over args.s seconds. |
| `roblox/spikes/human_test.luau` | 32 | Human round test (Play mode, TwinMain active: KEEP_TWIN=1 PLAY=1 node tools/runmode.mjs spikes/human_test.luau). Needs tools/joint_server.py on 127.0.0.1:8765 (Studio default) to receive the trace. Starts a round for the real Player exactly as the PLAY button does (TwinMain's startRound), then feeds that player's station the same teleop messages a client sends (via Twin.ScriptedOperator, phases stripped) and waits for TwinMain's own success detection to end the round and for the upload callback. Returns the player's state. |
| `roblox/spikes/keyboard_watch.luau` | 25 | Real-keyboard teleop check (Play mode, TwinMain active): records what the client's key handler sends and how the end-effector command and gripper respond, for args.s seconds after a Player has joined. |
| `roblox/spikes/levels_bot.luau` | 101 | Run mode: Twin.SkillOperator plays each level (Twin.Round judges with Twin.TaskCheck). No uploads. node tools/runmode.mjs spikes/levels_bot.luau '{"levels":["cube_in_bowl"],"n":2,"seed":1}' 400 Returns per level: successes / n, outcomes, bot reasons, sim seconds per round. |
| `roblox/spikes/live_drive.luau` | 46 | Live drive test (Run mode): the twin follows joint targets streamed from tools/joint_server.py over SSE (Twin.LiveLink) for args.s seconds. Logs per physics frame: latest streamed q, its age, measured q. Returns message stats and the raw series (Python side estimates the stream -> twin lag). |
| `roblox/spikes/play_smoke.luau` | 75 | Play-mode smoke test of the game loop (TwinMain active, a real Player joins): KEEP_TWIN=1 PLAY=1 node tools/runmode.mjs spikes/play_smoke.luau '{"levels":["tutorial_lift","cube_in_bowl"],"pause":3}' 400 For each level: starts a round for the player exactly as the client's PLAY / level card does (TwinMain.startRound), drives the player's station with Twin.SkillOperator through Station.handleMessage (the client's message path, phase labels stripped), waits for TwinMain's server-side judge (Twin.Round + TaskCheck) to end the round, and records the result card state (stars, points, combo, unlocks) |
| `roblox/spikes/render_scene.luau` | 78 | Edit-mode render setup (tools/run.mjs, plugin context). args: spec : scene_spec table (default: ReplicatedStorage.Twin.DefaultScene) q : 6 joint angles (MJCF rad) for a kinematic pose of the arm (default zeros) cam : camera id in spec.cameras markers = true : overlay magenta corner markers on the viewport (to locate it in a window screenshot) clear = true : remove the preview twin and markers, restore the camera mask = true : robot drawn flat magenta (Neon), table/objects hidden: a silhouette for render-compare checks |
| `roblox/spikes/slide_probe.luau` | 32 | Free-slide probe (Run mode): blocks on the scene table, each launched at args.v m/s along +x, report how far they slide. Coulomb ideal: v^2 / (2 mu g). args.cases = {{friction, mass}...} |
| `roblox/spikes/stream_probe.luau` | 107 | Live joint-streaming probe (Run/Play mode server script; needs HttpService.HttpEnabled and tools/joint_server.py). Measures, against http://127.0.0.1:<port>: poll : back-to-back HttpService:GetAsync("/q") for args.poll_s seconds: request rate, RTT, sample age, errors sse : HttpService:CreateWebStreamClient(SSE, "/sse") for args.stream_s seconds: message rate, sample age ws : CreateWebStreamClient(WebSocket, "/ws"): message rate, sample age, ping->pong RTT Sample age = receive time - the server's t_ms stamp (same machine clock), i.e. end-to-end latency. |
| `roblox/spikes/teleop_test.luau` | 73 | Teleop test (Play mode with TwinMain active: KEEP_TWIN=1 PLAY=1 node tools/runmode.mjs spikes/teleop_test.luau). Plays the operator: drives _G.TWIN.input (the same fields the client's key handler fills through the RemoteEvent) with a bang-bang "key" controller to pick the red cube: above it, down, close, lift, carry, place. Reports end-effector tracking (IK target vs reached site), whether the cube came along, frame times and the episode logger's counters. |
| `roblox/spikes/torque_probe.luau` | 60 | Constant-torque probe (Run mode): gravity 0, one joint driven by a saturated spring (stiffness huge, damping 0, MaxTorque = args.torque N*m) toward a far target; all other joints held by their PD. Logs q(t) of that joint to estimate the effective inertia Roblox sees about the joint axis (MuJoCo: M diag). Also a reference: a lone 1 kg, 0.2 x 0.05 x 0.05 m rod on a hinge through its centre (I = m L^2 / 12). |
| `roblox/spikes/traj_replay.luau` | 127 | Joint-trajectory replay in Run mode (tools/runmode.mjs). Builds a scene (scene_spec, optional) + the SO-101, holds q[1] for args.settle seconds, then plays {dt, q} with targets updated every PreSimulation (60 Hz frames, linear interpolation at the current sim time) and logs at every PostSimulation: t (sim time since start), the target actually set, measured q, gripperframe position, object poses. args: q = {{6}...}, dt, gravity (studs/s^2, default 98.1), kp / kv (number or 6-list; default model gains), frictionloss (default true), scene (scene_spec table or nil), settle (s), tail (s after the  |
| `roblox/spikes/ttt_smoke.luau` | 58 | Play-mode smoke test of the tic-tac-toe level (TwinMain active, a real Player joins): KEEP_TWIN=1 PLAY=1 node tools/runmode.mjs spikes/ttt_smoke.luau '{"games":1,"pause":4}' 400 Starts the place_on_cell match for the player exactly as the PLAY THE ROBOT button does, then plays the player's moves with Twin.SkillOperator through Station.handleMessage: before each move the test picks a free cell (centre first, else random) and points the move's "cell" marker at it so the bot's place_at skill aims there. TwinMain's match logic (Twin.TicTacToeMatch) detects the placement, the robot answers, and the |
| `roblox/src/ReplicatedStorage/RobotKit/Builder.lua` | 211 | Builds a physical robot Model from a generated description (tools/mjcf_to_luau.py). Adapted from robot-academy RobotKit.Builder: fixed (anchored) bases, per-part colours, sites, joint friction loss, an origin to place the robot, and optional imported link meshes. Per body: an invisible, non-colliding inertia box (the assembly root, carrying the body's exact mass, CoM and principal inertia) plus massless collision parts welded to it. Per hinge: a HingeConstraint (axis, limits; its Motor with zero target speed and MotorMaxTorque = frictionloss is MuJoCo's dry joint friction) and a TorsionSpringC |
| `roblox/src/ReplicatedStorage/RobotKit/JointSim.lua` | 342 | Joint-space servo dynamics for armature-dominated arms (SO-101: armature 0.028 kg m^2 vs links ~1e-4). Why: RobotKit.Builder folds armature into each body's inertia box, but a rigid body cannot carry inertia about one axis only (triangle inequality): 0.028 about the joint axis forces >= 0.014 about the two perpendicular axes, so the arm swings ~4x too heavy about shoulder_pan (measured 0.18 vs MuJoCo 0.043 kg m^2) and the PD springs ring. What: the MJCF servo model is integrated in Luau at 480 Hz (semi-implicit Euler, like MuJoCo's default): M(q) qdd + c(q, qd) + g(q) = clip(kp (q* - q) - kv q |
| `roblox/src/ReplicatedStorage/RobotKit/Robot.lua` | 170 | Runtime handle for a robot built by Builder: PD targets, joint state and base state in MuJoCo conventions, so policies trained against MuJoCo/Isaac observation layouts can read it directly. (From robot-academy; adds the build origin, sites, the model's own actuator gains and kinematic posing.) Frames: Roblox (X, Y up, Z) <-> MuJoCo (x fwd, y left, z up) via v_mj = (v.X, -v.Z, v.Y). Positions in metres, relative to the model's WorldPivot (the Builder origin = the MJCF world origin for a fixed base). |
| `roblox/src/ReplicatedStorage/Robots/SO101.lua` | 36 | Generated by tools/mjcf_to_luau.py from model/so101_official/so101_new_calib.xml (key=None). Do not edit. Roblox frame, studs, MJCF world origin (fixed base) or base ground point; see tools/mjcf_to_luau.py. |
| `roblox/src/ReplicatedStorage/Robots/SO101_geoms1.lua` | 652 | Generated by tools/mjcf_to_luau.py: collision parts of SO101 (chunk 1). Do not edit. |
| `roblox/src/ReplicatedStorage/Robots/SO101_geoms2.lua` | 186 | Generated by tools/mjcf_to_luau.py: collision parts of SO101 (chunk 2). Do not edit. |
| `roblox/src/ReplicatedStorage/Twin/ArmIK.lua` | 108 | End-effector IK for the SO-101 twin (player teleop): the gripperframe site goes to a target point (MJCF frame, m) with the fingers pointing straight down, solved over shoulder_pan, shoulder_lift, elbow_flex, wrist_flex by damped Gauss-Newton on finite-difference Jacobians. wrist_roll and gripper pass through. FK composes the joint frames of RobotKit.Robot (the same transforms Robot:poseKinematic uses), so it is exact for the built model. |
| `roblox/src/ReplicatedStorage/Twin/Collector.lua` | 154 | Parallel trace collection: N Twin.Stations side by side in one place (80 studs apart, independent physics islands), each driven by a Twin.ScriptedOperator, episodes streamed to tools/joint_server.py /trace. Collector.run blocks until `minutes` of wall time (or maxEpisodes) and returns throughput/CPU stats. opts.levels (list of level ids from Twin.Levels, or "all" = every non-tutorial level): stations cycle through those levels instead, each round a Twin.Round (task scene from the task's randomize, judged by Twin.TaskCheck, trace with meta.task / object slots) driven by Twin.SkillOperator from  |
| `roblox/src/ReplicatedStorage/Twin/DefaultScene.lua` | 23 | Generated from scenes/default.json (copy of the scene_spec example). Returns the JSON string. |
| `roblox/src/ReplicatedStorage/Twin/EpisodeLogger.lua` | 58 | 30 Hz episode recorder for teleop data: joint state, action (the joint targets sent to the servos), the end-effector command and object poses (scene_spec frame). Batches are POSTed once a second to tools/joint_server.py /episode?name=<name> as JSON lines (one batch per line), well under HttpService's 500 requests/min. |
| `roblox/src/ReplicatedStorage/Twin/LevelGen.lua` | 163 | Scene sampling for a task (spec/task_spec.md, argon_task/1): task objects -> scene_spec objects with poses drawn from task.randomize. Pure Luau (no Roblox APIs beyond an rng with NextNumber(lo, hi)), so tests/levels_test.luau runs it under plain luau with a stand-in rng. randomize entries per object id: {r = {lo, hi}, az_deg = {lo, hi}, yaw_deg = {lo, hi}} (polar around the base), or {x = {lo, hi}, y = {lo, hi}}; an object with neither keeps its own `xyz` / `xy` (or rests where it is stacked via `on`). Physical randomisation in the same entry: size_scale {lo, hi} (all extents), size_m (per ext |
| `roblox/src/ReplicatedStorage/Twin/Levels.lua` | 150 | Level registry: the game's levels are the tasks in Twin.TaskData (generated from tasks/*/task.json by tools/gen_levels.py) plus a built-in 10-second tutorial. Pure Luau (tests/levels_test.luau). Level = { id, task (argon_task/1 table), title, blurb, icon, episode_s, par_s, tutorial?, order }. Round length = task.limits.episode_s (clamped to MIN_S..MAX_S for phone play), par time for the 2nd/3rd star = task.game.par_s or PAR_FRAC of the round. Daily challenge: one non-tutorial level and one seed per UTC day, the same for every server, so everyone plays the identical scene and shares a leaderboa |
| `roblox/src/ReplicatedStorage/Twin/LiveLink.lua` | 80 | Live joint targets from tools/joint_server.py (the real leader arm, or anything that POSTs /set). Server-side only (HttpService; the place needs Allow HTTP Requests / HttpService.HttpEnabled). Preferred transport is Server-Sent Events through HttpService:CreateWebStreamClient: every sample arrives (60 Hz measured) with ~0 ms median age on localhost and it costs one request against the HTTP budget. Fallback is polling GET /q: ~16 ms round trip in Studio (one frame), but HttpService allows 500 requests/min per server in a live game (~8 Hz), so it is only good for Studio demos. |
| `roblox/src/ReplicatedStorage/Twin/Round.lua` | 168 | One round of a level on a Twin.Station, shared by human players (ServerScriptService/TwinMain), the in-game bots and Twin.Collector, so every round is recorded and judged the same way. Round.start(station, level, opts) samples the scene (Twin.LevelGen from the task's randomize, seeded), starts the episode (meta: task, level, seed, daily, tutorial, novice, ...). round:update(dt) per PostSimulation frame: judges the goal with Twin.TaskCheck (all predicates, held for HOLD_S with objects at rest), tracks grasps / drops / container entries and returns juice events {kind = "grasp" / "drop" / "swish" |
| `roblox/src/ReplicatedStorage/Twin/SceneLoader.lua` | 398 | Builds a tabletop twin (table, objects, cameras, SO-101) from a scene_spec v0 JSON string or table (argon-hack/spec/scene_spec.md). Frames: the spec frame is the robot base footprint on the table surface, metres, x forward, y left, z up. In Roblox it sits at SceneLoader.ORIGIN (table top at Y = 7.5 studs), axes mapped (x, y, z) -> (X, -Z, Y), 1 m = 10 studs. The MJCF world origin of so101_new_calib is 2.4 mm above the table (the base mesh bottom is at z = -0.0024), so the robot is built at ORIGIN + 0.024 studs up. Contact friction convention: an object's `friction` is THE coefficient for every |
| `roblox/src/ReplicatedStorage/Twin/Score.lua` | 72 | Round scoring for the twin game. Pure Luau (tests/score_test.luau). points = (BASE + time bonus + clean bonus) x combo multiplier, 0 on a failed round stars = 1 success, 2 success under par time, 3 success under par with a clean grasp (one grasp per object, nothing dropped) combo = consecutive successful rounds (any level); a failure resets it. Multiplier 1, 1.5, 2, 2.5, 3 (cap). Cosmetics: gripper colours unlock at total-star thresholds (stars summed over levels, best per level). |
| `roblox/src/ReplicatedStorage/Twin/ScriptedOperator.lua` | 261 | Scripted, deliberately imperfect operator for a Twin.Station: picks the cube and drops it in the bowl by sending the same teleop messages a keyboard player sends (axes in {-1, 0, 1} per frame, Space-like grip toggle, Shift-like fine mode). Imperfections, all drawn from the episode seed: speed: per-segment duty cycle 0.35-1.0 on the held keys (like tapping), fine mode near the goal; aim: noisy cube estimate (sigma 4 mm) and, in ~15% of episodes, a deliberate 12-20 mm first-grasp offset; grasp height noise +-5 mm; misses are detected (cube did not rise) and re-grasped from a fresh estimate; DART |
| `roblox/src/ReplicatedStorage/Twin/SkillOperator.lua` | 457 | Scripted operator for ANY task (spec/task_spec.md): runs the task's `skeleton` (pick, place_in, place_on, place_at, push_to, release, home, wait) on a Twin.Station by sending the same teleop messages a keyboard player sends, with the imperfections of Twin.ScriptedOperator (per-segment duty cycle, aim / height noise, DART disturbances on the executed command with a clean label, hesitations) drawn from the episode rng. Twin.ScriptedOperator stays the byte-stable cube_in_bowl collector; this one plays every level (bots on idle stations, Collector `level`). Grasp geometry comes from the twin's own |
| `roblox/src/ReplicatedStorage/Twin/Station.lua` | 456 | One teleop station: a table + SO-101 twin at its own origin, the teleop command path, RobotKit.JointSim, and a 30 Hz episode recorder in the spec/policy_io.md trace format. Humans (TwinMain + the TwinTeleop client) and scripted operators (Twin.ScriptedOperator) drive a station only through Station.handleMessage, i.e. the exact message the client sends: {move = Vector3 axes in -1..1 (x fwd, y left, z up), roll = -1..1, grip = bool, fine = bool (slow mode)}. Scripted operators may add `perturb` (Vector3, m): a DART disturbance that shifts the executed end-effector command while the recorded labe |
| `roblox/src/ReplicatedStorage/Twin/TaskCheck.lua` | 280 | Luau port of argon_tasks.check(task, state) (argon_tasks/check.py, spec/task_spec.md "Success"): the Roblox game decides success with it on the server. Pure Luau, no Roblox APIs; tests/taskcheck_test.luau replays the shared vectors (argon_tasks/tests/vectors.json, generated into tests/taskcheck_vectors.luau) and must match every case. state = { objects = { [id] = { xyz = {x, y, z}, quat = {w, x, y, z} / yaw_deg = deg, v = speed m/s?, size_m = {...}? } }, gripper = { open = bool, site_xyz = {..}?, holding = {ids}? }?, -- absent: no gripper_open check initial = { [id] = { xyz = {...} } }?, -- no |
| `roblox/src/ReplicatedStorage/Twin/TaskData.lua` | 500 | GENERATED by roblox/tools/gen_levels.py from tasks/*/task.json. Do not edit; rerun the tool. |
| `roblox/src/ReplicatedStorage/Twin/TicTacToe.lua` | 132 | Tic-tac-toe rules and the robot opponent, pure Luau (tests/tictactoe_test.luau). Board: 9 cells, index 1..9 row-major (1 2 3 / 4 5 6 / 7 8 9); 0 = empty, 1 = player (X), 2 = robot (O). The robot plays minimax (perfect play) except that with probability `slip` it picks a random legal move instead of the best one, so players can win sometimes. Draws stay the norm for careful players. |
| `roblox/src/ReplicatedStorage/Twin/TicTacToeMatch.lua` | 314 | One tic-tac-toe match on a Twin.Station (level place_on_cell, tasks/place_on_cell/task.json): the player is X and moves by driving the arm: pick the brown supply cube and set it down on a free cell. The robot (O) answers at once with Twin.TicTacToe (minimax with slips) by dropping a blue disc on its cell. Every player move is one recorded episode in the place_on_cell format: objects cube (the supply cube), cell (the cell the player chose; its slot is a Station "late" slot patched when the move ends), p1/p2 = the two newest X cubes, p3/p4 = the two newest O discs (board pieces are anchored; ext |
| `roblox/src/ReplicatedStorage/Twin/TouchMap.lua` | 63 | Player input -> Twin.Station teleop axes, as pure functions (no Roblox APIs, so tests/touchmap_test.luau runs it under plain luau). All horizontal input is camera-relative: "up" on the stick / W moves the gripper away from the camera along the table, "right" moves it to the screen's right. The station's message is {move = (x fwd, y left, z up) axes in -1..1, roll = -1..1}, speed proportional to the axis value. |
| `roblox/src/ServerScriptService/TwinMain.server.lua` | 951 | Play-mode / live-server entry point for the SO-101 twin game. Unless a test harness owns the session (tools/runmode.mjs sets workspace attribute TwinMode = "test"), every player gets their own Twin.Station (table + arm, STATION_SPACING studs apart) and plays LEVELS: one per task in tasks/*/task.json (Twin.Levels, generated Twin.TaskData) plus a 10-second tutorial. A round = a fresh seeded scene from the task's randomize (Twin.Round), judged server-side by Twin.TaskCheck (the Luau port of argon_tasks.check), scored by Twin.Score (stars, combo), recorded in the spec/policy_io.md trace format ext |
| `roblox/src/ServerStorage/TwinConfig.lua` | 36 | Server-only settings for the published SO-101 twin game (ServerStorage: never replicated to clients, so the trace key stays secret). Read by ServerScriptService/TwinMain. |
| `roblox/src/StarterPlayerScripts/TwinTeleop.client.lua` | 1283 | Player client for the SO-101 twin game (server: ServerScriptService/TwinMain). Drives this player's own station through the Station teleop message {move, roll, grip, fine}; horizontal motion is camera-relative (Twin.TouchMap), so "up" always moves the gripper away from the camera. Touch: left joystick = move over the table, UP / DOWN buttons, roll buttons, big GRAB toggle. Keyboard: W/S A/D move, Q/E up/down, Z/X roll, Shift (hold) slow, Space grab, N / Enter = play / next, R retry. Gamepad: left stick move, R2 / L2 up / down, R1 / L1 roll, A grab, Y retry, X (hold) slow, A = play / next. Scre |
| `roblox/tests/gen_taskcheck_vectors.py` | 31 | argon_tasks/tests/vectors.json -> tests/taskcheck_vectors.luau (plain luau has no JSON decoder), for tests/taskcheck_test.luau. Run after the argon_tasks vectors change: py -3 roblox/tests/gen_taskcheck_vectors.py (any cwd) py -3 roblox/tests/gen_taskcheck_vectors.py --check exit 1 if stale |
| `roblox/tests/joint_server_trace_test.py` | 102 | /trace endpoint test for human episodes, the way ServerScriptService/TwinMain + Twin.Station send them (all parts back to back at the end of a round, part 0 starts with the meta line, the last part ends with {"meta_final": ...}). .venv/Scripts/python tests/joint_server_trace_test.py (from roblox/; writes only to a temp dir) |
| `roblox/tests/levels_test.luau` | 167 | Level registry, scene sampling and scoring (Twin/Levels, Twin/LevelGen, Twin/Score, Twin/TaskData). Pure Luau: C:\Users\sarta\g4studio\backend\bin\luau.exe tests/levels_test.luau (from roblox/) |
| `roblox/tests/taskcheck_test.luau` | 75 | Twin/TaskCheck (the Luau port of argon_tasks.check) against the shared vectors (argon_tasks/tests/vectors.json, converted by tests/gen_taskcheck_vectors.py): overall ok and every per_predicate (label + ok) must match. Pure Luau: C:\Users\sarta\g4studio\backend\bin\luau.exe tests/taskcheck_test.luau (from roblox/) |
| `roblox/tests/taskcheck_vectors.luau` | 33644 | GENERATED by roblox/tests/gen_taskcheck_vectors.py from argon_tasks/tests/vectors.json. Do not edit. |
| `roblox/tests/tictactoe_test.luau` | 142 | Tic-tac-toe rules + robot opponent (Twin/TicTacToe): win detection, legality, perfect play at slip 0, slips. Pure Luau: C:\Users\sarta\g4studio\backend\bin\luau.exe tests/tictactoe_test.luau (from roblox/) |
| `roblox/tests/touchmap_test.luau` | 99 | Unit test of the touch / gamepad / keyboard -> station command mapping (src/ReplicatedStorage/Twin/TouchMap.lua). Pure Luau, no Roblox APIs: C:\Users\sarta\g4studio\backend\bin\luau.exe tests/touchmap_test.luau (from roblox/) |
| `roblox/tools/bridge.mjs` | 122 | Studio bridge client (same transport as flycar/tools/bridge.mjs). BRIDGE=server targets the Play-mode server DataModel ("<token>-server"). Every Studio window shares the plugin's session token and commands land in whichever window polls first, so each command is guarded: it errors WRONG_PLACE unless ReplicatedStorage.Twin exists, and is retried. |
| `roblox/tools/gen_levels.py` | 91 | Generate src/ReplicatedStorage/Twin/TaskData.lua from tasks/<name>/task.json (spec/task_spec.md, argon_task/1). py -3 roblox/tools/gen_levels.py (from argon-hack/; or any cwd) py -3 roblox/tools/gen_levels.py --check exit 1 if TaskData.lua is stale Every task.json becomes one entry of the Luau table (the JSON verbatim as a Luau literal, JSON null dropped) plus `order`, the task names sorted by the optional "game": {"order": n} hint then by name. Twin/Levels.lua turns these into game levels, so a new task.json is a new level after a rebuild, with no Luau changes. |
| `roblox/tools/joint_server.py` | 270 | Joint-target server for driving the Roblox twin live (stand-in for the real SO-101 leader arm). .venv/Scripts/python tools/joint_server.py [--port 8765] [--hz 60] [--source sine/leader] [--key SECRET] [--out DIR] Endpoints (all JSON; q in MJCF radians, joint order shoulder_pan .. gripper; t_ms = Unix epoch milliseconds when the sample was produced, so a client on the same machine can measure end-to-end latency): GET /q latest sample {"seq", "t_ms", "q"} GET /since?seq=N all samples after seq N (up to 2 s), for batched polling under HttpService's 500 req/min cap GET /sse Server-Sent Events, one |
| `roblox/tools/load_meshes.mjs` | 51 | Load the per-link OBJs (meshes/*.obj) into the open Studio as EditableMesh-backed MeshParts, edit mode only. node tools/load_meshes.mjs No asset upload: AssetService:CreateEditableMesh + CreateMeshPartAsync from plugin context (no budget in edit mode). The parts live in a parentless Folder kept in the plugin's _G.TWIN_MESHES (with the EditableMeshes in _G.TWIN_EDITABLE, because destroying an EditableMesh blanks its MeshPart), never in the place, so Play mode and saved places are unaffected. spikes/render_scene.luau passes the folder to SceneLoader for real-mesh renders. Vertices are recentred  |
| `roblox/tools/mjcf_to_luau.py` | 471 | Convert a MuJoCo MJCF robot into a Luau data module the Roblox builder instantiates. .venv/Scripts/python tools/mjcf_to_luau.py model/so101_official/so101_new_calib.xml SO101 --fixed-base --mesh-boxes 4 --obj-dir meshes --boxes-json model/so101_boxes.json Generalised from robot-academy/tools/mjcf_to_luau.py (Go2/G1, floating base) for fixed-base arms: - a base body with no joint is emitted `anchored = true` and the frame is the MJCF world frame (no ground shift); - mesh collision geoms become a voxel box decomposition (`--mesh-boxes <mm>`) instead of one AABB, so a gripper jaw keeps its finger |
| `roblox/tools/render.py` | 79 | Render the Roblox twin from a scene_spec camera (edit mode, Studio window capture), for the render-compare loop. .venv/Scripts/python tools/render.py scenes/default.json --cam front --q 0.2,-0.4,0.6,1.0,0,0.4 --out shots/x.png Builds the scene with Twin.SceneLoader, poses the arm kinematically at q (MJCF rad), points the edit camera at the spec camera (vertical FOV = fovy), locates the 3D viewport in the window capture with magenta corner markers, crops the centred region of the spec's aspect ratio (FieldOfView is vertical, so the full height is the spec's fovy) and resizes to width x height.  |
| `roblox/tools/run.mjs` | 21 | Run a Luau file in the Studio edit DataModel (plugin context) and print the result. node tools/run.mjs spikes/foo.luau [timeout_ms] [args.json] (args.json -> _G.SPIKE_ARGS) The file may `return` a value; tables are JSON-encoded by the bridge. |
| `roblox/tools/runmode.mjs` | 64 | Run a Luau file as a server Script inside a real Run-mode session and print what it returns. node tools/runmode.mjs spikes/foo.luau '{"n":16}' [timeout_s] node tools/runmode.mjs spikes/foo.luau @out/args.json -> args from a file PLAY=1 node tools/runmode.mjs ... -> real Play mode (a Player joins) instead of Run mode The file's top-level `return` value is handed back through StudioTestService:EndTest. Its args are available as _G.SPIKE_ARGS. |
| `roblox/tools/shot.mjs` | 13 | Save a Studio window screenshot: node tools/shot.mjs out/x.jpg/out/x.png (the bridge captures the largest visible Studio window; keep the twin's window largest) |
| `roblox/tools/sync.mjs` | 95 | Push src/ into the open Studio place. src/<Service>/<folders>/<Name>.server.lua -> Script src/<Service>/<folders>/<Name>.client.lua -> LocalScript src/<Service>/<folders>/<Name>.lua -> ModuleScript Managed scripts carry the RAManaged attribute; ones no longer in src are removed. |
| `roblox/tools/trace_summary.py` | 83 | Summarise out/traces/*.jsonl (spec/policy_io.md episodes) into out/traces/summary.json and validate them. .venv/Scripts/python tools/trace_summary.py [--dir out/traces] Checks every file: first line is {"meta": ...} with success, every later line a 30 Hz tick with t, q, q_target, ee_cmd, cube_pose (7), bowl_xy (2); tick spacing exactly 1/30 s. |
| `roblox/tools/twin_bridge_server.mjs` | 61 | Private command channel to the twin's Studio window (port 8766). node tools/twin_bridge_server.mjs (leave running) Why: every Studio window shares the studio-bridge plugin token, so shared-bridge commands land in whichever window polls first (another open place can take nearly all of them). bridge.mjs installs, once, a plugin-context loop in the twin's own window that long-polls GET /next here, runs the Luau with loadstring and POSTs /result. Clients POST /submit {code} and get {ok, value / error} back. |
| `scratch_draw_precision.py` | 43 | Idea-1 feasibility: how big a drawing can SO-101 make on the table, and how much does 1 deg of joint error move a pen tip. |
| `tools/assemble_spec.py` | 79 |  |
