# Agent R: Argon Lab - Gemini 4 Argon experiments on the real arm and builds the twin itself (any task)

URGENT (deadline 14:35). Repo C:\Users\sarta\real2play, `.venv/Scripts/python.exe`. Read AGENTS.md, docs/briefs/COMMON.md,
r2p/gemini.py (Interactions transport; image_part, video_part, frames_to_video; roles: MAIN = gemini-4-argon only),
r2p/agents.py (Antigravity managed agent wrapper), scripts/lead/jaw_move.py (PROVEN real motion: our JawIK at the real
gripper opening, roll locked at owner 81.9 deg, sag compensation, re-aim; speaks the repo daemon), scripts/lead/ttt_armd.py
(Argon finds caps; park()), scripts/lead/active_live.py (Argon cap sensor; sheet guard), r2p/real2sim/active.py
(Validator safety: reach band, vertical moves near the table, grid protection, live jaw-height abort), r2p/sim/twin.py
(TaskTwin, build_xml, load_physics, render), r2p/real2sim/evaluate.py (physics_replay, overlay), config/camera.json,
config/joint_map.json (active vcal_oct25b), config/physics/so101_bench.json. Commit with explicit paths, pull first.
Other agents (O, P) edit other files; do not edit r2p/real2sim/active.py or r2p/policy/*.

## The owner's requirement
Gemini 4 Argon is as capable at robotics/real2sim as a frontier engineer. Let IT decide where the arm goes, what to
experiment, and build the twin itself, starting from a basic sim we provide, until the sim matches the real environment
visually AND physically, for ANY task. Follow Google's "Gemini 4 Argon for robotics" pattern
(https://developers.googleblog.com/gemini-4-argon-for-robotics/): a local harness with HTTP primitives, annotated camera
images, `# Plan:` on turn 1 and `# Thought:` every turn, re-verify visually after every action, recover by re-observing.
Real2sim per the post: replay logged joint commands with joint-space PD; match the camera renders and the object
motion; tune ONLY static assets (geometry, collision, mass, inertia, friction, joints, appearance), initial object poses
and actuator/servo parameters; "no sim cheating" after t=0 (no external forces, no qpos resets, no welds, no disabling
collisions).

## Build (new package r2p/lab/ + scripts/argon/lab.py)
1. Real harness (r2p/lab/harness.py, HTTP on 127.0.0.1:8081): GET /obs -> {ee_pos (jaw midpoint, spec m), gripper
   (0-100), holding (Argon/vision or gripper stall), status, workspace bounds, scene_cam_path, wrist_cam_path (latest
   frames saved with a world X/Y grid in cm + axes drawn using config/camera.json; wrist frame raw), last_episode}.
   POST /plan with primitives (metres, spec frame: +x away from base, +y left, +z up): goto(x,y,z,speed),
   move_by(dx,dy,dz,speed), grasp(x,y,z) [open, hover 45 mm, straight vertical descent, close, verify stall, lift],
   place(x,y,z) [hover, vertical descent, open, lift], place_until_contact / move_until_contact (descend in 2 mm steps
   until the encoder lag says contact, then relieve), look_at(x,y,z) (move so the wrist camera sees the point),
   deproject(u,v,z_plane,camera) -> (x,y), open_gripper, close_gripper, home (park high, then rest), done. Implement
   motion with scripts/lead/jaw_move.py's goto/gripper/park (import them). Safety ENFORCED here, not in prompts:
   reach band, finger height >= 6 mm except inside grasp/place descents at their own xy, protected regions (task
   targets/printed grid) only entered by vertical grasp/place descents at their own xy, live abort on measured jaw <
   7 mm outside grasp/place, speed caps, the daemon guard's refusals returned as status, sheet-moved guard via
   scripts/lead/measure_grid.py measure() only when the arm is away from the board. Every /plan call is RECORDED as an
   episode: 30 Hz /state (q, q_target, xyz) + scene camera (http://127.0.0.1:8792) + wrist camera (http://127.0.0.1:8794)
   frames with timestamps -> runs/lab/<run>/episodes/<n>/ (frames as JPG, states.jsonl, primitives.json, mp4s).
2. Sim workspace (r2p/lab/simws.py): runs/lab/<run>/sim/ starts as a BASIC MuJoCo scene we generate (the calibrated
   SO-101 from r2p/sim/twin.py build_xml with the current physics, a table, the calibrated camera as a MuJoCo camera,
   an optional wrist camera on the gripper, nothing else). Argon OWNS these files: it may write/replace scene.xml
   includes, object MJCF (primitives or meshes), materials/textures (it can crop textures from the real frames),
   physics json (servo kp/kv/sag, frictions), lighting. Tools: sim_render(camera) -> image; sim_plan(primitives) ->
   rehearse the same primitives in the sim with the same IK and return the rendered video + final state;
   replay(episode) -> drive the sim joints toward the episode's logged q_target with joint PD at the logged times,
   render scene+wrist cams, write side_by_side.mp4 (real | sim, both cameras) and residuals (joint RMS deg, jaw path
   RMS mm, object final pose error mm from Argon's own reads of the real frames, image similarity); score(split) over
   held-out episodes; edit_sim(path, content) restricted to the sim dir with an XML/JSON validity check and a "no sim
   cheating" lint.
3. The agent loop (r2p/lab/agent.py, `python scripts/argon/lab.py --task <task.json> --run <name> [--max-turns N]
   [--approve ask|auto-sim]`): gemini-4-argon via r2p/gemini.py, thinking_level high, server-side state if available.
   Each turn: send the goal (the task file + "make the sim match this real environment visually and physically for
   this task; experiment with the real arm as needed"), /obs, the annotated images, the latest replay video and
   residuals, the sim file list; Argon returns JSON {thought, plan (turn 1), calls:[{tool, args}], done, why}. Execute
   calls in order (real_plan requires owner approval per call in --approve ask: print the primitives and wait for y;
   sim tools run freely), feed results back. Held-out: Argon must keep >= 2 recorded episodes it never edits against;
   done requires held-out residuals to improve and Argon's own statement. Log every turn to runs/lab/<run>/ledger.jsonl
   and status.json (dashboard), keep the video files for the demo. Antigravity managed agent (r2p/agents.py): when Argon
   asks for heavy sim code (e.g. a mesh generator or a new MJCF from photos), it may call tool `managed_build(spec)` that
   runs the managed agent on the sim dir contents and returns files, which still go through edit_sim validation.
4. A sim-only dry mode (--real-mock: the "real" side is a hidden-truth MuJoCo copy with different friction/sag/visual
   colour) used by tests (fake Argon) so the loop is verified without hardware, plus one live Argon dry-mode turn.
Report: commands, a dry transcript excerpt (Argon thoughts + tool calls), commits, and the exact real command.
