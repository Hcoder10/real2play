# Agent D: Gemini 4 Argon brain (client, real2sim, tic-tac-toe, labeling, sysid)

Read docs/briefs/COMMON.md first.
Own: r2p/gemini.py, r2p/real2sim/, r2p/ttt/, r2p/labeling.py, r2p/sysid/, tests/argon/, scripts/argon/.
Use r2p.model, r2p.tasks, r2p.traces (A), r2p.sim.TaskTwin (B), r2p.perception.CameraModel (C2), r2p.robot.Session
(C1): import lazily, fake them in tests until they land.

Deliver:
1. r2p/gemini.py per AGENTS.md rule 6 + INTERFACES: google-genai client, roles MAIN/SMALL from env (defaults
   gemini-4-argon / gemini-3.8-flash), disk cache cache/gemini/ keyed by sha256(model, config, parts) with a calls.jsonl
   log (never the key), R2P_CACHE_ONLY, QuotaError only for PerDay / limit-0 429s (RPM 429s back off and retry; never
   match the word "billing"), R2P_ALLOW_FALLBACK, bounded thinking (R2P_THINKING default 2048), JSON mode with schema,
   JSON repair via the SMALL role reusing the cached reply, image_part / video_part (mp4 from frames), served_by
   recorded. A FakeGemini for tests.
2. r2p/ttt: geometry from tasks/place_on_cell (cell centres, ids), rules, perfect minimax;
   decide(frame, cam_model, history, robot_mark, robot_desc, human_desc, ask=...) = ONE MAIN call with (a) the scene
   frame with the 9 cells projected + labelled, (b) a perspective-warped top-down crop of the grid, (c) move history and
   piece descriptions -> JSON {board[9], move, why}; validate locally (earlier marks unchanged, at most one new human
   mark, legal move); one re-ask with the exact error; then minimax on the consistent board; print the answering model.
   scripts/argon/tictactoe.py: live game loop (robot via C1's Session with a per-turn place_on_cell task; the robot's
   piece from a measured supply spot as a known pose; operator confirms the human move if Argon is unavailable;
   publishes "argon_says" to a status JSON for the dashboard), --dry-run on C2's synthbench with an oracle board
   reader, --frame-only on a saved image, --piece-size, --robot-desc, --human-desc.
3. r2p/real2sim: session loader (r2p_trace/1 + frames + camera.json), grasp/release anchors from the robot's own
   gripper COMMANDS (most-closed command = empty-closed reference; jaw gap gives object width), detect (one SMALL call
   per frame set -> boxes), the twin loop: iteration 0 from detection + anchors; each iteration renders the twin
   (r2p.sim) over the real frame on critic frames (eval frames held out), one MAIN call critiques and returns a JSON
   patch (object pose/size/friction/mass), apply, score on held-out frames (per-object mask IoU, centroid px, anchor
   error, physics-replay trajectory px). Keep rule: keep only if score improves > 0.003 AND no object's held-out IoU
   drops > 0.10 below its best AND container IoU does not drop > 0.02 when anchors are not enforced AND the camera term
   is not worse AND (anchors enforced) grasp-anchor error not worse + release inside the container; stop on plateau
   (2 iterations without gain). Live output out/live/<run>/: run.json (r2s_live/1: status, stage, iters[{iter, score,
   kept, verdict, best_so_far, thumb}], best_iter, stop_reason, calls{by_model, served_by}, calibration, anchors,
   session_kind, video) + iter_N.json + iter_N_overlay_J.jpg (1280x480: real with outlines | sim) +
   iter_N_physics.jpg, written atomically. task_from_video(session|video, instruction=None): ONE MAIN request with the
   whole episode as video + first/last frame + joint log (5 Hz table) + anchor facts -> task.json validated with
   r2p.tasks.validate (max 2 re-asks with the errors), randomize ranges filled locally, then the twin loop ->
   tasks/<name>/scene.json. CLIs scripts/argon/run_twin.py and task_from_video.py.
4. r2p/labeling.py: segment(trace, task) -> [[t0, t1, command]] from gripper events + motion phases (deterministic),
   with an optional MAIN-role refinement; labels Roblox/human traces with Gemini's command vocabulary.
5. r2p/sysid: experiment planner (excitation trajectories, pushes, grasps where the twin residual is largest) +
   parameter fit of config/physics values (servo kp/kv, sag k, latency, friction) from recorded real attempts vs
   TaskTwin replays; held-out validation (jaw rms mm, object end pose mm, grasp outcome agreement) with a pass/fail gate
   (SPEC section 11).
6. tests/argon with FakeGemini: cache hit/miss/quota semantics, ttt validation + fallback + exhaustive minimax (robot
   never loses), keep-rule cases, anchors from commands, task_from_video validation loop, labeling on synthetic
   traces, sysid recovers known parameters on synthetic data.

Acceptance: `pytest -q tests/argon` passes; `python scripts/argon/tictactoe.py --dry-run --games 2` plays to completion
(once C1/C2 land; until then an oracle-only mode).
