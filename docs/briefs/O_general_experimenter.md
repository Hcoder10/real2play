# Agent O: Argon experiments for ANY task and builds the best twin for it (for the next run)

Read docs/briefs/COMMON.md, AGENTS.md, docs/INTERFACES.md, scripts/argon/I_ACTIVE_TWIN_REPORT.md, r2p/real2sim/active.py
(engine), scripts/lead/active_live.py (real runner: Argon is the cap sensor, re-survey before each experiment, sheet-moved
guard), r2p/tasks (task schema: argon_task/1, objects/roles/shapes, skeleton of skills, randomize, success predicates),
tasks/*/task.json. Repo C:\Users\sarta\real2play, `.venv/Scripts/python.exe`. Commit with explicit paths, small commits,
pull first. A REAL campaign is running right now from r2p/real2sim/active.py and scripts/lead/active_live.py: do NOT
edit those two files in place until it finishes (check `runs/active_live_02.log` for a final JSON line); build in new
modules and switch over at the end. NEVER POST to the arm daemon (:8793) or touch COM3; read-only GETs of :8792 (phone)
and :8794 (wrist camera) are fine. No live Gemini in tests; live budget 15. Deadline 14:30.

## Owner's rules
- Argon experiments ONLY for what the chosen task needs, in THAT environment: the experiment menu is derived from the
  task (its skeleton: pick / place_at / place_in / push / stack ..., its objects, its success predicates) and the
  surveyed scene, never a fixed generic list. Today's place_on_cell menu (in active.py, Oct 25) is the hand-written
  example: holds on the task's own path (over the supply lane and the board), a sweep of the joint that matters, and
  the task's own pick-and-place motions onto the task's targets and back. Generalize that: e.g. cube_in_bowl -> picks
  and drops into the bowl (bowl wall contact), push_to_target -> real pushes at a safe contact height with standoff,
  stack_two_cubes -> stacking/settling. Parameters fitted = the ones those motions excite (per-skill mapping).
- Real-bench safety lessons that every generated experiment must respect: vertical approaches/retreats near the table
  (joint-space interpolation at low height swept the fingers and dragged the paper); contact heights above the servo sag
  (commanded 8 mm ran the fingers 4 mm over the paper); live abort on measured jaw height; reach limits near the base
  (fingers-down contact unreachable < ~12.7 cm from the base with the calibrated wrist limit); the grid/targets are
  protected except the experiment's own vertical pick/place spots; sheet-moved guard after every experiment.
- Argon designs: give Argon (MAIN, thinking high) the task JSON, the surveyed scene, the twin's current parameters with
  uncertainty, the residual history and the safety rules; it PROPOSES task-relevant experiments in a typed schema
  (validated by our Validator; refusals fed back); a deterministic task-derived menu is the fallback.
- Both cameras: record the wrist camera (http://127.0.0.1:8794) alongside the scene camera in every real experiment
  session (frames + timestamps), and give Argon wrist frames where they help (grasp confirmation, slip, release).
- Sim perspective for the demo: for every experiment render the twin's replay (before-fit and after-fit) as an MP4
  next to the real recording, plus a side-by-side real | sim-before | sim-after video per experiment
  (runs/active_twin/<run>/videos/). A script scripts/argon/twin_videos.py <run> builds them from an existing run.
Report: the new modules, a sim campaign for cube_in_bowl and place_on_cell with the generated menus, the video script
output on runs/active_twin/bench-active-02 (once it exists), commits.
