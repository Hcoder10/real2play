# Agent K: the learned policy on the real arm (Gemini commands, policy controls), ready to run

Read docs/briefs/COMMON.md, AGENTS.md, docs/INTERFACES.md, scripts/robot/H_HARNESS_REPORT.md, r2p/policy/README.md.
Repo C:\Users\sarta\real2play, `.venv/Scripts/python.exe`. Commit with explicit paths, small commits, pull first
(agents I, J, L work in parallel; the lead edits scripts/lead, tasks/, config/). Full suite before each commit:
`.venv/Scripts/python.exe -m pytest -q tests/core tests/argon tests/perception tests/robot tests/app tests/sim tests/policy`.
NEVER open COM3 or POST to the arm daemon (http://127.0.0.1:8793, GET /state is fine). Deadline 14:00.

## State
- The lead trains the place_on_cell policy on the rig now (3-seed ensemble, ~6000 expert traces, board_pose
  randomization: the whole printed board moves +-25/15 mm, +-8 deg; bottle caps 30-37 mm in a supply zone x 0.11-0.23,
  y -0.142..-0.106). Export lands in exports/<name>/ (config.json declares start_q, obs_const, eval). Until then use any
  export in exports/ or train a tiny CPU one for plumbing.
- Real bench: joint map config/joint_map.json active **vcal_oct25b** (wrist_roll offset -164.4 deg: the old map was
  ~190 deg off in roll; gripper open_rad 2.70: the real jaw opens much wider than the URDF mapping). armd owner limits
  from GET /state "lo"/"hi" (wrist_flex hi 101.8 owner deg binds fingers-down poses near the robot; wrist_roll +-179).
  The proven real path today is scripts/lead/jaw_move.py (our JawIK at the real gripper opening, roll LOCKED at owner
  81.9 deg, sag compensation, re-aim by encoder-measured miss) and scripts/lead/ttt_armd.py (Argon reads board, Argon finds
  caps, place via goto). Two real pick-and-places succeeded; placement error ~14 mm.
- Live policy path (H): scripts/argon/tictactoe.py --real -> RealRobot + Session (commanded-q, gravity comp, guard),
  requires armd released first (owner action at rest).

## Do
1. Make the policy path correct for the REAL joint map: every model-space target the policy emits must map into armd's
   owner limits (lo/hi) - especially wrist_roll with the -164.4 offset and wrist_flex near its limit; the expert's
   _grasp_yaw roll choice vs the real roll range; start_q vs the real rest pose (rest owner [0.1,-89.9,99,-101.3,81.9,2.7]);
   gripper squeeze with the new open_rad. If the trained distribution needs a roll convention change, say exactly what
   (in the sim expert / ik) and implement it behind tests; the lead regenerates traces.
2. A twin rehearsal of exactly what will run live: MockRobot (or the MuJoCo twin as robot) with the real joint map,
   armd owner limits, sag, latency, the CURRENT measured task (tasks/place_on_cell/task.json, board at its measured pose),
   caps at Argon-found supply positions [[0.1274,-0.1272],[0.1748,-0.1245],[0.2205,-0.1248]], all 9 cells: report
   success, placement error, guard clipping, resets, limit hits. Script: scripts/robot/rehearse_policy.py --export X.
3. The live command (one placement, Gemini 4 Argon picks the cell or --cell given) with the export, plus an armd-free
   handover checklist (when the owner releases armd at rest). Prefer a path that does NOT need armd stopped if you can
   make one safely: e.g. a policy runner that streams the policy's absolute joint targets to armd via "joints" commands
   at the chunk rate (armd interpolates at its SPEED and checks clearance; read C:\Users\sarta\argon-hack\deploy\
   scripts\armd.py) - document the latency/fidelity trade-off honestly.
Report: rehearsal numbers per cell, fixes, commits, the exact live commands.
