# Agent V: Argon-seeded, non-blocking cap perception for the live policy runner (deadline 16:15)

Repo C:\Users\sarta\real2play, `.venv/Scripts/python.exe`. Read scripts/robot/run_real_policy.py, r2p/robot/policy_vision.py,
r2p/perception/task_perception.py (TaskPerceiver: estimate(frame, q) -> {"poses": {id: {xyz, yaw_deg, source, seen}},
"masks"}; known poses, held-object tracking), scripts/lead/ttt_armd.py (locate_caps: gemini-4-argon finds white caps,
deprojects the rim at 17 mm), scripts/lead/active_live.py (ArgonCapPerceiver). Commit with explicit paths, pull first.
Do NOT edit scripts/lead/*, r2p/lab/*, r2p/policy/*. NEVER move the arm (no POSTs to :8793), no GPU.

Problem: white bottle caps on white paper defeat the colour-segmentation TaskPerceiver, so run_real_policy.py cannot
get a cap pose (it requires --perception-setup and a "vision" pose). The session calls perceiver.estimate every 30 Hz
control tick, so a blocking Argon call (3-5 s) cannot go there.

Build r2p/robot/argon_perception.py: ArgonPerceiver(camera, task) with the TaskPerceiver estimate() contract:
- setup/first call: one Argon read (locate_caps logic, MAIN role, thinking low) picks the supply cap nearest the task's
  piece xy; report it with source "vision", seen True.
- non-blocking afterwards: a background thread re-reads with Argon every ~4 s on the latest frame while the cap is on the
  table; estimate() always returns immediately with the latest pose.
- while held (gripper closed on the cap: gripper target << measured, near the jaw), the piece pose follows the jaw
  midpoint from FK of the measured q with the offset fixed at grasp time (source "vision+fk", seen True); after release it
  stays where it was released until the next Argon read confirms it.
- board cells come from the task (fixed targets).
Add `--argon-perception` to scripts/robot/run_real_policy.py (no --perception-setup needed then) and use it in both
--check and live paths. Tests with a fake Argon client and MockRobot: estimate never blocks (> 25 Hz with a 3 s fake
Argon latency), held tracking follows the jaw, release keeps the pose. Report the exact live command for an export
(e.g. wide_v4) with Argon deciding the cell.
