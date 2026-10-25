# Agent P3: the expert fails under the real-arm bench profile (hard stop 16:30)
Repo C:\Users\sarta\real2play, .venv/Scripts/python.exe. `python -m r2p.sim.traces --task place_on_cell --bench-profile
scripts/robot/K_BENCH_PROFILE.json` on the CURRENT tasks/place_on_cell/task.json: only 68/6000 succeed (57% grasp_miss,
33% dropped_during_carry, 8% goal_not_met); without the profile the same expert succeeds ~90%. Read r2p/sim/expert.py
(deployment profile: model_limits, locked roll, close_grip .00214 / release_grip .4038), r2p/sim/bench.py,
scripts/robot/K_BENCH_PROFILE.json, scripts/robot/K_POLICY_REAL_REPORT.md. Also read runs/argon_step/step_01/sim.json:
gemini-4-argon's REAL measurements today: grasping a 33 mm cap stalls the gripper at owner ~11 with load ~284, closing
shifts the jaw midpoint (-7.6, +3.3) mm toward the fixed jaw, it grasped at jaw z ~14.6-15.5 mm. Find why the expert
can't hold the cap under the profile (gripper close target vs the profile's convention and the twin's actuator, the
locked roll making the jaw axis oblique to the cap, grasp height, fixed-jaw geometry) and fix it so the profiled expert
reaches >= 70% on 64 seeds (probe with ProcessPoolExecutor, 8 workers max on this laptop). Commit with explicit paths;
report root cause + numbers + the exact trace command. Do not touch scripts/lead, r2p/lab, r2p/policy.
