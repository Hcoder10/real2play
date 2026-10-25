# Agent B: reference sim, experts, traces, policy

Read docs/briefs/COMMON.md first.
Own: r2p/sim/, r2p/policy/, config/physics/, exports/, tests/sim/, tests/policy/. Use r2p.model / r2p.frames (lead)
and r2p.tasks / r2p.traces (agent A; if not committed yet, code against docs/INTERFACES.md and poll `git log`; A
commits within ~30 min).

Deliver:
1. config/physics/so101_bench.json: per-joint position servo (kp Nm/rad, kv, armature, frictionloss, torque limit),
   latency_ticks, default object mass/friction, contact solref/solimp. Choose kp so that STATIC SAG EMERGES from the
   servo model and matches the measured real arm: jaw midpoint 9-12 mm below the commanded height at ~0.30 m reach,
   ~3 mm at 0.18 m (equivalently k = 1/kp ~ 0.05 rad/Nm on lift/elbow/wrist_flex). Document how to refit from real
   data (r2p.sysid will fit them).
2. r2p/sim/twin.py TaskTwin per INTERFACES: the official MJCF (assets/so101/so101_new_calib.xml) placed so the spec
   frame origin is the footprint (r2p.frames), table plane at z=0, task objects of every shape (bowl = disc + 16 wall
   staves; plate/zone non-colliding when "collide": false; stick = square prism), physics config applied (servo
   actuators, sag emerges), PHYSICAL contact grasps (test: a 38 mm cube and a 30x30x14 mm piece lifted 5 cm, held
   2 s), 30 Hz step with substeps, poses() in the spec frame, render(camera) via mujoco.Renderer.
3. r2p/sim/expert.py: Expert(task) plays the skeleton: pick (top grasp, jaws across the narrow side via wrist roll,
   descend so fingertips are ~3 mm above the table for low objects, close, lift), place_in / place_on / place_at
   (square the held object to the target, lower, open only to 0.5 rad on release, retreat up), push_to, release, home.
   Damped least-squares IK on the jaw midpoint (r2p.model.Arm.jac_tip) with fingers-down in the null space. ABSOLUTE
   targets that still land accurately WITH sag (closed-loop on measured q or sag-aware). DART: perturb executed q only;
   labels = q_target. Emit command segments ("pick(piece)", "place_at(piece,r1c2)", ...). Targets: success >= 0.9 on
   cube_in_bowl and place_on_cell (median placement error <= 6 mm) over 100 sampled scenes; report all six tasks.
4. Trace generation: `r2p.sim.gen_traces(task, n, out_dir, workers<=4)` -> r2p_trace/1 (A's format) with commands.
5. r2p/policy: command-conditioned state policy. Input: commanded q (6) + K x 9 slots + command one-hot + FK features
   (jaw midpoint, finger axis) + yaw-symmetry features; output 15 x 6 ABSOLUTE joint targets. MLP width 512,
   3 residual blocks, GELU; torch for training (install with
   `uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu128`;
   in YOUR tests train tiny on CPU only). Augmentations: joint-offset noise, stale/noisy object poses, aim relabel,
   recovery up-weighting, chatter smoothing. obs_const clamp. Export per INTERFACES (pure-numpy runtime.py; ensembles
   of N seeds). Eval in TaskTwin from start_q with noise {clean, real, realx0.5}, recorded into the export config.
   CLI: `python -m r2p.policy.train --task T --traces DIR --name NAME --steps S --device cuda|cpu`;
   `python -m r2p.policy.eval --export NAME --n 100 --noise clean`.
6. tests/sim, tests/policy: grasp holds, sag within band, expert success on 10 scenes per task (fast), trace format,
   tiny CPU train + export + numpy/torch parity (1e-4) + eval runs.

Acceptance: `pytest -q tests/sim tests/policy` passes; `python -m r2p.sim.expert --task place_on_cell --n 100` reports
success >= 0.9 (report the honest number and the main failure modes if not).
