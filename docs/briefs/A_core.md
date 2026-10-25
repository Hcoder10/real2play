# Agent A: core (tasks, success checker, traces)

Read docs/briefs/COMMON.md first.
Own: r2p/tasks/, tasks/, r2p/traces.py, tests/core/ (not test_model.py). Others are blocked on you: commit the API
skeleton with working load/validate/encode/check FIRST (target 30 min), then complete.

Deliver exactly the A sections of docs/INTERFACES.md:
1. r2p/tasks: load_task, validate (closed vocabulary of SPEC section 4 / Appendix A task_spec: shapes
   box/cylinder/sphere/bowl/plate/stick, roles, grasp, the 11 goal predicates incl. optional goals and the gripper_open
   rule, skills, randomize incl. xy_choice, p_present, min_sep_m, xy as object id), sample_scene (deterministic per
   rng), encode_slots/encode_obs (slot_size per shape; yaw for sticks in [-90, 90)), yaw_symmetry (k=4 square box,
   2 stick/oblong, 0 round), commands(task) and command_onehot (fixed vocabulary: skill x object-slot index), check and
   check_window (final 0.5 s at 30 Hz), CLI `python -m r2p.tasks validate|sample|check`.
2. tasks/<name>/task.json: cube_in_bowl (38 mm brown cube, blue straight-walled bowl inner 110 mm x 30 mm, wall 5 mm),
   pick_place_stick (12 cm square-prism stick, grasp "along", place_at a target zone plate), stack_two_cubes,
   push_to_target, two_cubes_in_bowl, place_on_cell (tic-tac-toe: 9 cells as fixed target plates "collide": false,
   45 mm, centred (0.245, 0), ids r<i>c<j> with i along x .200/.245/.290 and j along y -.045/0/+.045; the robot piece
   "piece" (box 30x30x14 mm default, overridable) starts at a supply spot x .23 y .15; target cell = a slot drawn over
   the 9 cells via xy_choice; up to 4 other pieces on other cells as optional obstacles p_present 0.5; goal
   at(piece, cell, 0.012) + optional not_moved(others, 0.01)). Spec frame (footprint). Reachable r in [0.15, 0.32] m.
3. r2p/traces.py: r2p_trace/1 read/write per INTERFACES (jsonl, meta line first), validation, and a helper to join
   Roblox upload parts.
4. r2p/tasks/vectors.json: >= 300 hand-built + generated check cases (each with its task inline and the expected
   per-predicate result) incl. every predicate, pairs 0.5 mm / 0.5 deg either side of every tolerance, yaw wrap at
   +-180, optional goals, and >= 5 check_window cases. Generator: r2p/tasks/make_vectors.py.
5. tests/core/: validation of all tasks, sampling determinism, encode shapes, every predicate, vectors
   self-consistency, trace round-trip.

Acceptance: `.venv/Scripts/python.exe -m pytest -q tests/core` passes, and
`.venv/Scripts/python.exe -m r2p.tasks validate tasks/*/task.json` prints ok for all six.
