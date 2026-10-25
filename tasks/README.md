# Shared task and trace contracts

All coordinates are metres in the robot base footprint frame, with x forward,
y left and z up. The table surface is z=0. This module applies no frame offset.
Task files use `argon_task/1`; scenes use `version: 0`; traces use `r2p_trace/1`.

```powershell
.venv/Scripts/python.exe -m r2p.tasks validate tasks/*/task.json
.venv/Scripts/python.exe -m r2p.tasks sample place_on_cell --seed 12 --out scene.json
.venv/Scripts/python.exe -m r2p.tasks check cube_in_bowl state.json
.venv/Scripts/python.exe -m r2p.tasks.make_vectors --check
.venv/Scripts/python.exe -m pytest -q tests/core
```

`check` takes one state JSON object, or a list of states to check the final
half-second. The CLI exits 1 for an unmet goal and 2 for malformed input.
`load_task` accepts a name, directory, JSON file, or dictionary, validates it and
returns an independent copy. `validate` returns an error list.

## Geometry and scenes

`slot_size(shape, size_m, axis="z")` returns full local xyz extents:
box/plate `[x,y,z]`, sphere `[d,d,d]`, bowl `[inner_d,inner_d,h]`, stick
`[length,width,width]`, cylinder `[d,d,h]` for z, `[h,d,d]` for x, `[d,h,d]` for y.
Sticks are square prisms. Bowl `wall_m` is separate from its inner diameter.
`yaw_symmetry(obj)` returns 4 for square boxes/plates, 2 for oblong boxes/plates,
sticks and lying cylinders, and 0 for round upright objects.

`sample_scene(task, rng)` uses only the supplied NumPy Generator. It supports
polar or Cartesian ranges, discrete `xy_choice`, `p_present`, yaw or bearing-relative
yaw, per-dimension size ranges, uniform size scaling, mass and friction. Rejected
placements restart the scene, retaining the episode's sampled presence flags.
It raises if placement cannot satisfy the constraints after 500 attempts.
`min_sep_m` is the gap between conservative circular footprints. Non-colliding
target markers and deliberate support chains (`on`) do not block placement.
Two `xy_choice` objects reserve distinct spots, including non-colliding targets.
Optional objects with `p_present: 0` need no position distribution.

`poses_from_scene(scene)` produces the pose mapping accepted by encoding/checking.
`pose_of(pose)` returns `(xyz_array, yaw_degrees)`; it accepts a pose dictionary,
xyz, or `[x,y,z,qw,qx,qy,qz]`. Quaternions are normalized.

## Observation and command layout

`encode_slots(task, poses)` returns float32 `(K,9)`, where K is
`limits.max_objects`. Rows follow task object order, including absent objects:
`[x,y,z,sin(yaw),cos(yaw),sx,sy,sz,present]`. Absent and padding rows are all zero.
A pose's `size_m` overrides task dimensions. Stick yaw is canonicalized to
`[-90,90)` before encoding. `encode_obs` returns q(6) followed by flattened slots;
when `command` is supplied, it appends the command vector. q is the caller's
commanded joint state, in MJCF radians.

`commands(task)` returns every skeleton step in order, including repeated release
commands. Examples: `pick(piece)`, `place_at(piece,cell)`, `wait(0.5)`, `home`.
`command_onehot` accepts these strings or a skeleton entry dictionary.

Command encoding has **8 * (K+1)** entries, in this fixed skill order:
`pick, place_in, place_on, place_at, push_to, release, home, wait`.
Index = `skill_index * (K+1) + object_slot`; the final slot K means no object for
release/home/wait. Target arguments live in the task and target pose slots, so
they do not create new command dimensions. Object names and skeleton order do
not affect indexes. `command_vocab(task)` returns export labels such as
`pick(slot:0)` and `home(none)` in exactly that order. Some coordinates are
reserved; all tasks with the same K share the same coordinate meaning.

## Success and portable vectors

`check(task, state)` returns `ok` and ordered `per_predicate` entries with `pred`
and `ok`. Goal labels are bare predicate names; duplicate predicates are
distinguished by their position in the goal list. Additional diagnostic fields
may explain absent objects or malformed evidence. All tolerances are strict.

The ten geometric predicates follow SPEC Appendix B: `inside`, `on_top`, `at`,
`near`, `lifted`, `grasped`, `upright`, `yaw`, `not_moved`, `stacked`.
`gripper_open` may also be an explicit goal. Otherwise it is appended when
gripper state is supplied, unless any goal is `grasped`.
Flat `gripper_open`/`site_xyz`/`holding` and nested `gripper.open`/`site_xyz`/`holding`
are accepted; flat values take precedence. Physical simulations should supply
their actual `holding` list for grasp checks. Without it, the geometric grasp
test requires closed jaws, proximity AND the object's bottom above 5 mm.

`settled` is always appended: goal-referenced objects with supplied scalar or
vector `v` must move slower than 0.5 m/s. Unobserved velocities add no condition.
An optional goal passes when any referenced object is absent. Present optional
objects receive the same checks as required ones. `not_moved` needs `initial`.
`check_window(task, states, hz=30)` requires all predicates on the final
`ceil(0.5*hz)` ticks; shorter histories fail. It uses the first state of the whole
history as the initial pose unless a tick supplies its own `initial` mapping.

`r2p/tasks/vectors.json` contains `cases` and `window_cases`. Every case includes
an inline task and an independently assigned `expected: {ok, per_predicate}`.
Normal cases carry `state`; window cases carry `states` and `hz`. Compare the
overall boolean and every ordered `(pred, ok)` pair, ignoring diagnostic fields.
The generator checks its independent expectations against Python before writing.

## Built-in tasks

Six tasks cover a 38 mm cube and 110 mm bowl, a 120 mm square-prism stick,
stacking, pushing, two cubes in a 140 mm bowl, and tic-tac-toe placement.
Object centres remain in the 0.15–0.32 m reach interval when sampled.

`place_on_cell` has 15 slots: `piece`, highlighted `cell`, the nine fixed
non-colliding grid plates (`r0c0` through `r2c2`), and `other1`–`other4`.
The grid is centred at `(0.245,0)` with 45 mm pitch. `board` metadata repeats
these coordinates for board-reading callers. The default piece is a
30 x 30 x 14 mm box at `(0.23,0.15)`; callers can override its shape and dimensions.
The highlighted target and optional obstacles sample distinct cells. Success is
`at(piece, cell, 0.012)` and optional `not_moved(otherN, 0.01)`.

## Trace files and upload parts

`r2p.traces.write_trace(path, meta, ticks)` writes atomically after validation.
`read_trace(path)` returns `(meta, ticks)`; `validate_trace(meta, ticks)` returns
errors. The writer accepts NumPy values and tick iterables. It supplies explicit
defaults for optional metadata and null command/phase values, while requiring
task, source, measured q, q_target, poses and grip_closed. The reader requires the
canonical format. It never substitutes q or q_exec for a missing q_target label.
Timestamps must be nonnegative and spaced at 30 Hz, allowing 10 microseconds of
serialization rounding. Metadata command intervals must be ordered and disjoint.

Line 1 is `{"meta": {...}}` with schema, task, task_name, source
(`expert|human|bot|real`), player, level, seed, scene, success, fail_reason, commands
and rig. Every later line contains t, q(6), q_target(6), poses, grip_closed,
command and phase. Poses contain xyz and yaw_deg. Additional fields are preserved.

`join_upload_parts(parts, path=None)` (also `join_roblox_parts` / `join_trace_parts`)
accepts ordered text/bytes chunks, `{part_number: chunk}`, or numbered
`{part, data}` records. Raw chunks concatenate verbatim, even inside a UTF-8
character or a JSON line. It also accepts Roblox HTTP payloads:
`{trace_id, part, final, meta?, ticks, meta_final?}`. Parts may arrive out of order
but must be contiguous from zero without duplicates; only the last HTTP payload
has `final: true`. An optional final `meta_final` updates completion metadata,
but cannot change trace identity. The helper returns `(meta, ticks)` and optionally
writes a canonical file; it performs no HTTP requests.
