# Agent E local acceptance — 2026-10-25

Run commands and the owner publish checklist are in [README.md](README.md).
Raw local reports and canonical traces are retained under `roblox/out/`.

- All five Luau suites pass: 576 dynamics assertions, 1,807 scene/slot/release checks,
  442 checker cases plus 11 windows (3,143 assertions), 23,165 tic-tac-toe
  checks, and 27 numeric touch checks plus invalid-input checks.
- All eight Python tests pass, including authenticated chunk upload, UTF-8,
  retry conflicts, immutable provenance, fixture exclusion and parity failures.
- The real Play smoke passes: a Player sits, controls/camera activate, a full
  match completes, authenticated traces arrive, and standing restores the camera.
  Placement fixtures are explicitly `source=bot`, `test_only=true`; the latest
  fixture is 16 mm off-centre to exercise the game's 18 mm tolerance.
- Eight independently advancing tables remain manually server-owned in Run mode:
  123 control ticks each. The test also verifies attract mode records zero ticks.
- 36 scene checks verify masses, canonical poses, server ownership and that waking
  an airborne object changes no pose, velocity or collision setting. Six joint
  checks verify reset clears the preceding round's motor targets.
- The fresh-preset / actual Studio parity / Rojo build gate passes. The generated
  place is `roblox/real2play.rbxl` (ignored binary artifact).

## Measured parity

`out/publish_parity.json` replays 180 identical commands and the same cube-drop
scene in MuJoCo and actual Roblox constraints. Inputs and both trajectories are
retained beside the report.

| Measurement | Result | Limit |
|---|---:|---:|
| Joint RMS | 0.148043 degrees | 1 degree |
| Jaw RMS | 1.468900 mm | 2 mm |
| Cube final position | 4.972018 mm | 5 mm |
| Cube final orientation | 0.006346 degrees | 5 degrees |

The object position result is close to the limit. This is a short engine smoke
gate, not a claim of comprehensive contact equivalence. `build.py` fails closed
when a future replay exceeds a limit. The pure Luau numerical run separately
measured 0.008664 degrees joint RMS and 0.072527 mm jaw RMS.

The physics fingerprint matches MuJoCo experts and policy exports:
`8f42f7a072096cbeeb3fb0376286a5cf1c7cde3aa37d74aed2268c5e33cb62d4`.

## Physical bot attempts

The current task describes the camera-measured printed board: nominal 32.8 mm
cells, centre (0.1778, -0.0135) m, 25 x 25 x 14 mm pieces, and supply at
(0.13, 0.13) m. Cell selection, marks and recorded slots consume that preset.
Overlapping 18 mm acceptance regions choose the nearest cell; an occupied
nearest cell cannot be reassigned to a free neighbour.

The six-task run in `out/bots_verified/` passed five tasks and exposed a placement
release failure. After correcting release, limiting thin-piece joint motion and
clearing old motor targets, `out/placement_validation/` passed all three placement
seeds. These are separate runs; they are not a six-task success-rate benchmark.

| Task | Seed(s) | Physical success time | Evidence |
|---|---|---|---|
| cube_in_bowl | 101 | 14.23 s | `out/bots_verified/` |
| pick_place_stick | 101 | 8.70 s | `out/bots_verified/` |
| push_to_target | 101 | 37.97 s | `out/bots_verified/` |
| stack_two_cubes | 101 | 13.47 s | `out/bots_verified/` |
| two_cubes_in_bowl | 101 | 28.07 s | `out/bots_verified/` |
| place_on_cell | 101, 102, 103 | 15.60, 23.60, 27.00 s | `out/placement_validation/` |

Every listed attempt arrived through the authenticated chunk receiver and was
read back with `r2p.traces.read_trace`. These are physical contacts, with no
attachments or object teleports. Earlier failed attempts, including the rejected
narrow-pad geometry experiment, remain under `out/board_*` and
`out/physical_pads/`. The shipped arm retains voxel collision geometry.

The Play smoke's latest report is `out/smoke/result.json`: three fixture move
traces arrived. Fixtures are excluded from the receiver's training statistics.
The full match tests game rules and transport; the bot attempts above test
physical manipulation separately.

## Owner work

These checks do not establish a multi-seed success rate, real-robot calibration,
or eight networked clients. Public publishing, account asset permissions,
DataStore persistence in a published session, and hands-on phone/gamepad visual
checks remain owner steps. The native desktop capture helper was unavailable;
client GUI and camera assertions were exercised in Studio instead.
