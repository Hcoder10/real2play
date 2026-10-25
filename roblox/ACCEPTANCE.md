# Agent E local acceptance — 2026-10-25

Run commands and the owner publish checklist are in [README.md](README.md).
Raw local reports and canonical traces are retained under `roblox/out/`.

- All five Luau suites pass: 576 dynamics assertions, 1,804 scene/slot checks,
  442 checker cases plus 11 windows (3,143 assertions), 23,179 tic-tac-toe
  checks, and 27 numeric touch checks plus invalid-input checks.
- All eight Python tests pass, including authenticated chunk upload, UTF-8,
  retry conflicts, immutable provenance, fixture exclusion and parity failures.
- The real Play smoke passes: a Player sits, controls/camera activate, a full
  match completes, authenticated traces arrive, and standing restores the camera.
  Placement fixtures are explicitly `source=bot`, `test_only=true`; the latest
  fixture is 16 mm off-centre to exercise the game's 18 mm tolerance.
- Eight independently advancing tables remain server-owned in Run mode.
- Scene checks verify masses, canonical poses, server ownership and that waking
  an airborne object changes no pose, velocity or collision setting.

The physics fingerprint matches MuJoCo experts and policy exports:
`8f42f7a072096cbeeb3fb0376286a5cf1c7cde3aa37d74aed2268c5e33cb62d4`.

`out/bots_acceptance/` records six successful physical task attempts at seed 101
before the concurrent bench board relocation. The updated board is centred at
x=0.160 m, with its supply at (0.13, 0.13). Game rules now consume those task
coordinates directly. Physical thin-piece handling on that layout is being
rechecked; failed attempts remain recorded under `out/board_*`.

These checks do not establish a multi-seed success rate, real-robot calibration,
or eight networked clients. Public publishing, account asset permissions,
DataStore persistence in a published session, and hands-on phone/gamepad visual
checks remain owner steps. The native desktop capture helper was unavailable;
client GUI and camera assertions were exercised in Studio instead.
