# Agent Y — measured cap workshop, 2026-10-25

The place is built from `roblox/default.project.json` to
`out/roblox/Real2Play.rbxlx`. Owner publishing steps are in
`out/roblox/PUBLISH.md`. No Open Cloud publish was attempted because all three
Roblox environment variables were absent; there is no new published version.

## Changes

- Applied the managed agent's two candidate files only after checking its
  passing verification result and each file's recorded SHA-256. The game
  preset is further adapted for the judge's default cap-placement flow.
- Selected a committed copy of the Argon step_01 physics preset, with
  fingerprint `5556cc8a03587754a0928ce9691ae0fd06fda8caddd997e06967711fb7f0ef82`.
  The parity runner now uses the workshop-selected physics for both engines.
- Added a measured-bench overlay: 36.8 mm rotated cells, all nine Argon cell
  centres, 33 × 17 mm white cylindrical caps (3 g, friction 0.6), and supply
  positions (0.142, −0.068) and (0.218, −0.075) m. The cap at (0.155, 0.008)
  is already on the board and is deliberately excluded from the supply lane.
- Preserved Argon's arm observations in task/trace metadata. The real runner
  documents +10/+11 mm as a **held-cap correction**. It is not another robot
  base/world transform. The −1.5 mm residual is likewise retained as an
  observation. Physical contacts and the measured object-to-jaw offset govern
  placement; the footprint-frame conversion is unchanged.
- Corrected cylinder extents used by tic-tac-toe marks, rendered white caps
  as white, rotated board tiles with the measured board, and distinguished the
  highlighted target from the surrounding cells.
- Added per-table score/streak labels and HUD, a fresh server leaderboard,
  explicit spectator prompts/camera/exit, and a first action that starts the
  highlighted-cap level. The first solo tutorial uses configurable seed 101;
  later rounds sample the measured layout. Twenty independent seats remain available.
- Created ignored server-only `LocalConfig` from `out/demo/trace_key.txt` and
  the requested tunnel. No key appears in source control or client modules.
- Corrected grip-only control to preserve the fixed finger while the jaw
  midpoint shifts. Short caps use a slower pickup and measured stall pressure.
  The measured cap level uses the repository's permitted rigid-in-gripper-frame
  model, acquired only after two ticks of bilateral physical finger contact.
  Acquisition/release assign no object pose, the measured jaw opening is a
  closure floor, and opening releases the cap. Other levels remain physical.
  Each trace declares its `grasp_model`.
- Trace uploads expose completion state; collection waits for all chunks and
  server shutdown drains pending uploads for up to 25 seconds.

## Verification

- `.venv/Scripts/python.exe roblox/tools/test.py`: all six suites pass:
  1,311 measured-bench object checks; 576 dynamics assertions; 2,020 level,
  slot and grasp/release assertions; 442 shared checker cases and 11 windows
  (3,143 assertions); 23,165 tic-tac-toe checks; touch mapping/validation.
- `.venv/Scripts/python.exe -m pytest tests/roblox -q`: 9 passed.
- Pure Luau parity: 0.008664° joint RMS, 0.072527 mm jaw RMS.
- Final build's actual Studio parity: 0.147763° joint RMS, 1.466199 mm jaw RMS,
  4.971925 mm cube final-position error, 0.004208° cube rotation error.
  All are within the unchanged 1° / 2 mm / 5 mm / 5° limits.
  The cube position is close to the limit; the build continues to fail closed
  on a future regression. The earlier run is retained separately.
- Play smoke: a real Player sits, completes the default highlighted-cap
  fixture, receives score/streak, starts a full tic-tac-toe match, uploads
  four validated traces, stands, spectates table 2 without control authority,
  and leaves the spectator view with the walking camera restored.
- Concurrent-table smoke: 8 independent tables each advanced 122 control
  ticks with manual server ownership. Attract-mode traces recorded zero ticks.
- Scene smoke: 36 shape/pose/mass/ownership/wake assertions and 6 reset-joint
  assertions passed.
- Contact-grasp smoke rejects proximity and single-finger contact, requires
  persistent bilateral collider overlap, preserves acquisition/release poses,
  and checks the fixed-gripper weld and jaw floor.
- The HTTPS tunnel returned healthy status and authenticated `/stats` HTTP
  200. An actual Play fixture was uploaded through the public tunnel and
  acknowledged complete. It is explicitly `source=bot`, `test_only=true`,
  and excluded from training statistics.

Raw reports: `roblox/out/agent_y_parity.json`,
`agent_y_studio_parity.json`, `publish_parity.json`, `agent_y_concurrency.json`,
`agent_y_scene_smoke.json`, `agent_y_tunnel.json`, and `smoke/result.json`.
Traces and trajectories remain beside the reports.
The gated build passed, and `agent_y_build.json` records the artifact hash and
XML verification that the deployment key occurs only in ServerStorage/LocalConfig.

## Limits

Fixtures validate the success checker, match loop, UI state and transport;
they are not evidence of physical manipulation success. Physical cap attempts
are retained separately under `roblox/out/agent_y_caps*/`.
The initial contact-only cap runs failed (one timeout, two falls); those are
retained in `agent_y_caps`. Seed 101 subsequently completed a contact-confirmed
rigid cap placement in 22.17 seconds (`agent_y_caps_contact`) with a complete
authenticated trace. Follow-up seeds 102/103 (`agent_y_caps_verified`) produced
one 40-second timeout at the nearer supply cap and one success in 15.10 seconds.
All three traces arrived and passed the canonical reader. **2/3 is a small
smoke result, not a success-rate estimate.** These runs validate the rigid
grasp model, not contact-only cap fidelity or a real-robot transfer rate.

The source managed report says `twin_verified=false`; numerical/Studio smoke
parity does not establish held-out real-world calibration or comprehensive
grasp/contact parity. No hardware, GPU or Gemini requests were used.

The Windows native capture helper was unavailable, so a visual phone/gamepad
review remains outstanding. One real Play client and eight server stations
were exercised; eight networked clients, published asset permissions,
published DataStore behavior and live multiplayer remain owner checks.
