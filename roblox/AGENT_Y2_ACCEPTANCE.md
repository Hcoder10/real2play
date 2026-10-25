# Y2 Roblox polish acceptance — 2026-10-25

Rebuilt `out/roblox/Real2Play.rbxlx` after the actual Studio parity gate passed.
Publish instructions: `out/roblox/PUBLISH.md`. No cloud publish was performed.

## Changes

- Deep teal floor and warm table surfaces distinguish the white printed paper.
- Black SO-101 meshes, measured 33 mm diameter / 17 mm tall caps, printed 3×3
  grid, and brown bench weights at Argon's measured coordinates. All dimensions
  retain the existing 10 studs/metre conversion and shared robot frame.
- Idle tables show cap tasks and the measured blue bowl / two brown boxes task;
  the attract table runs `bowl_bench`.
- Responsive HUD reserves separate space for navigation, goal, timer and score.
  Menus are mutually exclusive and the tutorial no longer covers the score.
- Sit and View have separate attachment anchors and screen offsets. Only the
  nearest table exposes prompts: an available table offers Sit, an occupied
  table offers View, and seated or spectating players see neither prompt.
- Studio-only demo and audit tools are separate from the production project.
  Demo traces are explicitly bot/test-only and uploads are disabled for filming.

## Verification

- `.venv/Scripts/python.exe roblox/tools/test.py`: all six Luau suites passed
  (1,311 bench, 576 dynamics, 2,020 levels, 3,143 checker, 23,165 tic-tac-toe,
  27 touch assertions).
- `.venv/Scripts/python.exe -m pytest tests/roblox -q`: **9 passed**.
- Pure Luau parity: maximum joint error 0.008664 degrees, jaw error 0.072527 mm.
- Full Play smoke passed seating, scoring, complete tic-tac-toe, standing,
  spectator flow and four authenticated bot/test-only trace uploads.
  Report: `roblox/out/smoke/result.json`.
- `node roblox/tools/runmode.mjs roblox/tools/polish_smoke.luau '{}' 90
  roblox/out/y2_scene_smoke.json`: **429 scene assertions passed** across 20 tables.
- Actual Play UI audit: **59 samples, zero visible text/button intersections**,
  including exact 1366×768 and 1920×1080 simulator resolutions, menus, scores,
  seated play and walking. Prompt fixtures verified Sit / View / none states.
  Evidence: `roblox/out/y2_ui_audit.json` and `y2_ui_samples.json`.
- Final `.venv/Scripts/python.exe roblox/tools/build.py`: passed generated-preset
  freshness and 180 actual Studio physics ticks. Errors: joint RMS
  0.146813 degrees, jaw RMS 1.450329 mm, cube endpoint 1.311680 mm, cube rotation
  0.025430 degrees; all below the 1 degree / 2 mm / 5 mm / 5 degree limits.
  Evidence: `roblox/out/publish_parity.json`.

## Videos

Actual Roblox Studio Play captured with ffmpeg gdigrab, CPU H.264, real-time.
Studio chrome was cropped out; final clips are 1920×1080. No alternate renderer
or fabricated gameplay was used. A real Play-session avatar sits at the table;
the existing bot controller moves the arm through the normal simulation.

| File | Duration | Content |
| --- | ---: | --- |
| `out/demo/roblox_lobby.mp4` | 20 s | Twenty tables, teal floor, bench displays |
| `out/demo/roblox_cap.mp4` | 25 s | Seated avatar, physical cap placement |
| `out/demo/roblox_bowl.mp4` | 20 s | Blue bowl and two brown cubes, physical pickup |

The cap runs passed the shared success checker after 22.17 and 15.20 simulated
seconds. The final cap clip includes the success panel, then the bowl transition.
Raw window recordings remain as `roblox_*_studio.mp4`; file hashes and format
metadata are in `roblox/out/y2_videos.json`.

To reproduce, open the built place in Studio, run
`node roblox/tools/install_demo.mjs`, then start **Play (F5)**. Set `FFMPEG` to
the ffmpeg executable and `STUDIO_WINDOW_TITLE` to the exact Studio window
title, then run `node roblox/tools/record_demo.mjs`. The recorder waits for the
demo stages. Stop Play and run `node roblox/tools/install_demo.mjs --remove`
afterward. `finalize_demo.py --ffmpeg <path>` crops this laptop's inspected
capture layout; inspect and update the crop if the Studio layout changes.

## Limits and remaining owner checks

- The brown weights are fixed, non-colliding visual bench props. Their collision
  model has not been newly calibrated; existing arm/task dynamics are unchanged.
- Occupied-table prompt behavior was tested with a local occupancy fixture,
  not two independent network clients. Follow the published two-client check.
- Videos use bot control, not human demonstration data. No real-robot transfer,
  hardware run, GPU work, or Gemini requests were performed.
- The bowl clip demonstrates pickup, not completion of the whole bowl task.
- Publish to existing place **86276076528667**, verify asset permissions and
  HTTP requests, and confirm the current trace receiver URL before release.
  The private server configuration and generated videos/place are local artifacts.
