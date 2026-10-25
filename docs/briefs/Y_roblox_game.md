# Agent Y: the Roblox game from Argon's twin, multiplayer, ready to publish (deadline 16:20)
Repo C:\Users\sarta\real2play. Read roblox/README.md, roblox/ACCEPTANCE.md, docs/SPEC.md section 12 (lobby of ~20
tables, sit at a table to play, preset tools, configurable levels), runs/after_step/step_01/ (the Antigravity managed
agent's game build from gemini-4-argon's step-loop sim: config/physics/so101_bench.argon_step_01.json, game/**/result.json
and candidate files), runs/argon_step/step_01/sim.json (Argon's measured bench: cap positions/sizes, arm offsets, gripper
behaviour). The published place is https://www.roblox.com/games/86276076528667/SquarianSquares-Place-10252026-1.
Do:
1. Apply the managed agent's verified candidate files (if its gates passed) and Argon's measured bench values to the game
   presets (tic-tac-toe placement level: board cell size/positions, caps 33 mm x 17 mm, supply lane, arm offsets) so the
   game twin matches the real bench; keep every Luau unit test / parity check passing (roblox/tools/test.py).
2. Make the judge-facing flow great: lobby with 20 tables, multiple players at once (each at their own table, plus
   spectators), a short on-screen goal ("place the cap on the highlighted cell; the robot learns from your play"),
   score/streak per table, a leaderboard, and trace upload to Config.TRACE_URL (tunnel
   https://fairfield-explained-steve-constraint.trycloudflare.com) with the server-only key from ServerStorage/LocalConfig.
3. Build the place file with rojo (roblox/default.project.json or the README's build command) to out/roblox/Real2Play.rbxlx
   and write out/roblox/PUBLISH.md: the exact owner steps (open in Studio, File > Publish to Roblox > existing place
   86276076528667, Game Settings > Security > Allow HTTP Requests, LocalConfig with TRACE_URL/TRACE_KEY from
   out/demo/trace_key.txt - never print the key). If an Open Cloud API key is present in the environment
   (ROBLOX_API_KEY / ROBLOX_UNIVERSE_ID / ROBLOX_PLACE_ID) publish via the Place Publishing API and report the version;
   otherwise do not attempt it.
Commit with explicit paths (roblox/*, out/roblox/PUBLISH.md, tests). Report: what changed, tests, build path, publish status.
