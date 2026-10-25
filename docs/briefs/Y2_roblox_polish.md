# Agent Y2: Roblox polish + recorded gameplay videos (deadline 16:25, hard)
Repo C:\Users\sarta\real2play. Start from agent Y's work (commit de74f14, roblox/AGENT_Y_ACCEPTANCE.md, out/roblox/PUBLISH.md,
out/roblox/Real2Play.rbxlx). Owner's requests:
1. The game floor a distinct colour (clearly different from the tables/paper/table-top; e.g. a deep slate/teal floor).
2. No UI overlap anywhere (HUD, goal text, score/streak, leaderboard, per-table panels) at 1920x1080 and 1366x768.
3. ProximityPrompts never overlap: "Sit"/"Play" and "View/Spectate" at each table use distinct anchors/offsets; when a
   player is seated (playing) they get NO view/spectate prompt; when someone else sits, others only get "View".
4. Make the in-game robot/bench match the real bench: black SO-101 arm (not yellow), white bottle caps (33 mm x 17 mm
   scaled per the game's studs convention), printed white paper with the black 3x3 grid, the two brown weights, the
   measured layout from tasks/place_on_cell/task.json and runs/argon_step/step_01/sim.json (Argon's measured bench), and
   the blue bowl + two brown boxes level from tasks/bowl_bench/task.json.
5. Keep all tests passing (roblox/tools/test.py, Luau suites, parity).
6. Record videos of it working: run the place in Roblox Studio Play (use the repo's existing Studio automation in
   roblox/tools / StudioTestService harness if available; Studio is installed on this laptop) and screen-capture 15-25 s
   clips with ffmpeg gdigrab (C:/Users/sarta/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1-full_build/bin/ffmpeg.exe):
   (a) lobby with tables, (b) a player sitting and placing a cap (bot/headless if needed), (c) the bowl level. Save to
   out/demo/roblox_*.mp4 (H.264). If Studio automation is impossible, render the same scenes another way and say so.
Rebuild out/roblox/Real2Play.rbxlx. Commit with explicit paths. Report the video paths and what the owner must do to
publish. Do not touch the arm, r2p/policy, r2p/lab, scripts/lead.
