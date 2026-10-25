# Agent T: the judges' demo app (extend the dashboard), deadline 15:10

Repo C:\Users\sarta\real2play, `.venv/Scripts/python.exe`. Read dashboard/README.md, dashboard/server.py, adapters.py,
static/, docs/PITCH.md, docs/architecture.html. Commit with explicit paths, pull first. Do NOT edit r2p/lab,
scripts/lead, r2p/policy, r2p/real2sim (other agents and a live real-arm run use them). Read-only access to their outputs.
The dashboard runs at http://127.0.0.1:8800 (started by scripts/demo_up.py); restart only your own dashboard process.

Judging: 50% live demo, 3 minutes, on a projector (1920x1080). The story: Gemini 4 Argon turns a physical bench into an
accurate twin by experimenting on the real arm, builds the game from it, people play, the play data goes back through the
twin, a policy is trained with Argon issuing commands, and it deploys back to the real arm (tic-tac-toe vs the robot).

Build (keep the current panels working; honest labels: real vs simulated vs headless players):
1. ARGON LAB panel (most important): live from runs/lab/<latest run>/ (status.json, ledger.jsonl): Argon's latest
   `# Thought:` text (large, readable), the tool calls of the last turns (real_plan / sim_plan / replay / edit_sim /
   score ...), the latest real | sim side-by-side video (runs/lab/<run>/**/side_by_side*.mp4 or replay videos; serve
   mp4 with range requests so <video> plays/loops), the latest scene/wrist camera frames (runs/lab/<run>/turns/<n>/),
   residual numbers (joint deg, jaw mm, object mm, image similarity) as a small sparkline over turns, and a list of sim
   files Argon edited (with a diff count). Pick the newest run directory automatically.
2. CHAIN timeline across the top: Real -> Twin -> Game -> Play -> Replay -> Train -> Deploy, each with state
   (pending/running/done/blocked) and its key number, from runs/argon_orchestrator/<latest>/status.json + ledger.jsonl
   when present, else derived from the individual artifacts (lab status, runs/game_live_*/result.json, roblox/out/traces
   counts, r2p pipeline status (`python -m r2p status` or its files), exports/*/config.json evals, out/ttt/status.json).
3. GAME panel: the published game link https://www.roblox.com/games/86276076528667/SquarianSquares-Place-10252026-1 with
   a QR code (generate locally, no network), live trace counts by source (human / headless / bot) from the trace
   receiver (GET http://127.0.0.1:8770/health and roblox/out/traces).
4. DEPLOY panel: the live tic-tac-toe board from out/ttt/status.json (board, Argon's move + reason, latency), the phone
   camera (http://127.0.0.1:8792/frame/0.jpg) and wrist camera (http://127.0.0.1:8794/frame/0.jpg) live.
5. A full-screen "demo" route (/demo) laid out for 1920x1080 with large type, dark theme, no scrolling; keyboard 1-4
   to focus a panel. Screenshot it with dashboard/shots.py and check for JS errors.
Report: the URL, a screenshot path, commits.
