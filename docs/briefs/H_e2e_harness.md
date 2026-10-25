# Agent H: end-to-end harness integration + production hardening

Read docs/briefs/COMMON.md, AGENTS.md, docs/INTERFACES.md first. All seven build agents are finished; you now own
FIXES in r2p/pipeline/, r2p/cli.py, r2p/robot/ (software only), r2p/sim/, r2p/policy/, r2p/traces.py, dashboard/,
scripts/robot/, scripts/argon/tictactoe.py and their tests (commit with explicit paths; keep every existing test
passing; full suite before each commit: `.venv/Scripts/python.exe -m pytest -q tests/core tests/argon tests/perception
tests/robot tests/app tests/sim tests/policy`). Another agent (G) is fixing Gemini/ttt/real2sim/perception in parallel:
coordinate through git (pull before editing shared files; small commits).

The REAL bench exists (Windows laptop): follower on COM3 is OWNED by a running arm daemon (http://127.0.0.1:8793):
NEVER open COM3, never POST to armd, never stop it. Phone scene camera read-only at http://127.0.0.1:8792/frame/0.jpg.
config/bench.json + config/bench.win32.json (COM3, http camera 960x540, board), config/camera.json, config/joint_map.json
(active vcal_oct25), tasks/place_on_cell/task.json (camera-measured grid, cells ~36 mm, piece supply (0.13, 0.13)).
No GPU: the lead trains under a lease. No live Gemini (cache-only / fake) except via agent G.

Goal by 13:30 (the demo is 17:00, judged 50 % on the live demo):
1. `python -m r2p run --task place_on_cell --dry-run --rounds 2` and `--task cube_in_bowl` complete end to end
   (probe on the synthetic bench, twin, data, label, train on CPU tiny, eval, gate, deploy dry run), resumable;
   fix every BLOCKED/crash; honest manifests; `python -m r2p status`.
2. The LIVE tic-tac-toe path, software side: scripts/argon/tictactoe.py live mode must run on the Windows bench config
   (COM3 via a NEW RealRobot only when the lead stops armd; for your testing use MockRobot + the live phone frames
   read-only) with the fitted joint map, gravity comp, commanded-q, the measured supply spot, small pieces (25 mm);
   a --check mode that prints the plan for each cell (IK, clearance, approach/retreat heights) without moving.
3. Guard/runner tuning: the stub dry run clipped ~70 % of ticks and reset commanded-q ~280 times per attempt; find
   why (step caps vs scripted chunk, sag model, comp), make normal motion clip < 5 % without weakening safety.
4. Dashboard production: live mode reading the real files (camera via the runner's publisher, r2s_live runs,
   roblox/out/traces, exports, runs/), honest labels, the tic-tac-toe widget wired to the ttt status JSON; real-time
   screenshot via dashboard/shots.py of live + mock modes; a single launcher scripts/demo_up.py that starts phonecam
   check, dashboard, trace server and prints URLs.
Report: what runs end to end (commands + timings), fixes, commits, remaining risks.
