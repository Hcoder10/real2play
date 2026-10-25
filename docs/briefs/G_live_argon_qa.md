# Agent G: live Gemini 4 Argon QA + hardening (production readiness)

Read docs/briefs/COMMON.md, AGENTS.md, docs/INTERFACES.md first. All seven build agents are finished; you now own
FIXES anywhere in r2p/gemini.py, r2p/ttt/, r2p/real2sim/, r2p/labeling.py, r2p/sysid/, r2p/perception/ and their
tests and scripts (commit with explicit paths; keep every existing test passing; run the full suite before each commit:
`.venv/Scripts/python.exe -m pytest -q tests/core tests/argon tests/perception tests/robot tests/app tests/sim tests/policy`).

The REAL bench exists now (Windows laptop): follower arm on COM3 (OWNED by a running arm daemon at
http://127.0.0.1:8793: NEVER open COM3, never POST to armd, never stop it), a calibrated phone scene camera at
http://127.0.0.1:8792/frame/0.jpg (read-only GETs are fine; ~18 fps), config/camera.json (calibrated, 0.70 px),
config/joint_map.json (active vcal_oct25, fitted by Argon visual calibration), tasks/place_on_cell/task.json with the
printed grid's camera-measured cells. Real frames: runs/bench/vcal/p*.jpg (arm in 14 poses over the grid), runs/bench/
flat3.jpg, grid frames runs/bench/grid*.jpg, videos in runs/bench/video/. GEMINI_API_KEY is in the gitignored .env
(auto-loaded; NEVER print it). gemini-4-argon (MAIN, ~4 s/call) and gemini-3.8-flash (SMALL, ~1.4 s) both work.
Budget: at most 80 live requests total; reuse the disk cache; log every call (calls.jsonl).

Goal by 13:30: every Argon-dependent path works on REAL inputs, robustly, with measured latency and accuracy:
1. Tic-tac-toe brain on real frames: r2p.ttt.decide on live phone frames of the real grid (empty board now; also
   synthesize boards by compositing coin/piece crops onto the real frame at cell centres projected with the camera,
   so you can test 10+ board states without touching anything). Measure board-read accuracy, legality, latency;
   harden the prompt (overlay labels, top-down crop) until reads are >= 95 % correct; robust JSON; fallback paths.
   Pieces: human = coins (pennies/nickels, flat, metallic), robot = brown 38 mm cubes or small 25 mm pieces.
2. Real2sim on real data: run the twin loop (scripts/argon/run_twin.py) and task_from_video on a session built from
   the real frames (r2p_trace/1 + frames; the joint log may come from runs/bench/vcal/poses.json); fix crashes, make
   overlays render the real camera model (config/camera.json), keep-rule behaviour, live output for the dashboard.
3. Visual calibration as a product feature: turn scripts/lead/visual_calib.py into r2p/sysid/visual.py (+ CLI
   scripts/argon/visual_calib.py) with plan -> collect (via an injectable mover; the lead runs the real collection) ->
   point (Argon) -> robust fit (soft-L1, leave-one-out) -> report; unit tests with synthetic projections.
4. Every live call path: timeouts, retries with backoff on 5xx/RPM 429, clear QuotaError, cache, served_by in logs.
Report: per path the live numbers (accuracy, latency p50/p90, requests used), bugs fixed, commits, what remains.
