# Real2Play: rules for every agent working in this repo

Real2Play: show Gemini 4 Argon your robot; it experiments until the twin is exact, turns your task into a game the
world plays (Roblox), and trains the robot on that play. Gemini plans, the policy moves.
**Design doc: docs/SPEC.md.** Part I sections 3 (frames), 4 (contracts), 9 (pitfalls), 11-13 (product pipeline, Roblox
harness, visual twin) are binding. The old project it describes is NOT in this repo: write fresh code from the spec.

## Binding design decisions (they fix bugs the old build hit)
1. ONE frame everywhere: spec frame = robot base FOOTPRINT centre on the table surface, metres, x forward, y left,
   z up. `BASE_ORIGIN_MJCF = [0.0168, 0.0, -0.0024]` (footprint in the MJCF base frame): spec = mjcf - BASE_ORIGIN_MJCF.
   Sims, Roblox, perception, policies all use the spec frame. No frame shifts anywhere else.
2. Policies output ABSOLUTE joint targets (15 x 6 chunk, 30 Hz, MJCF radians); the runner also feeds the policy the last
   COMMANDED q (reset to measured when any joint differs > 4 deg). Never predict deltas on measured q.
3. The reference twin models the real arm's gravity sag and servo response from `config/physics/<rig>.json`; the
   real runner applies gravity compensation with the same model. Measured real sag: 9-12 mm at 30 cm reach.
4. Grasps in every sim are physical or rigid-in-gripper-frame (never "near the jaws counts").
5. Every policy export declares task, start_q, obs layout, obs_const (clamp never-varied inputs beyond 5 training stds),
   and its recorded eval. Policies start from their start_q.
6. Gemini roles: MAIN = gemini-4-argon (all reasoning), SMALL = gemini-3.8-flash (per-frame pointing, single-frame
   boxes, JSON repair), via env R2P_MODEL_MAIN / R2P_MODEL_SMALL. Disk cache keyed by sha256(model, config, parts).
   R2P_CACHE_ONLY=1 = zero requests (raise on miss). Daily-quota 429 on MAIN = clear error, NO silent fallback unless
   R2P_ALLOW_FALLBACK=1. Never print or copy the API key (env GEMINI_API_KEY).
7. One success checker (`r2p.tasks.check`) used by sims, the Roblox game (Luau port, shared vectors) and the real arm.

## Working rules
- Own ONLY the paths your task names. Never edit, revert or reformat other agents' files. Commit with explicit paths
  (`git add <paths>`), never `git add -A` / `git add .`, never stash, never rewrite history.
- Python: uv. The shared venv is `.venv` (Python 3.12): run with `.venv/Scripts/python.exe` on Windows. Add a package with
  `uv pip install --python .venv/Scripts/python.exe <pkg>` and list it in pyproject.toml under your component's extra.
- Cross-platform code (the robot bench is a macOS Mac mini): no Windows paths, no msvcrt-only code, pathlib everywhere.
- No real hardware, no real Gemini requests in tests (use the fake client / cache-only). GPU only if your task says so,
  and then only under a landlord lease (ask the lead; do not start GPU work yourself).
- Laptop RAM is tight: at most one heavy process (training, big eval, MuJoCo pools > 4 workers) at a time.
- Tests: `pytest` under `tests/<component>/`. Your task is done only when its acceptance command passes.
- Finish with a short summary: files, how to run, test results, what is NOT done.
