# Agent L: game play -> back through the twin -> training (the data half of the loop), end to end

Read docs/briefs/COMMON.md, AGENTS.md, docs/INTERFACES.md, docs/SPEC.md sections 11-13, roblox/README.md,
roblox/ACCEPTANCE.md, r2p/pipeline/README.md, scripts/robot/H_HARNESS_REPORT.md. Repo C:\Users\sarta\real2play,
`.venv/Scripts/python.exe`. Commit with explicit paths, small commits, pull first (agents I, J, K in parallel). Full
suite before each commit: `.venv/Scripts/python.exe -m pytest -q tests/core tests/argon tests/perception tests/robot
tests/app tests/sim tests/policy`. No hardware, no GPU, no live Gemini (fakes / cache only). Deadline 14:00.

## The product chain (owner's words)
Gemini 4 Argon turns a physical env into an extremely accurate sim, into a game; the game is deployed and people play
it; their data goes back through the accurate sim; a policy is trained alongside Gemini conditioning (Argon gives
high-level commands, the policy does 30 Hz control); it then works seamlessly in the real world.

## Do
1. Prove link "game -> data -> twin -> training" with REAL game traces: the Roblox harness records r2p_trace/1 from
   play (see roblox/ - trace recorder, tables, levels; the authenticated trace receiver on :8770 started by
   scripts/demo_up.py). Produce genuine traces by running the game logic headless if possible (the Luau unit/acceptance
   tooling in roblox/; Studio is not available to you) or by a faithful Python replay of the game's player model;
   label them clearly as headless/simulated players, never as human play.
2. Ingest -> validate (r2p.traces) -> replay each human/bot trace in the MuJoCo twin with the CURRENT physics
   (config/physics) to (a) reject physically impossible ones, (b) relabel actions into the robot's absolute joint-target
   chunks with the commanded-q convention, (c) attach the Argon command segments (pick(piece) / place_at(piece,cell) /
   release / home) via r2p/labeling. Then mix with expert traces and train a small CPU policy; show the before/after
   eval in the twin (more play -> better policy) with honest n.
3. One command: `python -m r2p loop --from-game <dir> --task place_on_cell` (or extend the pipeline) that runs ingest ->
   replay/relabel -> train -> eval -> gate and publishes to the dashboard (dashboard/ adapters: GAME panel counts,
   POLICY panel curve). Update the dashboard labels honestly (headless vs human).
Report: commands, counts (ingested / rejected / relabeled), eval numbers, commits, what the human-play path needs.
