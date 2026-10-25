# Agent N: the Gemini 4 Argon orchestrator (one autonomous loop over the whole product)

Read docs/briefs/COMMON.md, AGENTS.md, docs/INTERFACES.md, docs/SPEC.md sections 11-13, docs/PITCH.md,
docs/architecture.html, r2p/pipeline/README.md, scripts/argon/README.md, scripts/argon/I_ACTIVE_TWIN_REPORT.md,
scripts/robot/H_HARNESS_REPORT.md, README.md ("How Gemini 4 Argon is used"). Repo C:\Users\sarta\real2play,
`.venv/Scripts/python.exe`. Commit with explicit paths, small commits, pull first (agents J, K, L and the lead work in
parallel). Full suite before each commit: `.venv/Scripts/python.exe -m pytest -q tests/core tests/argon tests/perception
tests/robot tests/app tests/sim tests/policy`. Deadline 14:15. NEVER move hardware or call live Gemini in tests;
live Argon budget for your own smoke: 15 requests.

## Goal (owner's words)
Gemini 4 Argon, fully autonomous end to end: turn a physical environment into an extremely accurate sim (by
experimenting on the real rig in a loop until the twin matches), into a game; deploy the game and collect play data;
send that data back through the accurate sim; train a policy alongside Gemini conditioning (Argon issues high-level
commands, the policy does 30 Hz control); then run seamlessly in the real world. Problem statement answered: render
what it made, watch the result, critique against the goal (reality), fix in a loop.

## Today every stage exists but a human sequences them. Build the orchestrator.
r2p/orchestrator.py + `python -m r2p argon --task place_on_cell [--dry-run] [--approve auto|ask]`:
- Argon (gemini-4-argon via r2p/gemini.py Interactions API, thinking_level high for planning) is the controller. Give
  it the goal, the current state (status files below), and a closed TOOL set; it returns JSON {tool, args, why,
  done_when}; validate args against each tool's schema; execute; feed back the tool's measured result; loop until the
  stop criterion Argon declares is met AND our numeric gates pass. Keep a ledger (runs/argon_orchestrator/<run>/
  ledger.jsonl + status.json for the dashboard: stage, last tool, why, metrics, next).
- TOOLS (wrap existing code; each returns measured numbers, never prose-only):
  1. measure_scene: grid by camera (scripts/lead/measure_grid.py logic), caps by Argon (scripts/lead/ttt_armd.py
     locate_caps), camera health.
  2. calibrate_arm: r2p/sysid/visual.py (Argon points at jaw tips; open-jaw frames for roll/gripper) -> candidate map +
     held-out px; never auto-activates without the gate.
  3. active_twin: real-arm experiment campaign (scripts/lead/active_live.py: Argon is the cap sensor; agent I's engine)
     or SimMover in dry-run -> held-out error before/after, candidate physics.
  4. twin_gate: held-out real-vs-sim metrics (joint deg, jaw mm, object mm, IoU) vs thresholds -> pass/fail.
  5. build_game: Antigravity managed agent game builder (scripts/argon/build_game.py, agent J) -> Luau tests/parity.
  6. collect_play: game traces from the trace receiver / headless players (agent L's path) -> counts.
  7. replay_and_train: game+expert traces through the twin -> policy training (local CPU tiny in dry-run; the rig GPU
     run is launched by the lead under a landlord lease: expose it as a tool that writes a request file
     runs/argon_orchestrator/<run>/train_request.json and waits for an export to appear) -> twin evals clean/real noise.
  8. deploy_real: one real placement or a tic-tac-toe turn (scripts/lead/ttt_armd.py / agent K's live runner).
- Human-in-control gates (required by the hackathon rules and the owner): any tool that moves the arm, publishes the
  game, or spends money/GPU needs approval: `--approve ask` prompts on the terminal with the exact action; `--approve
  auto` is allowed only for sim/dry-run tools. Log every approval.
- Dry-run mode end to end with fakes (fake Argon decisions, SimMover, MockRobot, headless play) must complete in < 3 min
  and is the test. Then one live Argon planning call (no motion) to show the plan it produces for the current bench.
- Dashboard: add an "ARGON" strip (dashboard/) that shows the orchestrator's ledger (stage, tool, why, metric) live.
Report: the commands, a dry-run transcript excerpt, the live Argon plan, commits.
