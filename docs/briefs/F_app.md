# Agent F: orchestration CLI + dashboard

Read docs/briefs/COMMON.md first.
Own: r2p/cli.py, r2p/__main__.py, r2p/pipeline/, dashboard/, tests/app/. Code against docs/INTERFACES.md; other
modules land over the day: import lazily and make every stage report BLOCKED with a clear message when its dependency
is missing.

Deliver:
1. `python -m r2p run --task <name|task.json|"instruction"> [--video <session>] [--rounds N] [--dry-run | --real]
   [--main-model --small-model --allow-fallback --no-cache-only] [--force rN:<stage> --no-cascade]
   [--device cuda|mps|cpu]`; `python -m r2p status`. Stages per round: session (probe attempt) -> twin (D's real2sim)
   -> sysid/close (twin vs reality -> physics/focus scenes) -> data (B's expert traces + Roblox traces for the task
   from roblox/out/traces) -> label (D's labeling) -> train (B; refuses without runs/lease.json from a landlord lease;
   prints the exact lease step) -> eval -> gate (challenger vs incumbent on IDENTICAL scenes + noise; the line names
   incumbent and scenes) -> deploy (C1's run_policy; plan mode prints the exact bench command; --dry-run runs it on C2's
   synthbench). Idempotent, cached under runs/<task>/<round>/, manifest runs/<task>/manifest.json (r2p_run/1) written
   atomically. Dry runs cache-only by default (cache miss / daily 429 = BLOCKED).
2. dashboard/server.py (stdlib http + one static page, port 8800, --mock): header "Real2Play" + four panels:
   REAL (camera frames + status from C1's publisher http://127.0.0.1:8791/frame.jpg and status.json incl. argon_says;
   only real attempts unless --show-dry-runs), REAL2SIM (r2s_live/1 run pinned in config: gap = 1 - mean object IoU
   with best-so-far chart, motion-match px chart, image rotation SIM / REAL / MOTION with the badge synced to the
   cross-fade, SYNTHETIC badge for synthetic sessions, best twin when done), GAME (QR to game_url, rounds / successes /
   hours from traces by meta, per-level strip incl. tic-tac-toe, bots separate, leaderboard), POLICY (pinned eval: clean
   + real-noise numbers, scaling curve from runs/scaling.json); plus /runs (manifests, SIM DRY RUN labels) and a
   TIC-TAC-TOE live board widget (board, Argon's "why", the model that answered). Honest labels everywhere; never show
   sim as real. 1920x1080 dark theme, readable across a room.
3. dashboard/shots.py: real-time screenshots over Chrome DevTools (not headless virtual time).
4. tests/app: manifest atomicity, stage caching/resume, gate logic, BLOCKED paths, dashboard adapters on fixture
   folders.

Acceptance: `pytest -q tests/app` passes; `python dashboard/server.py --mock` serves all panels (take a screenshot with
shots.py and report the path).
