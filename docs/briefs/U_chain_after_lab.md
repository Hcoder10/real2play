# Agent U: the autonomous chain after Argon Lab, ready to fire the moment the Lab finishes (deadline 15:10)

Repo C:\Users\sarta\real2play, `.venv/Scripts/python.exe`. Read r2p/orchestrator.py, scripts/argon/N_ORCHESTRATOR_REPORT.md,
r2p/lab/{simws,agent}.py (Lab run layout: runs/lab/<run>/sim/{scene.xml,physics.json}, heldout.json with
result.improved/sealed, episodes/), scripts/lead/lab_to_game.py (Lab sim -> verified game build via the Antigravity
managed agent), scripts/lead/lab_auto.py (how the lead runs Argon robustly: stateless gemini.Gemini client, a
plain-language preface, retries on filter blocks / malformed JSON, H.264 >= 2 s video parts, tolerant first-object
JSON, turn errors fed back), r2p/pipeline/L_GAME_DATA_REPORT.md (`python -m r2p loop --from-game`),
scripts/robot/K_POLICY_REAL_REPORT.md (guarded live policy runner), scripts/lead/ttt_armd.py (Argon tic-tac-toe with
calibrated IK; proven). Commit with explicit paths, pull first. Do NOT edit r2p/lab/*, scripts/lead/lab_auto.py,
scripts/lead/jaw_move.py (a live Lab run uses them), r2p/policy/* (agent P2), r2p/vla/* (agent S), dashboard/* (agent T).
NEVER move the arm or start GPU jobs yourself.

## Make `python -m r2p argon --task place_on_cell --lab-run <run> --approve owner-authorized` work end to end:
1. Twin stage = the Lab: twin_gate reads runs/lab/<run>/heldout.json (+ residuals) instead of running the old active
   twin; pass = sealed and improved (record the numbers); frozen hashes = the Lab's sim/physics.json + scene.xml.
2. build_game = scripts/lead/lab_to_game.py logic (verified when the Lab gate passed, else a clearly labelled
   provisional build).
3. collect_play = real traces from the receiver (roblox/out/traces, source human) + headless simulated players run in
   the Lab's physics (L's path), counted separately and labelled.
4. replay_and_train = L's replay/relabel through the twin (Lab physics) + expert traces; heavy training is the rig:
   write runs/argon_orchestrator/<run>/train_request.json {traces dirs, physics path, recipe: the policy trainer
   command (or P2's newest recipe if committed), name} and wait for a receipt export (the lead launches it under the
   GPU lease); in --dry-run use the CPU smoke.
5. deploy_real = one real tic-tac-toe turn: Argon reads the board and issues the command; the arm executes via the
   learned policy if its twin eval passes the gate, else via scripts/lead/ttt_armd.py place() (calibrated IK) - the
   ledger must say which.
6. Approval: add `--approve owner-authorized` (the owner authorized autonomous operation for this session): no
   prompts, every risky action logged with that authorization; keep `ask` as default.
7. Robustness: use the same Argon client hardening as scripts/lead/lab_auto.py (stateless client by default; the
   chained previous_interaction_id made every second Lab turn fail with malformed_tool_call).
8. Prove it: a dry run with a FAKE Lab run directory you create (copy runs/lab/<any>/sim + a synthetic heldout.json
   improved=true) completes all stages in < 3 min; then `--plan-only` one live Argon decision for the real bench.
Report: the exact live command, the dry transcript excerpt, commits.
