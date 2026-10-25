# Agent P: the trained place_on_cell policy scores 0% in the twin; find why and fix it (urgent, 30 min)

Repo C:\Users\sarta\real2play, `.venv/Scripts/python.exe`. Read r2p/policy/README.md, r2p/policy/{data,features,
network,runtime,train,eval}.py, r2p/sim/{expert,traces,twin}.py, scripts/robot/K_POLICY_REAL_REPORT.md,
r2p/pipeline/L_GAME_DATA_REPORT.md. Commit with explicit paths, pull first. Do NOT edit r2p/real2sim/active.py or
scripts/lead/* (a real-arm campaign is running). No hardware, no live Gemini. CPU only on this laptop (RAM is tight:
one heavy process at a time).

Facts:
- exports/wide_v1 (copied from the rig): 3-seed MLP ensemble, 20000 steps, batch 256, 4449 successful expert traces of
  place_on_cell with board_pose randomization (task.json now). Training loss 0.44 -> 0.005. Twin eval clean / real /
  realx0.5: 0/30 each, ALL timeouts at 1200 ticks, ~1050-1080 commanded-q resets per episode (the >4 deg
  measured-vs-commanded reset), commands_completed 0-1. K's CPU smoke export: 0/27; L's tiny CPU policies: 0/12.
- A sample expert trace: runs/train/expert_place_on_cell_00200000.jsonl (meta.target_space "servo").
- So the policy's closed-loop inputs or outputs do not match training: e.g. q_target in servo space (gravity-comp
  offset) vs measured model q; the commanded-q feedback convention; feature/slot encoding (board_pose moves cells but
  slots/command vocabulary may not); normalization/obs_const clamps; chunk indexing; start_q; units.
Do: reproduce with `python -m r2p.policy.eval --export wide_v1 --n 5 --noise clean` (find the right flags), then
open-loop check: feed the expert trace's own observations to the policy and compare predicted vs labelled chunks; then
closed-loop with the expert's exact scene. Find the mismatch, fix it in training/eval/runtime consistently (the same
runtime is used on the real arm), add a regression test (policy trained on 20 traces must replay its own expert
trace open-loop within tolerance and complete >0 closed-loop episodes on the training scenes), and write the exact
retrain command for the rig. Report: the root cause in one paragraph, the fix, numbers before/after, commits.
