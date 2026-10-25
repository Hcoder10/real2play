# Agent P2: wide_v2 stalls (13% clean / 7% real noise, 26/30 timeouts) - make the place_on_cell policy succeed, fast

Continue from your own report r2p/policy/P_POLICY_ZERO_REPORT.md. Since then the lead fixed the root cause you found:
the >4 deg commanded-q reset now covers arm joints only (commit 6d3a64a, in r2p/policy/data.py, eval.py,
r2p/robot/runner.py). wide_v2 = 3-seed MLP, 20000 steps, 4449 successful wide-range traces (board_pose + cap lane
randomization), command_age feature: clean 4/30, real 2/30, failures 26 timeouts. Export copied to exports/wide_v2 when
you start (scp from cube@100.68.107.11:real2play/exports/wide_v2, ssh -i ~/.ssh/id_ed25519 -o BatchMode=yes).
Find where episodes stall (per-phase: approach / descend / close / lift / carry / lower / release / retreat / home; use
your diagnose.py on the exact scenes), fix the cause (data labels at command boundaries, gripper action thresholds,
chunk execution/ensembling, command scheduler progression in eval, success-window settling, training steps/size),
prove it on a 200-trace CPU retrain (> 50% clean on its training scenes), and give the exact rig retrain command. The
rig GPU 0 is leased (apt_e05759e0, env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 LANDLORD_LEASE_ID=apt_e05759e0,
memory fraction 0.167), shared with agent S's VLA training; traces /dev/shm/r2p/wide_ok. You MAY launch ONE rig training
yourself with that env once your CPU proof passes (name wide_v3, PYTHONPATH=$HOME/r2p_deps:., ~/venvs/diag/bin/python,
setsid nohup, sync code via `git bundle` like the lead). Do not touch r2p/lab, scripts/lead, r2p/real2sim. Deadline 14:55.
