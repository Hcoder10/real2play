# Agent S: a real VLA for tic-tac-toe placement, fine-tuned on the twin, conditioned on Gemini 4 Argon's commands

URGENT: the owner's hard limit for the first full autonomous run (ending with this VLA deployed) is 15:00.
Repo C:\Users\sarta\real2play. Read AGENTS.md, r2p/policy/README.md, r2p/sim/{expert,traces,twin}.py,
r2p/real2sim/evaluate.py (physics_replay renders from the calibrated camera), config/camera.json, tasks/place_on_cell/
task.json (board_pose + cap lane randomization), scripts/robot/K_POLICY_REAL_REPORT.md, docs/briefs/P_policy_zero.md
(agent P is fixing why the state policy scores 0% in the twin - read its commits; the same commanded-q / servo-space
conventions apply to you). Commit with explicit paths, pull first.

## Goal
Fine-tune a pretrained VLA (SmolVLA, `lerobot/smolvla_base`, LeRobot) - not train from scratch - on twin-rendered
episodes of place_on_cell: inputs = scene camera image rendered from the calibrated camera pose + wrist camera image
rendered from the gripper camera mount + joint state; language instruction = Argon's command text exactly as Argon
issues it at runtime (e.g. "pick(piece)" / "place_at(piece,r1c2)" mapped to a natural sentence); output = action chunks
of absolute joint targets at 30 Hz in the same convention the real runner uses. Then evaluate closed-loop in the twin
(success over 30 scenes, clean and real noise) and ship an export the real runner (scripts/robot/run_real_policy.py or
a small adapter) can load, with Argon issuing the commands.

## Resources
- Expert traces (state-only) are on the rig in /dev/shm/r2p/wide_ok (4449 successful, board_pose randomized) and can
  be regenerated (`python -m r2p.sim.traces`, many shards in parallel, see how the lead ran 60 shards). Render images by
  replaying traces in the twin (MuJoCo offscreen; on the rig use MUJOCO_GL=egl) at reduced resolution (e.g. 256x192).
- Rig: ssh -o BatchMode=yes -i ~/.ssh/id_ed25519 cube@100.68.107.11 (Ubuntu, 64 cores, 251 GB RAM, /dev/shm 92 GB
  free, / has 5 GB free: keep datasets in /dev/shm). NO internet/DNS on the rig: download wheels/weights on the laptop
  and scp them (laptop HF cache may be on D:; the laptop env C:\Users\sarta\reachy-duplex\.venv-so101 has lerobot).
  Existing venv ~/venvs/diag (torch 2.11 cu128, numpy) + extra packages via `uv pip install --target ~/r2p_deps` of
  scp'd wheels (see how the lead did it). GPU: a landlord lease is REQUIRED - the lead holds apt_e05759e0 on
  squaredcube1-ubuntu GPU 0 (60%, 16 GB, until 14:51; use env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
  LANDLORD_LEASE_ID=apt_e05759e0 and torch.cuda.set_per_process_memory_fraction(0.167) / 16 GB). If you need more or
  longer, write runs/S_lease_request.json {gpus, pct, vram_gb, minutes, why} and the lead will rent it; do not rent
  yourself.
- Known traps (owner's notes): SmolVLA micro-batch throughput peaks at 64; smolvla_base needs real train_stats
  normalization (identity normalizer otherwise) - compute stats from your dataset; LeRobot camera names must match the
  dataset features exactly.
Do it in the fastest way that can work by 14:50: small dataset first (e.g. 400 episodes x 2 cams), short fine-tune,
twin eval, then scale if time remains. Report: dataset size, training time, twin success numbers, the export path and
the exact deploy command; commits.
