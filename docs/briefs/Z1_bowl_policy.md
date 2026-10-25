# Agent Z1: a real-arm-deployable policy for two brown cubes -> blue bowl, trained FAST on the B200 (deadline 16:20)
Repo C:\Users\sarta\real2play. B200 box: ssh -o BatchMode=yes -p 32926 -i ~/.ssh/id_ed25519 root@ssh9.vast.ai, repo
/root/real2play (sync from this checkout with `git bundle create ... main` + scp + `git pull`), python3 (torch 2.8
cu128, mujoco 3.13), 192 cores, 2 TB RAM. GPU lease env for EVERY GPU command: CUDA_VISIBLE_DEVICES=0
LANDLORD_LEASE_ID=apt_8fc32d1f LANDLORD_BUILDING=vast-57730383 LANDLORD_VRAM_GB=170.0 LANDLORD_MAX_PCT=100. Copy results
back to this laptop (exports/<name>) as soon as they exist (the box dies at 17:30). Do not touch scripts/lead, r2p/lab,
dashboard, roblox. Agent Z2 writes tasks/bowl_bench/task.json (measured positions) in parallel: pull before using it;
until it exists use tasks/two_cubes_in_bowl/task.json with both cubes brown.
Facts: the default expert solves cube_in_bowl ~95% in sim, but the K bench profile (scripts/robot/K_BENCH_PROFILE.json:
real start_q, real model_limits with the daemon's 2 deg margin, LOCKED wrist roll, profile gripper) drops place_on_cell
to ~2% (grasp_miss / dropped_during_carry), and gripper close/open/travel variants do NOT fix it (probed: 2-4/192) - so
the locked roll and/or limits are the cause; agent P3 may already have a fix (git log r2p/sim). Real roll is now
calibrated (vcal_oct25b), so the real arm can use roll FREE within +-85 deg of the rest roll (model ~-1.44 rad) while
staying 5.7 deg inside the daemon limits.
Do, fastest path first:
1. A deployable expert: a profile variant with roll free within that window (not locked), the real start_q, real limits,
   travel height ~5 cm; get the expert >= 70% on the bowl task under it (probe 192 seeds on the B200 in < 1 min).
2. Traces: thousands on the B200 (190 parallel processes), keep successes.
3. Train: P2 recipe (r2p/policy/P2_POLICY_STALLS_REPORT.md: --steps 60000 --batch-size 256 --seeds 0 1 2 --width 512
   --blocks 3 --no-augmentation --eval-n 30), device cuda:0 on the B200 (scale batch/steps if it saturates fast).
4. Evals in the twin + scripts/robot/rehearse_policy.py with the real joint map/limits (adapt it for the bowl task if it
   is place_on_cell-specific) - report success and whether starts are accepted.
5. Pull the export to exports/bowl_v1 and report the exact real deploy command (scripts/robot/run_real_policy.py or a
   bowl variant; colour perception works for brown cubes / blue bowl).
