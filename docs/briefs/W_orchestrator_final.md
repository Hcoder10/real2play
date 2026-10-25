# Agent W: Argon orchestrates the WHOLE pass, including training and deployment (deadline 16:25)
Repo C:\Users\sarta\real2play, `.venv/Scripts/python.exe`. Read r2p/orchestrator.py + scripts/argon/U_CHAIN_AFTER_LAB_REPORT.md
(current chain), scripts/lead/argon_step.py (the WORKING real experiment->sim loop: one move per turn, both cameras, low
thinking; output runs/argon_step/<run>/{sim.json,ledger.jsonl}), scripts/lead/after_step.py (Argon sim -> physics
config -> managed-agent game build -> play data), scripts/robot/run_real_policy.py (--argon-perception, Argon decides the
cell), scripts/lead/ttt_armd.py (Argon + calibrated IK placement with Argon's measured offset; proven), scripts/lead/lab_auto.py
(Argon client hardening: stateless, plain-language preface, retries). Commit with explicit paths, pull first. Do not edit
scripts/lead/*, r2p/lab/*, dashboard/*.
Make `python -m r2p argon --task place_on_cell --step-run <run> --approve owner-authorized` (owner authorized
autonomous operation) where gemini-4-argon (thinking low/medium, stateless hardened client) decides each stage as a tool:
1. experiment_sim: run scripts/lead/argon_step.py (or accept an existing --step-run) -> sim.json + measured offsets.
2. build_game: after_step's physics conversion + scripts/argon/build_game.py (Antigravity managed agent; provisional twin
   label unless a gate passes).
3. collect_play: real receiver traces (roblox/out/traces, source human) + headless players. RUN THE HEADLESS PLAY AND
   ALL HEAVY CPU/GPU WORK ON THE RENTED B200, not the laptop (laptop OOMs): ssh -o BatchMode=yes -p 32926 -i
   ~/.ssh/id_ed25519 root@ssh9.vast.ai, repo at /root/real2play (sync with `git bundle` from this checkout), python3
   (torch 2.8 cu128, mujoco 3.13), 192 cores, 2 TB RAM. GPU lease env to prefix every GPU command:
   CUDA_VISIBLE_DEVICES=0 LANDLORD_LEASE_ID=apt_8fc32d1f LANDLORD_BUILDING=vast-57730383 LANDLORD_VRAM_GB=170.0
   LANDLORD_MAX_PCT=100 (torch memory fraction 0.949). Copy results back as they are produced (the box is destroyed at
   17:30).
4. train: P2's recipe (r2p/policy/P2_POLICY_STALLS_REPORT.md: 60k steps, batch 256, seeds 0 1 2, width 512, blocks 3,
   --no-augmentation, --eval-n 30) on expert traces (+ replayed play traces) on the B200; use the bench-profile expert
   traces if agent P3 has committed a fix (r2p/sim/expert.py; check git log), else the default expert and say so; pull the
   export back to exports/<name>.
5. evaluate: twin evals + scripts/robot/rehearse_policy.py --export <name> (real joint map, daemon limits; no motion).
6. deploy: if the rehearsal reaches goals, ONE real turn with scripts/robot/run_real_policy.py --export <name>
   --argon-perception --transport armd --bench-profile scripts/robot/K_STREAM_PROFILE.json --cap <i> --real
   --allow-unverified (Argon reads the board and picks the cell); otherwise ONE real turn with scripts/lead/ttt_armd.py
   (Argon + calibrated IK). The ledger states which path ran and why.
Prove it with a dry run (fake Argon, fake ssh, MockRobot) < 3 min, then report the exact live command. Do NOT run the
live command yourself (the lead runs it with the owner watching).
