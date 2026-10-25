# Agent T2: wire the demo app (http://127.0.0.1:8801/demo, built by agent T: dashboard/) to the REAL evidence. Deadline 16:05.
Repo C:\Users\sarta\real2play. Read dashboard/README.md and agent T's commits (93530ec, 7007324). Only edit dashboard/*
(+ its tests). Restart only the 8801 demo process you own (see README); leave port 8800 alone.
1. Argon Lab panel: the real experiment run is the step loop: runs/argon_step/<newest>/ (ledger.jsonl lines: {turn, obs,
   reply{thought, action, sim}}; images t<NN>_real_vs_sim.jpg (left real scene camera, right sim render) and
   t<NN>_wrist.jpg; sim.json = Argon's sim). Show: the latest thought (large), the last ~6 actions as a list (move
   [x,y,z] / gripper v / done), the latest real|sim image and wrist image, a slideshow over all turns (auto-advance 1 s,
   loop) and Argon's sim summary (objects count, arm offsets z/xy mm, notes). Prefer argon_step over runs/lab when newer.
2. Chain strip: derive from real artifacts, not the synthetic orchestrator rehearsal: Real = argon_step turns + 'real
   pick-and-place into r1c1'; Twin = sim.json objects/offsets; Game = runs/after_step/<run>/game/**/result.json passed;
   Play = roblox/out/traces counts by source + runs/after_step/<run>/loop; Replay/Train = exports/wide_v3/config.json
   evals (clean 0.967 / realx0.5 0.80 / real 0.433, n=30) labelled "twin"; Deploy = out/ttt/status.json. Label each
   state honestly (done / running / pending), no "synthetic" unless it is.
3. Replay & train panel: show wide_v3's real numbers (clean / half noise / full noise) and say "state policy, twin eval;
   real-arm deployment pending (bench-profile expert)".
4. Screenshot /demo again with dashboard/shots.py; zero JS errors. Commit explicit paths. Report the screenshot path.
