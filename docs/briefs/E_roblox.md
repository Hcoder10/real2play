# Agent E: Roblox game harness

Read docs/briefs/COMMON.md first.
Own: roblox/ (Rojo project), tests/roblox/. Build SPEC section 12 (harness) + 6.4 (what worked before).
Tools: rojo 7.6.1 (`rojo build`); pure Luau runner C:\Users\sarta\g4studio\backend\bin\luau.exe for unit tests (never
lune). Roblox Studio is open with a studio-bridge plugin server on http://localhost:3000; see
C:\Users\sarta\argon-hack\roblox\tools\bridge.mjs and run.mjs for how to send Luau to Studio and run Play mode (that
old folder is reference for the PROTOCOL ONLY). Exactly one Studio window; do not close it. Mesh assets: the 13 SO-101
meshes are already uploaded; their asset ids are in C:\Users\sarta\argon-hack\roblox\meshes\SO101Meshes.model.json
(copy that file; it is asset data).

Deliver:
1. Physics: JointSim (MuJoCo-style servo model per joint integrated at 480 Hz in Luau: mass matrix + bias via a compact
   rigid-body algorithm on the official SO-101 kinematics, armature, kp/kv, torque limits) driving HingeConstraint
   motors each frame; collision voxel boxes (Transparency 1) + visual MeshParts; ALL twin parts server-owned.
   Parameters from a physics preset generated from config/physics/<rig>.json (agent B owns that file; read it with
   roblox/tools/gen_presets.py) so the Roblox arm sags like the real arm.
2. Harness configs (generated Luau modules from JSON): robot preset, task presets from tasks/*/task.json (agent A),
   game presets (mode solo / vs bot / timed / daily / tic-tac-toe, scoring, stars, round length, tutorial steps, juice
   set, leaderboard keys, unlocks), physics preset. Parity: the same command sequence in Roblox Run mode vs r2p.sim
   (B) -> joint rms, jaw position, object end pose; roblox/tools/parity.py reports pass/fail.
3. World: a lobby with 20 tables; walk up and SIT at a table (Seat or ProximityPrompt) to start; camera locks to an
   over-the-shoulder table view; standing up frees the table. Each table = arm twin + task instance; 8+ concurrent
   players.
4. Controls (phone first): joystick MOVE, UP/DOWN, TWIST, GRAB/DROP, keyboard + gamepad; IK end-effector control;
   depth cues (goal glow, gripper shadow dot); tutorial with a ghost hand.
5. Game modes: Tic-Tac-Toe vs Robot (player places the piece on a free square: centre within 18 mm, at rest, jaws open,
   0.5 s; robot answers with minimax + 25 % slip, never skips a win; win 300 / draw 100), plus every task level (cube
   in bowl, stick, stack, push, two cubes), daily challenge, stars, combo, streak, leaderboards (OrderedDataStore,
   guarded for Studio), unlocks, juice (sounds, particles, camera shake, haptics), "N rounds today, you've sent M to the
   robot" counter, REAL ARM card (animated placeholder; /realarm sets an image).
6. Success: TaskCheck.lua = exact Luau port of r2p.tasks.check / check_window; tests/roblox/taskcheck_test.luau
   replays r2p/tasks/vectors.json (agent A) and must pass all cases.
7. Data: every round logged at 30 Hz in r2p_trace/1 (A's r2p/traces.py; meta.source human/bot, task, level,
   fail_reason, commands from the game's own phase events), chunked authenticated upload (<= 600 ticks per part,
   retries) to roblox/tools/trace_server.py (POST /trace with key, UTF-8 safe, writes roblox/out/traces/; /health);
   TRACE_URL/TRACE_KEY in a server-only config; owner chat command /traceurl. Bots: a SkillOperator plays every task
   headless (data when nobody is online); attract mode behind the menu is never recorded.
8. Tests: Luau unit tests (task-check vectors, touch mapping, levels/scoring, tic-tac-toe rules incl. exhaustive game
   tree), trace server test (python), rojo build succeeds, a Play-mode smoke test via the bridge (sit at a table,
   finish a tic-tac-toe round, a trace arrives).

Acceptance: `rojo build -o roblox/real2play.rbxl` succeeds; all Luau tests pass with the luau runner; the trace server
test passes; Play smoke passes (report if Studio is unavailable). Write the owner's publish checklist into
roblox/README.md.
