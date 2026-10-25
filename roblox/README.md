# Real2Play workshop

Twenty tables, one SO-101 twin per table. Sit at a blue chair, choose a skill or
play tic-tac-toe against the robot, and stand to return to the lobby. Phones use
Move, Up/Down, Twist and Grab/Drop. Keyboard: WASD, Q/E, Z/X, Space, Shift for
fine movement, R to retry. Gamepad: left stick, triggers, shoulder buttons, A,
X for fine movement, Y to retry.

All game authority, success checks, commands and traces run on the server.
The client sends bounded control axes; it cannot supply scores, poses, targets,
success flags or trace credentials. Studio DataStores are disabled by default.

## Run and verify

Run these from the repository root. Python uses the shared Python 3.12 venv.
On macOS/Linux, substitute `.venv/bin/python` and set `LUAU_BIN` if necessary.
The tests use the **pure Luau runner**, never lune.

```powershell
.venv/Scripts/python.exe roblox/tools/gen_presets.py --robot
.venv/Scripts/python.exe roblox/tools/test.py
.venv/Scripts/python.exe -m pytest tests/roblox -q
rojo build roblox/default.project.json -o roblox/real2play.rbxl
```

Equivalent Rojo command from `roblox/`: `rojo build -o real2play.rbxl`.
`LUAU_BIN` selects a runner; otherwise the test script checks PATH and the
existing `~/g4studio/backend/bin/luau.exe` installation.

The existing Studio window can be updated without closing it:

```powershell
node roblox/tools/sync.mjs
.venv/Scripts/python.exe roblox/tools/parity.py
.venv/Scripts/python.exe roblox/tools/parity.py --studio --out roblox/out/parity_runtime.json
.venv/Scripts/python.exe roblox/tools/smoke.py
node roblox/tools/runmode.mjs tests/roblox/concurrency_smoke.luau '{}' 120 roblox/out/concurrency.json
node roblox/tools/runmode.mjs tests/roblox/scene_smoke.luau '{}' 120 roblox/out/scene_smoke.json
```

Studio's bridge must be listening on `http://localhost:3000`; override with
`STUDIO_BRIDGE_URL`. The clients implement the bridge's MCP transport directly,
so npm packages are not needed. `run.mjs` executes an edit-mode Luau file;
`runmode.mjs` uses StudioTestService for a real Run session. Set `PLAY=1` for a
real Player/client. For arguments on Windows, use `@path/to/args.json` to avoid
shell quotation changes. Tests remove only their own temporary test scripts.

The Play smoke sits a real Player, checks client controls/camera, completes a
tic-tac-toe match through settled placement fixtures, verifies server ownership,
stands, and validates every received trace with `r2p.traces.read_trace`. Fixture
traces carry `source: bot`, `test_only: true`; they are not human demonstrations
or proof of manipulation quality. They live under `roblox/out/smoke/`.

## Configuration and physics

`tools/gen_presets.py` imports the official MJCF, the canonical task JSON files,
and `config/physics/so101_bench.json`. It generates `Twin/Generated/{Robot,Tasks,
Physics,Game}.lua` and the shared task-check vectors. Edit the source JSON, then
regenerate. `--check` rejects stale modules. The game JSON is
`presets/game/workshop.json`: modes, scoring, stars, tutorials, sounds, unlocks,
table count and leaderboard keys are all there.

JointSim uses a fresh recursive Newton–Euler implementation of the official
SO-101 kinematics/inertias. It solves the full six-joint mass matrix plus
gravity/Coriolis bias, rotor armature, damping, friction and torque-limited
position servos at **480 Hz**. Targets are absolute MJCF radians at 30 Hz, with
the preset's latency. Finite stiffness produces sag; the config's `sag_gain`
describes real-arm compensation and is not added a second time to the twin.

Roblox HingeConstraint actuators follow JointSim in servo mode. Collision-free
mass proxies condition Roblox's articulated solver; they do not replace the
identified dynamics. Contact geometry is generated from 4 mm mesh voxels and
merged into transparent boxes. The thirteen supplied MeshParts provide the
appearance. Grasping is physical finger/object contact: no proximity attachment,
object teleport or success shortcut exists in gameplay. The server owns all
unanchored twin assemblies. Idle arms sleep with anchored roots.
Active airborne objects are kept awake through an unused touch-event property,
following Roblox's [documented sleep behavior](https://create.roblox.com/docs/physics/sleep-system).
This changes no pose, velocity, collision shape or force; it prevents a slowly
held object from remaining suspended after its contact support moves.

There is exactly one frame conversion: MJCF coordinates subtract
`[0.0168, 0, -0.0024]` at the robot root. Public positions are footprint-frame
metres, x forward/y left/z up. Roblox maps `(x,y,z)` to `(10x,10z,-10y)` plus
the table origin. Task slots use the canonical object order and K×9 layout.

The pure dynamics oracle checks FK, mass and bias against MuJoCo. Parity then
replays identical absolute commands and scene JSON through `r2p.sim.TaskTwin`
and either pure Luau or actual Roblox Run-mode constraints. Reports include
joint RMS, jaw RMS and every object's end-position error, with explicit
pass/fail and a nonzero exit on failure. Default limits: 1° joint RMS / 2 mm
jaw RMS / 5 mm object position / 5° object orientation. Quaternion comparison
handles the equivalent opposite signs.
The default held-out excitation and object drop are a smoke gate, not a
hardware calibration or a comprehensive contact/grasp equivalence claim.
Supply `--input file.json` with `{scene, q0, commands}` for held-out push/grasp
experiments. Keep the resulting report, input and reference/candidate artifacts.

## Traces, bot collection and Argon integration

```powershell
$env:TRACE_KEY = '<set a private key locally>'
.venv/Scripts/python.exe roblox/tools/trace_server.py --port 8770
.venv/Scripts/python.exe roblox/tools/collect.py --rounds 2
```

The receiver writes canonical `r2p_trace/1` JSONL files to `roblox/out/traces/`.
POST `/trace` uses `X-Trace-Key` and JSON
`{trace_id, part, final, meta?, ticks, meta_final?}`. Parts contain at most 600
ticks, are stored durably, and may arrive out of order. Identical retries are
idempotent; conflicting retries, malformed data, missing parts, identity changes,
bad time spacing and path traversal are rejected. Only complete validated traces
become visible as `.jsonl` files. UTF-8 metadata is preserved.

GET `/health` is public. GET `/stats` and `/presets` require `X-Trace-Key`.
Stats aggregate rounds, successes, source counts and failure reasons per task,
excluding test fixtures. Presets expose the current source task/game/physics
JSON for the harness controller. Argon can update source configs through their
owning components, run the generator, run parity, and build through the gate:

```powershell
.venv/Scripts/python.exe roblox/tools/build.py
```

This checks freshness, syncs the current Studio window, runs real engine parity,
and builds only if that gate passes. Publishing remains an owner action.

Trace metadata contains the task, level, scene, seed, rig, physics hash, source,
success, failure reason and command intervals from the game's phase events.
The physics hash uses the same sorted JSON encoding as `r2p.sim` experts and
policy exports, so Windows line endings cannot produce a different identity.
Each 30 Hz tick includes measured `q`, absolute `q_target` labels, poses,
gripper state, phase, command and object slots. The tic-tac-toe target is bound
to the selected cell consistently across every tick. Each player move is a
separate trace. Robot answers use minimax with a 25% slip, always taking a win.
Board coordinates come directly from the canonical task preset. Tic-tac-toe
declares an 18 mm placement tolerance in its copied task and trace metadata;
the standalone placement level retains the task's tighter tolerance. Both use
the same TaskCheck implementation and settled-window requirement.

`SkillOperator` covers every skeleton skill. `collect.py` runs sequential
headless bot rounds and a temporary authenticated receiver, retaining summaries
and traces under `roblox/out/bots/`. Successful and failed attempts are recorded.
The empty-server bot cycles all six tasks; the lobby demonstration sets
`attract=true` and cannot record or upload. Collection uses no hardware, Gemini
requests, training process or GPU.

The local acceptance record is in [ACCEPTANCE.md](ACCEPTANCE.md). It distinguishes
physical bot attempts from the placement fixtures used to test the match loop.

`ServerStorage/TwinConfig.lua` is the safe server-only default. For deployment,
create the ignored `ServerStorage/LocalConfig.lua` with `TRACE_URL`, `TRACE_KEY`,
and optional `OWNER_IDS`. Set `TRACE_URL` to the base URL without `/trace`.
Never put credentials in ReplicatedStorage or a LocalScript. `/traceurl <url>`
changes the endpoint for an authorized owner. `/realarm <assetid>` replaces the
clearly labeled animated preview with a photo. The key is never accepted in chat.

## Owner publish checklist

1. Run the tests, Play smoke, concurrent-table smoke and `tools/build.py`; retain
   reports under `roblox/out/`. Re-run parity after any physics/robot change.
2. Configure a private trace key in server-only LocalConfig. Start the receiver;
   use a private HTTPS tunnel or deployed endpoint for public game servers.
   Confirm `/health`, then a complete round and its JSONL file. Keep the receiver
   bound to localhost when a local tunnel is the only public entry point.
3. Publish under the Roblox account that owns the supplied mesh assets. Open
   the generated place (or sync the existing window), then File → Publish to
   Roblox. This project does not publish automatically.
4. In Game Settings, enable HTTP requests, select Phone/Tablet/Computer/Console,
   and set maximum players to at least 8 (20 tables are provided). Enable Studio
   API access only if you deliberately want to test persistent stores; the
   default `STUDIO_DATASTORES=false` avoids accidental writes during tests.
5. Test a published private session with multiple clients. Verify the seated
   camera, touch joystick, all tasks, daily selection, bot race, unlocks and
   leaderboards. Confirm that standing/disconnecting frees a table.
6. Set the real-arm image if available; otherwise keep the animated-preview
   label. Verify robot mesh/audio asset permissions under the publishing account.
7. Review Roblox experience access, audience/device settings and the experience's
   data notice. Traces contain in-game actions and Roblox player IDs; configure
   retention on the receiver. Publish the settings, make the experience public
   when ready, and give the experience URL to the dashboard owner.

The physics preset is currently provisional; only the owner/bench workflow can
identify it against real hardware. Publishing, hardware calibration and large
multi-seed contact/grasp validation are separate from this local acceptance run.
