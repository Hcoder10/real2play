# Agent N: autonomous Argon orchestration

`python -m r2p argon` connects the existing scene, calibration, active-twin,
managed-game, headless-play, replay/training and robot components through a
closed, schema-validated tool loop. Interactions uses `gemini-4-argon`, high
thinking, native function calls and `previous_interaction_id`; measured results
return as matching `function_result` messages. The ledger records decisions,
approvals, model records, tool metrics and failures. `status.json` is atomic.

## Commands

```powershell
.venv/Scripts/python.exe -m r2p argon --task place_on_cell --dry-run --approve auto
.venv/Scripts/python.exe -m r2p argon --task place_on_cell --plan-only --request-budget 1
.venv/Scripts/python.exe -m pytest -q tests/core tests/argon tests/perception tests/robot tests/app tests/sim tests/policy
```

Each invocation gets a fresh directory under `runs/argon_orchestrator/`. `--run`
sets its name; existing evidence is never overwritten. `--max-steps` bounds the
controller loop. Missing/invalid results cannot satisfy finish. Repeating an
upstream tool invalidates dependent gates, including when its new execution fails.

Live execution uses `--approve ask`. Every motion, managed compute, model-assisted
measurement and training/publication request asks for the exact action and logs
the answer. `--approve auto` is rejected outside dry mode. The controller cannot
change thresholds or provide approval arguments. No GPU is launched by this code.

Live inputs: `--poses <poses.json>` with at least three open-jaw views,
`--scene <measured_scene.json>`, `--from-game <receiver-directory>`,
`--perception-setup <setup.json>` and `--bench-profile <stream-profile.json>`.
Calibration operates on recorded images and keeps its map inactive. Real policy
execution uses the existing streaming daemon, never reconnects serial, never
overrides an unverified map, and records one placement.

The twin gate requires held-out joint RMS <=1 degree, jaw RMS <=2 mm, object
endpoint error <=5 mm and IoU >=0.80, including each experiment and per-object
visual non-regression. Calibration requires <=5 px held-out p90. Frozen task,
physics, scene and camera hashes must still match before downstream execution.
These gates do not establish independent real-world transfer.

`build_game` runs the managed builder and all its local Luau/parity checks.
With `publish=true`, it writes `publish_request.json` and waits for the owner's
`publish_receipt.json` containing `request_sha256` and the published Roblox
`url`. The owner publishes the verified workspace; no tunnel or game is published
by this implementation. Publication remains a required live completion gate.

Live replay creates `train_request.json` containing the frozen inputs, accepted
trace hashes, expert paths and validated config. The lead rents a landlord lease
and runs training. `train_receipt.json` must contain the matching
`request_sha256`, `export` and `incumbent` paths. The orchestrator independently
loads and evaluates both exports on the same held-out clean/real-noise scenes.
Real deployment requires improvement over incumbent, >=0.80 success in both
conditions and >=12 scenes. `--wait-seconds` controls receipt wait duration
(default 3600 seconds); expiry leaves an explicit BLOCKED status and preserves
the request without resubmitting it. Keep the process running while supplying
the receipt: recovery of a terminated orchestration process is not implemented.
Both receipt hashes use `sha256(r2p.gemini._canonical(request).encode())`.

## Managed compute

`build_game` and optional `author_task` reuse agent J's Antigravity wrappers.
`analyze_experiments` mounts training residuals, training command recordings,
real|sim image summaries, immutable twin code, physics and (live mode) robot
assets. It returns bounded relative stiffness/damping and absolute material
parameters. The managed sandbox runs physical replay tests; local code validates
the patch and scores held-out recordings that were never sent to the sandbox.
A non-improving patch is rejected and the incumbent retained. Arbitrary returned
Python/Luau files, modified source hashes and reported failed tests are rejected.

`train_plan` returns a training config and tests. Local verification binds the
data count, task/physics hashes, CPU/lead-leased-GPU mode, 30 Hz and absolute
15x6 targets. The ARGON dashboard strip shows managed agent IDs, durations,
check counts and synthetic provenance, alongside the current decision/metrics.

## Measured dry-run evidence

Run `runs/argon_orchestrator/n-acceptance-01` completed in **84.090 seconds** with fake
Argon decisions, fake managed agents, real CPU MuJoCo/SimMover, real Luau tests,
headless game control, physical replay, tiny CPU learning and MockRobot.
No live Gemini, GPU or hardware execution occurred in this run.

```text
ARGON calibrate_arm: passed=1, poses=8, open_jaw_frames=8, activated=0
ARGON active_twin: experiments=4, object_mm=0.190132, iou=1.0
ARGON analyze_experiments: passed=1, kept=0
ARGON twin_gate: passed=1, joint_deg=0, jaw_mm=0, object_mm=0.190132
ARGON build_game: passed=1, local_checks=4, parity_passed=1, published=0
HEADLESS 1/1: success=True, ticks=390
ARGON collect_play: traces=1, human=0, headless=1
ARGON replay_and_train: replayed=1, eval_n=1, clean=0, real=0, improved=0
ARGON deploy_real: attempts=1, ticks=60, success=0
ARGON finish: gates_passed=1, stop_met=1
```

The smoke policy failed; both paired scores were zero and selection was HOLD.
DONE here means the offline workflow and its explicit smoke criteria completed.
It is not policy qualification, human gameplay, real calibration or real transfer.
Earlier independent CLI run `n-dev-03` took 74.316 seconds; test-isolated end-to-end
runs also passed the three-minute limit. The accepted twin gate additionally
requires IoU >=0.80 for every object, not only the average.

## Live Argon planning evidence

Run `runs/argon_orchestrator/n-live-plan-01`: one uncached Interactions request,
served by `gemini-4-argon`, high thinking, **47.360 seconds**, no motion or tool
execution. Argon returned this native function call:

```json
{
  "tool": "measure_scene",
  "args": {},
  "why": "Measure the board grid, supply caps, and verify camera freshness before calibrating the arm.",
  "done_when": [{"value": 1, "metric": "measure_scene.ok", "op": "eq"}]
}
```

This is the next action chosen from current recorded bench status, not evidence
that the bench has been freshly measured. Full input/response records stay in
the run directory. The live smoke used 1 of the authorized 15 requests.

## Optional managed controller and MCP

The default controller stays on local Interactions so it can reach the laptop's
camera/daemon directly. `scripts/argon/tools_mcp.py` additionally exposes the
same closed tools over authenticated, loopback-only MCP Streamable HTTP. Requests
are serialized; duplicate request IDs replay their response instead of repeating
motion. The server rejects untrusted browser origins and requires a bearer token.
It follows the [MCP JSON-response transport](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports).

The owner may start a tunnel themselves in a separate terminal:

```powershell
cloudflared tunnel --url http://127.0.0.1:8798
```

Using the resulting HTTPS hostname, the owner then runs:

```powershell
.venv/Scripts/python.exe -m r2p argon --task place_on_cell --controller managed --mcp-url https://OWNER-HOST.trycloudflare.com/mcp --approve ask --dry-run
```

This starts only the local MCP server and the approved Antigravity controller;
it never starts a tunnel. The ephemeral bearer token is delivered as an MCP
authentication header, never printed or put in a prompt. `--approve ask` is
required even in synthetic managed-controller mode because starting the remote
controller spends API credits. Remove `--dry-run` only with the live input flags
and owner present. Every motion still needs its separate local terminal approval.

For another MCP client, run the tools server alone (token file stays local):

```powershell
.venv/Scripts/python.exe scripts/argon/tools_mcp.py --dry-run --approve auto --token-file runs/argon_mcp_token.txt
```

Local MCP initialization, authentication, origin rejection and duplicate-request
handling are tested. No public tunnel or live managed-controller run was made.

## Final validation and commits

- Primary implementation: `1439a53` (`feat(argon): orchestrate gated twin-to-play-to-policy tools`).
- Follow-up commit containing this report: `fix(argon): stop pending owner requests and require publication evidence`.
- Lightweight boundary checks: **25 passed, 1 deselected** in 1.03 seconds.
- Required full command above: **995 passed, 8 failed** in 524.85 seconds. All **26 N
  orchestrator tests passed**, including the physical offline loop. That isolated
  end-to-end run completed in **92.576 seconds**, with DONE/synthetic status.

Full acceptance is **not green**. The final shared-tree run reported these failures:

| Component | Failing test | Observed result |
|---|---|---|
| active twin | `test_unsafe_experiments_refused[experiment3]` | Current validator accepts the high hold above the grid. |
| active twin | `test_whole_path_bounds_and_safe_finish` | Current menu entry 2 is accepted with the fixture's extra obstacle. |
| active twin | `test_sim_campaign_recovers_hidden_truth_without_fit_leak` | Friction remains 0.60; fixture expects 0.45 +/-0.06. |
| active twin | `test_menu_follows_moved_supply_and_scene_freezes_grid` | Hold follows y=-0.125; fixture expects -0.12. |
| perception | `test_synthbench_pose_accuracy[2-place_on_cell]` | Fixed-cell error 18.86 mm. |
| perception | `test_synthbench_pose_accuracy[4-place_on_cell]` | Fixed-cell error 30.71 mm. |
| perception | `test_synthbench_pose_accuracy[8-place_on_cell]` | Fixed-cell error 47.69 mm. |
| simulation | `test_trace_format_commands_dart_and_determinism` | Generated expert trace reports success=false. |

Other agents changed the active experiment menu and measured task geometry during
this work. N did not edit those components or relax their tests. These failures
need their owners' resolution before claiming repository-wide acceptance.

The perception fixture samples a randomized board but supplies nominal fixed-cell
coordinates to `TaskPerceiver`. An isolated in-memory probe supplying sampled
fixed-cell coordinates passed all three failing cases (3 passed, 21 deselected).
The source fix would touch C2's `r2p/synthbench/world.py`; permission was requested
under AGENTS.md's ownership rule and had not been granted at handoff. No source
change was applied outside N's paths. The probe is retained in
`runs/argon_orchestrator/n-perception-probe.py`.

## Files and limitations

Core: `r2p/orchestrator.py`, `r2p/orchestrator_tools.py`,
`r2p/orchestrator_managed.py`, minimal `r2p/__main__.py` dispatch.
UI: `dashboard/argon.py`, `dashboard/static/argon.{js,css}` and the read-only
route/static injection in `dashboard/server.py`. Existing agents' dirty CLI,
dashboard index/app and Gemini sources were not edited.
Acceptance coverage: `tests/argon/test_orchestrator.py`.

The ARGON strip was visually checked in Chrome against an isolated dashboard
preview on port 8808; the recorded live plan and reason were visible above the
existing four panels. The existing dashboard service was not restarted.

Real experiment execution, live managed analysis/training-plan sandboxes,
publication, human data, leased GPU training and a successful physical policy
placement remain owner/lead actions and have not been demonstrated by N.
