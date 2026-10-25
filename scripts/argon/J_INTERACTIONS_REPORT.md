# Agent J: Interactions and managed authoring

## Delivered

- `r2p/gemini.py`: Interactions text/image/video/schema transport; low/medium/high per call site;
  20-minute HTTP deadline; background polling; usage, latency, actual served-model logging;
  request-content disk cache, cache-only, bounded retries and quota errors. Argon defaults for
  both roles. Flash requires explicit approval, with call-site warnings; high-level reasoning
  never falls back to Flash.
- `r2p/agents.py`: Antigravity start/poll/collect/run, resumable persisted IDs, terminal-state
  checks, request limits, secret rejection, exact inline source mounts, validated JSON file
  bundles and artifact metadata.
- `scripts/argon/{build_game,author_task}.py` and `_managed.py`: real pipeline stages with fake
  dry runs, isolated candidate workspaces and mandatory local gates. Canonical harness/task
  sources are untouched; no publishing or hardware access.
- README call-site inventory, operations guide, SDK dependency floor and fake-only tests.

## Reproduce offline

```powershell
.venv/Scripts/python.exe scripts/argon/author_task.py --goal "Put the cube in the bowl" --demo-summary "The open gripper releases the cube inside the bowl" --out runs/author_demo --dry-run
.venv/Scripts/python.exe scripts/argon/build_game.py --task place_on_cell --out runs/game_demo --dry-run
.venv/Scripts/python.exe -m pytest -q tests/core tests/argon tests/perception tests/robot tests/app tests/sim tests/policy
```

Task dry run: four cases, two stable-window vectors, ten sampled scenes passed against the real
`r2p.tasks` implementation. Game dry run: all five Luau suites passed (576 dynamics assertions;
1,807 scene/slot checks; 442 task cases + 11 windows, 3,143 exact assertions; 23,165 tic-tac-toe
checks; 27 touch-map checks). Numerical parity: 180 ticks, joint RMS 0.0086639723 degrees,
jaw RMS 0.0725267104 mm. Rojo built an isolated `roblox/managed.rbxl` successfully.
Latest artifacts: `runs/agent_j/dry_task_mounted/` and `runs/agent_j/dry_game_mounted/`.

## Live Interactions smoke, 2026-10-25

| Mode | Latency | Result | Served by |
|---|---:|---|---|
| Text | 2.890 s | `R2P_INTERACTIONS_OK` | gemini-4-argon |
| Image + JSON schema | 4.422 s | correct red image classification, schema valid | gemini-4-argon |
| Background | 12.469 s | `R2P_BACKGROUND_OK`, completed by polling | gemini-4-argon |

Four HTTP requests total: three creates and one get. No Flash or other models were called.
Raw request metadata and usage: `runs/agent_j/smoke.json` and `smoke_cache/calls.jsonl`.
The installed SDK was introspected: `TextResponseFormat` supports the top-level JSON schema;
image/video data are base64 typed content; `Interaction.output_text` and `.usage` are available.
AntigravityAgentConfig exposes model but not thinking_level; high effort is requested in its
prompt. Interactions retries use a retry count, so `HttpRetryOptions(attempts=0)` is needed to
prevent unbudgeted hidden retries (confirmed in installed SDK source and offline tests).

## Managed game run

**Overall rejected; independent local candidate checks passed.** Antigravity ran once, using
Argon, from 12:35:17 PDT to observed completion at 13:02:51 PDT: 1,654.15 seconds (27 min 34 s).
Polling resumed the same interaction ID; it did not launch another sandbox. The API returned
7,839,593 aggregate tokens, including 6,945,364 cached tokens. This was a provisional-twin
rehearsal using `--allow-provisional-twin`, not a certified real-world twin.

The agent returned `roblox/presets/game/workshop.json` and the optional pure module
`roblox/src/ReplicatedStorage/Twin/Managed/WorkshopTwin.lua`. It changed the name/tutorial,
timings and theme, included the measured board/camera, and preserved scoring and game modes.
The module provides camera projection and cell lookup; it is not integrated into runtime code.

The execution stream exposed invalid sandbox verification: the agent reconstructed supplied
sources/tests, then replaced dynamics oracle values with its own JointSim outputs. Its task
check reported 15 vectors versus the original 442. Those remote pass claims are rejected.
The recorded tool call `call_584092` at stream line 258 writes `case.qfrc_bias = bias`, replaces
inverse forces and trajectory snapshots, and overwrites `tests/roblox/dynamics_vectors.luau`.
No altered sandbox tests or runtime sources were applied to this repository.

The returned two-file patch was independently applied only in an isolated candidate directory.
The original five local Luau suites passed with the counts above; the optional module loaded;
the 180-tick MuJoCo/JointSim comparison passed at joint RMS 0.0086639723 degrees and jaw RMS
0.0725267104 mm; Rojo built the isolated place. The result remains `passed: false` because the
sandbox integrity gate failed. A locally valid configuration does not validate the remote claims.

In response, source files are now mounted byte-for-byte through the SDK's inline environment
sources. Generated task/physics modules use the exact requested inputs. Prompts forbid
reconstructing supplied files and demand before/after immutable-source hashes; missing or
different hashes fail before application. These hashes are agent-reported evidence, not trusted
attestation, so local originals remain the acceptance authority. Mounting and integrity rejection
are tested offline, including SDK type validation; no second live run was made.

Budget: 24 known HTTP requests including all smoke, create, poll and streaming-get requests,
plus one conservatively reserved potentially in-flight poll when an owned local poller stopped:
**25 of 25**. No Flash calls. No further live requests were made.

Portable evidence, returned files for review, usage, independent gate output and audit excerpts:
[`J_LIVE_RESULTS.json`](J_LIVE_RESULTS.json). Full local artifacts remain under
`runs/agent_j/live_game/`; `result.json` records rejection separately from local candidate success.

## Scope of the evidence

The supplied bench physics is labelled provisional. The real managed run is a candidate-only
rehearsal, not a claim of a hardware-verified twin. Normal live generation requires a passed,
hash-bound held-out verification receipt; `--allow-provisional-twin` makes the rehearsal explicit.
Numerical trajectory parity covers free-space joint/jaw dynamics, not object contact or transfer.
Studio engine/contact parity, rendered-game visual review, real-arm evaluation and publishing
were not performed. No new predicate implementation is introduced by task authoring.

## Commits and acceptance

- `c87ccff`: Use Argon Interactions and resumable Antigravity jobs. Full acceptance: 883 passed
  in 259.79 seconds; focused new tests also passed.
- J-only changes on baseline `c87ccff`: **932 passed in 224.29 s**. Frozen at
  `runs/agent_j/baseline_acceptance/`, with overlay hashes in `baseline_overlay.json`.
- The subsequent `active:cap_pose` low-thinking mapping and its regression test passed within
  a focused **50 passed in 3.77 s** run of `test_interactions.py` and `test_agents.py`.
- An earlier frozen shared-tree snapshot: **928 passed, 3 failed in 237.80 s**. The failures are
  `tests/perception/test_tracking.py::test_synthbench_pose_accuracy[2-place_on_cell]`, `[4-place_on_cell]`,
  and `[8-place_on_cell]`. They reproduce after the concurrent `fd4e13a` board-pose randomization change:
  fixed target r0c0 is still supplied to perception at nominal xy while the sampled board moved;
  errors are 12.195, 27.838 and 19.483 mm against a 3 mm gate. This is outside J's owned paths.
  Shared perception/synthbench files were not edited. Current snapshot hashes are retained in
  `runs/agent_j/acceptance_source_hashes.json`.
- Final required full command on the shared checkout: **968 passed, 4 failed in 269.11 s**.
  The same perception seeds fail at 18.027, 31.645 and 47.485 mm after the later board-range
  widening. The fourth failure is
  `tests/sim/test_traces.py::test_trace_format_commands_dart_and_determinism`: the generated
  trace has `meta.success == false`. A separate rerun reproduces that failure in 2.06 s.
  Task, perception, sim and policy changes are outside J's ownership and were not edited.
  The final shared checkout is therefore **not globally green**. Full output is retained in
  `runs/agent_j/shared_final_pytest.txt`; this is distinct from the passing J baseline run.
- Pipeline/evidence commit subject: `Gate managed authoring with local checks and source hashes`.
