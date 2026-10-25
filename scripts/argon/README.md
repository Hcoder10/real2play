# Agent D: Argon brain

All coordinates are spec-frame metres; all joint targets are absolute MJCF radians.
The entry points bootstrap the repo imports, so they work from another working directory too.

## Offline acceptance

```powershell
.venv/Scripts/python.exe -m pytest -q tests/argon
.venv/Scripts/python.exe scripts/argon/tictactoe.py --dry-run --games 2
```

The dry run renders C2's CPU synthetic bench and reads its board through an explicit oracle.
It exercises game rules, projected/warped images, validation and minimax. It does not claim to
evaluate a placement policy or call Gemini. `--oracle-only` works without synthetic rendering.
The exhaustive minimax tests explore every human continuation for both possible robot marks.

## Gemini

`r2p.gemini.call(parts, role="main", tag=..., schema=...)` returns `(text, record)`.
Both roles default to `gemini-4-argon`. MAIN accepts only Argon. SMALL uses Flash only when
`R2P_MODEL_SMALL=gemini-3.8-flash` AND owner approval `R2P_ALLOW_FLASH=1` are set; the call site is
logged as a warning. See the complete [call-site table](../../README.md#how-gemini-4-argon-is-used).
The network client uses google-genai 2.33.0+ Interactions, typed inline image/video content,
`response_format` JSON schemas, `output_text`, and background create/get for long work.

`GEMINI_API_KEY` is loaded from the environment or gitignored `.env` and never sent in prompts.
`R2P_THINKING` accepts low/medium/high or legacy 0..32768 budgets (mapped to levels). Defaults are
selected by call-site tag: extraction/pointing low, board/critique medium, experiment/patch design high.
`R2P_CACHE_ONLY=1` makes zero requests and raises `CacheMiss` on a miss, including repairs/polling.
The request cache under `cache/gemini` hashes API, model, config and all input bytes; set
`R2P_GEMINI_CACHE` to change it. `calls.jsonl` records served model, attempts, latency and usage.

HTTP timeout defaults to 1200 seconds (`R2P_GEMINI_TIMEOUT_S`). HTTP 408/5xx, connection errors,
timeouts and per-minute 429s retry with 1/2/4-second backoff. SDK retries are disabled.
`Gemini(max_requests=N)` counts creates, polls, retries and JSON repair. A failed background poll
never retries the create; its interaction ID is logged for recovery. Daily/zero-limit quota raises
`QuotaError`; fallback for low-effort extraction/pointing/repair requires `R2P_ALLOW_FALLBACK=1`
as well as approved Flash. Board, critique and experiment reasoning cannot fall back to Flash.
JSON repair always uses Argon at low thinking and reuses the cached original reply.
`FakeGemini` exercises the same cache and validation without constructing a network client.

## Managed authoring stages

```powershell
.venv/Scripts/python.exe scripts/argon/author_task.py --goal "Put the cube in the bowl" --demo-summary "The open gripper releases the cube inside the bowl" --out runs/author_demo --dry-run
.venv/Scripts/python.exe scripts/argon/build_game.py --task place_on_cell --out runs/game_demo --dry-run
```

Dry runs use `FakeAgent`; the task checker, scene sampler, Luau tests and numerical parity still
run locally. They need no network or credentials. The Luau binary is discovered as described in
[roblox/README.md](../../roblox/README.md). The game gate runs trusted preset generation, all five
pure-Luau suites, and `roblox/tools/parity.py` against MuJoCo. It does not sync or alter Studio.
Returned files live in the reported `candidate-*` workspace, with `agent.json`, `input.json` and
`result.json` alongside it. Failure keeps `passed: false`. Nothing is written into the canonical
task or Roblox source directories and nothing is published. Pure optional modules are checked
for loadability; integrating them into runtime code requires review.

Remove `--dry-run` for an authorized remote Antigravity run. The agent receives only selected
source/config files, with secret values rejected before submission. It returns a JSON file bundle,
sandbox test results and a report; local verification decides acceptance independently. The game
builder edits the configuration consumed by the actual workshop. Task authoring uses the existing
closed predicate vocabulary; it supplies positive/negative single-state and stable-window vectors
rather than replacing the shared checker. The managed API exposes no thinking-level field in
`AntigravityAgentConfig`; the prompt requests high-effort reasoning with Argon as the model.

The SDK `environment={type: "remote", sources: [...]}` mounts exact UTF-8 sources under
`/workspace` using `type: "inline"`, `target` and `content`. The prompt also contains the source
context and requires measured SHA-256 values before and after tests. Missing or changed
`source_sha256` values reject the stage before candidate application. Returned hashes remain
agent-reported evidence, so independent local tests against repository originals are mandatory.
Generated task/physics modules use the requested twin inputs, not stale checked-in presets.
The first live run (before source mounting) modified its sandbox oracle and was rejected despite
passing the untouched local harness. Mounting and integrity checks are SDK-type/fake tested;
they have not had a second live run. See [the evidence report](J_INTERACTIONS_REPORT.md).

Live game generation normally requires `--verification path.json`. The receipt format is:

```json
{
  "schema": "r2p_twin_verification/1",
  "passed": true,
  "input_sha256": {"task": "...", "physics": "...", "board": "...", "camera": "..."},
  "evidence": [{"path": "runs/heldout/gate.json", "sha256": "..."}]
}
```

Input hashes use `sha256(r2p.gemini._canonical(value).encode())`; evidence hashes use raw file bytes.
Each evidence JSON must declare `passed: true` and `held_out: true`. Paths are resolved from the
current directory. Inputs include the board from the selected task and the supplied `--camera`
and `--physics`. A changed input, missing file or failed gate is rejected. For a clearly labelled
candidate-only rehearsal, `--allow-provisional-twin` permits live generation without claiming twin
fidelity. Even an accepted local build has `deployment_ready: false`: rendered-game review and
Studio contact/engine parity remain required before the owner's separate publish action.

`ManagedAgent.start/poll/collect/run` preserve interaction IDs and returned media artifact metadata;
the file contents come from the required JSON bundle, not arbitrary URL downloads. Re-running the
same prompt/output directory resumes its recorded job rather than starting another sandbox.
Completed jobs work cache-only. The default stage limit is 20 HTTP requests including polls,
with 60-second polling and a 20-minute deadline; start requests are not blindly retried because
a lost response may already have created a sandbox. `calls.jsonl` records each lifecycle operation.


## Tic-tac-toe

```powershell
.venv/Scripts/python.exe scripts/argon/tictactoe.py --frame-only frame.jpg --camera camera.json --history history.json
.venv/Scripts/python.exe scripts/argon/tictactoe.py --real --policy exports/place_on_cell --supply-xyz 0.18 -0.12 0.016 --piece-size 0.032
```

`--robot-desc` and `--human-desc` describe the pieces. `--piece-size` is metres. History is either
`[{"cell":"r0c0","mark":"X"}, ...]` or `{"board":["X","", ...]}`. Earlier marks cannot change.
One MAIN call receives the annotated real frame, perspective-warped grid, descriptions and history;
one validation re-ask is permitted. A consistent board permits perfect local minimax fallback. If
the current human move cannot be established, the live loop requires its operator confirmation.

Live mode uses the already-running C1 daemon, a verified joint map, measured supply coordinates,
the calibrated camera and a `place_on_cell` export. Each placement is its own C1 `Session`, starts
from the export's pose and records a trace plus camera calibration. Failed placements stop the game.
The known supply projection seeds perception; no inferred supply coordinate is used.
`out/ttt/status.json` is atomically updated with `argon_says`, move, board, model and status.

## Task authoring and scene fitting

```powershell
.venv/Scripts/python.exe scripts/argon/task_from_video.py runs/robot/attempt --out-live out/live/example --iters 6
.venv/Scripts/python.exe scripts/argon/task_from_video.py episode.mp4 --instruction "put the cube in the bowl" --no-twin
.venv/Scripts/python.exe scripts/argon/run_twin.py runs/robot/attempt --task cube_in_bowl --out out/live/example2
```

A recorded session contains `trace.jsonl` (`r2p_trace/1`), `frames/`, and `camera.json`. Tick `frame`
references are authoritative. Without them, use one frame per tick or explicit `frame_times` /
`frame_fps` in metadata; mismatched video/joint clocks are never guessed. The loader also accepts
bare MP4/MOV/WebM/AVI videos and images (a single image requires an instruction).

Task inference receives the whole video, first/last frames, a 5 Hz joint table and measured anchor
facts. It validates through `r2p.tasks.validate`, re-asks at most twice with exact errors, fills
randomization locally and pins badly mismatched grasp width. It writes `tasks/<name>/task.json`.
A calibrated episode with at least two frames then enters the twin loop and writes `scene.json`.
A bare uncalibrated video produces a task with `scene_status=requires_calibrated_camera`; metric
geometry is not claimed without calibration. A video without joints cannot validate robot physics.
Generated poses carry `pose_source=model_inferred`, then `calibrated_twin` after fitting. Perception
does not treat either as a surveyed location without `pose_verified=true`.

The loop uses one SMALL detection on the first critic frame, local Lab appearance masks and
calibrated silhouette fitting. `TaskTwin` replays **recorded q_target**, and C2's CPU mesh renderer
renders its joint/object state through the real camera. No CUDA, GPU lease or graphics context is
needed. The first and subsequent critic images never include held-out evaluation frames. Local
metrics include per-object IoU, centroid pixels, trajectory pixels, camera keypoints when supplied,
and grasp/release anchors. A missing camera-keypoint measurement stays null; calibration is locked.

Gripper commands define close/release times and the most-closed empty reference. Settled measured
jaw gaps establish grasp width; an empty close is not a grasp. `--anchors auto` enforces only
image-consistent anchors (within 25 mm); `hard` forces enforcement and `soft` reports them only.
Widths are constrained to ±20% of measured gap. Candidate poses remain scoreable so regressions
are rejected, not hidden. The keep rule requires gain >0.003, per-object IoU no more than 0.10 below
its accepted best, container loss no more than 0.02 without enforced anchors, camera error no worse,
and, with enforced anchors, grasp error no worse and release inside the container. Two unsuccessful
iterations stop the loop. Invalid patches are rejected. Quota/cache-only stops retain the best scene.
Transport failures and request exhaustion also retain the incumbent. Real masks are tracked once
independently of proposed scenes. Sensor-clipped objects use clipped predicted silhouettes.

Static calibration collections may carry measured `meta.pose_samples` and explicit `frame_times`
with an empty command trace. These render measured poses directly and report
`physics.mode=static_pose_samples`, `replayed=false`, and null trajectory error. Presentation times
are not motion evidence; no command targets are invented. Overlay panels preserve image aspect.

`out/live/<run>/` contains atomic `r2s_live/1` `run.json`, each `iter_N.json`, candidate scene JSON,
1280×480 `iter_N_overlay_J.jpg`, `iter_N_physics.jpg`, and `scene_best.json`. Existing run directories
are never overwritten. The dashboard's per-model request counts include retries and repairs.

## Labeling and system identification

```powershell
.venv/Scripts/python.exe scripts/argon/label_trace.py episode.jsonl labeled.jsonl --task cube_in_bowl
.venv/Scripts/python.exe scripts/argon/sysid.py plan residuals.json
.venv/Scripts/python.exe scripts/argon/sysid.py fit train1.jsonl train2.jsonl --heldout heldout.jsonl --physics so101_bench
```

`segment(trace, task)` returns contiguous `[t0,t1,command]` intervals from existing command labels,
gripper commands and motion phases. Optional `--refine` asks MAIN for typed integer tick intervals
and allows one semantic correction. Invented vocabulary, gaps and overlaps are rejected before
conversion to seconds. `label_trace` writes the shared trace format, including meta.commands.

The planner prioritizes residual/tolerance, proposes bounded absolute joint excitation and resettable
push/grasp/drop probes, and never executes hardware. `fit_parameters(train, physics, heldout=...)`
fits bounded servo gains/friction against replay trajectories, searches integer latency, and fits
runner `sag_gain` separately from static holds because TaskTwin derives sag from finite stiffness.
Default object friction is applied to replayed dynamic objects; per-object overrides can be fitted
with parameter paths such as `objects.cube.friction` when present in the supplied physics dictionary.
Focused `Parameter(path, low, high)` sets avoid fitting parameters the experiment does not excite.

Whole training/held-out traces must differ. Rank-deficient fits, unconverged optimizers, absent evidence,
and insufficient static holds fail certification. Held-out gates are jaw RMS <2 mm, maximum object
endpoint error <5 mm, physical opposing-jaw grasp agreement ≥95%, and timing error <1 tick. Grasp
success requires opposing physical jaw contacts across three ticks. `--output NEW.json` publishes
only a passing new version; it never overwrites an incumbent. Physical benchmark accuracy still
requires recorded hardware trials. Live Argon measurements are in [Agent G's report](G_LIVE_QA_REPORT.md).

## Visual joint calibration

```powershell
.venv/Scripts/python.exe scripts/argon/visual_calib.py plan --out runs/visual_plan
.venv/Scripts/python.exe scripts/argon/visual_calib.py collect --plan runs/visual_plan/plan.json --adapter bench_adapter:make --out runs/visual_capture
.venv/Scripts/python.exe scripts/argon/visual_calib.py point --poses runs/visual_capture/poses.json --out runs/visual_fit
.venv/Scripts/python.exe scripts/argon/visual_calib.py fit --poses runs/visual_fit/pointed.json --out runs/visual_fit
```

`bench_adapter:make` is an operator-provided adapter, not a bundled hardware connection. Its factory
returns `{mover, capture, close?}`. `mover(spec_xyz)` returns `{ok, owner_deg, settled_at}` after guarded
motion and settling; `capture(after_timestamp)` returns `(BGR_image, captured_at)` on the same clock.
Collection rejects stale captures and saves every completed pose. No serial port or daemon is opened
by this feature. The plan proposes 18 above-table poses; reachability remains the adapter's responsibility.

Pointing resumes saved observations and uses the shared cache. Soft-L1 fitting handles swapped tips
and independently refits every leave-one-pose-out fold. `fit.json` and `report.md` include rank,
convergence, reprojection errors and the default 5 px p90 gate. Candidates never activate themselves.

## Read-only live QA

`live_qa.py` provides `prepare`, `empty`, `boards`, `twin`, `task`, `visual`, `perception`, `label`, and
`summary` stages. Use one root sequentially: `request_budget.json` reserves every attempt before
network I/O, `qa.lock` excludes concurrent stages, and `calls.jsonl` preserves the audit. The default
cap is 80 requests across invocations. `empty` performs a camera GET only; other stages use files.
`compose_boards.py` builds reproducible photo composites with independent expected-board labels.
See the report for exact commands and the distinction between photos, composites and mock traces.
